"""Request middleware for the stdlib inference API.

Implements the request-protection layer a web framework would otherwise
provide out of the box:

* correlation / request ids (a client ``X-Request-ID`` is respected when
  it is a safe token; otherwise one is generated and echoed back)
* structured, security-conscious logging (explicit allow-list; never API
  keys, telemetry payloads, model internals, or secrets)
* payload size and content-type limits (server-enforced before parsing)
* a basic in-process fixed-window rate limiter (documented as a
  single-worker measure, not a substitute for a shared gateway limiter)
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Dict, Tuple

MAX_BODY_BYTES = int(
    os.environ.get("GRIDSENTINEL_MAX_BODY_BYTES", "65536")
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_logger = logging.getLogger("gridsentinel.api")

_LOG_KEYS = (
    "ts",
    "level",
    "correlation_id",
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
    "client_ip_hash",
    "principal",
    "error_category",
    "bytes",
    "limits",
)


def configure_logging() -> None:
    """Attach a single JSON-line handler to the module logger."""
    if _logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s gridsentinel.api %(message)s")
    )
    _logger.addHandler(handler)
    try:
        _logger.setLevel(os.environ.get("GRIDSENTINEL_LOG_LEVEL", "INFO").upper())
    except ValueError:
        _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_event(event: Dict) -> None:
    """Emit a single-line structured JSON log record (allow-list only)."""
    payload = {k: event[k] for k in _LOG_KEYS if k in event}
    try:
        _logger.info(json.dumps(payload, sort_keys=True))
    except Exception:
        _logger.error("log emission failed")


def correlation_id(headers) -> str:
    """Server-generated correlation id, unless a safe client token is given."""
    value = headers.get("X-Request-ID")
    if value and _SAFE_TOKEN.match(value):
        return value
    return uuid.uuid4().hex


def client_ip_hash(address: str) -> str:
    digest = hashlib.sha256(str(address).encode("utf-8")).hexdigest()
    return digest[:12]


_RATE_MEMORY_BOUND = 4096
_RATE_SWEEP_INTERVAL = 5.0


class FixedWindowRateLimiter:
    """In-process fixed-window limiter keyed by client IP address.

    Single-process by design: it cannot see requests served by other
    worker processes. Production multi-worker deployments must front the
    API with a shared/external rate limiter or API gateway. Client IP is
    taken from the socket peer; ``X-Forwarded-For`` is intentionally NOT
    trusted (it is client-controlled).

    Memory stays bounded: expired windows are swept periodically, and if
    the tracked-client map still exceeds ``_RATE_MEMORY_BOUND`` entries the
    oldest windows are evicted. Eviction never clears live windows.
    """

    def __init__(self, limit: int = 60, window_seconds: int = 60) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window must be positive")
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._lock = threading.Lock()
        self._counts: Dict[str, Tuple[float, int]] = {}
        self._last_sweep = time.monotonic()

    def _sweep_expired(self, now: float) -> None:
        stale = [
            ip
            for ip, (start, _count) in self._counts.items()
            if now - start >= self.window_seconds
        ]
        for ip in stale:
            del self._counts[ip]

    def _trim_oldest(self) -> None:
        while len(self._counts) > _RATE_MEMORY_BOUND:
            oldest = min(self._counts.items(), key=lambda item: item[1][0])[0]
            del self._counts[oldest]

    def allow(self, client_ip: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_sweep >= _RATE_SWEEP_INTERVAL:
                self._last_sweep = now
                self._sweep_expired(now)
            start, count = self._counts.get(client_ip, (now, 0))
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count >= self.limit:
                retry_after = int(self.window_seconds - (now - start)) + 1
                return False, max(1, retry_after)
            self._counts[client_ip] = (start, count + 1)
            if len(self._counts) > _RATE_MEMORY_BOUND:
                self._trim_oldest()
            return True, 0