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


class FixedWindowRateLimiter:
    """In-process fixed-window limiter keyed by client IP address.

    Single-process by design: it cannot see requests served by other
    worker processes. Production multi-worker deployments must front the
    API with a shared/external rate limiter or API gateway. Client IP is
    taken from the socket peer; ``X-Forwarded-For`` is intentionally NOT
    trusted (it is client-controlled).
    """

    def __init__(self, limit: int = 60, window_seconds: int = 60) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window must be positive")
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._lock = threading.Lock()
        self._counts: Dict[str, Tuple[float, int]] = {}

    def allow(self, client_ip: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            if len(self._counts) > 4096:  # bound memory
                self._counts.clear()
            start, count = self._counts.get(client_ip, (now, 0))
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count >= self.limit:
                retry_after = int(self.window_seconds - (now - start)) + 1
                return False, max(1, retry_after)
            self._counts[client_ip] = (start, count + 1)
            return True, 0