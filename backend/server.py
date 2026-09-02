"""Zero-dependency HTTP serving layer for the GridSentinel inference API."""

import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from . import API_VERSION, SERVICE_NAME
from . import errors as err
from . import schemas
from .auth import (
    API_KEY_HEADER,
    hashed_principal,
    is_configured,
    verify as verify_api_key,
)
from .middleware import (
    MAX_BODY_BYTES,
    FixedWindowRateLimiter,
    client_ip_hash,
    configure_logging,
    correlation_id,
    log_event,
)


# ---------------------------------------------------------------------------
# Engine bootstrap (lazy, thread-safe)
# ---------------------------------------------------------------------------
# Importing ai.ml.risk_engine_v2 runs V2 artifact manifest verification
# (fail closed) and loads the model (torch/numpy). That work - and the heavy
# ai.ml.artifact_guard dependency it pulls in - is deferred until the first
# authenticated, validated assessment, so /health and client-side 4xx
# failures never require the model to be resident and the accept loop is
# reachable almost immediately after process start.

_ENGINE: Dict[str, Any] = {}
_ENGINE_LOCK = threading.Lock()


def _get_engine():
    with _ENGINE_LOCK:
        existing = _ENGINE.get("fn")
        if existing is not None:
            return existing
        # Manifest verification happens here, inside the guarded import.
        from ai.ml.risk_engine_v2 import assess_risk_v2

        _ENGINE["fn"] = assess_risk_v2
        return _ENGINE["fn"]


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
# SIGTERM (systemd stop, ``kill``, ``subprocess.terminate()``) stops the
# accept loop, drains in-flight handler threads, and lets the process exit
# cleanly. http.server.shutdown() must be called from a thread OTHER than the
# one running serve_forever(), so a lightweight watcher performs it once the
# signal flag is raised. SIGINT keeps its normal KeyboardInterrupt path.

_SHUTDOWN_REQUESTED = threading.Event()


def _request_shutdown(signum, frame) -> None:
    _SHUTDOWN_REQUESTED.set()


class GridSentinelServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class GridSentinelHandler(BaseHTTPRequestHandler):
    server_version = "GridSentinel"
    sys_version = ""

    # -- plumbing ----------------------------------------------------------

    def _write_json(self, status: int, payload: Dict[str, Any],
                    headers: Dict[str, str] | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", self._correlation_id)
        self.send_header("Server", "GridSentinel")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # Silence the base logger; access/error records are emitted
        # through the structured logger only.
        return

    def version_string(self) -> str:
        # Avoid leaking the Python runtime version in the Server header.
        return self.server_version

    def _log_record(self, *, status: int, error_category: str = "",
                    duration_ms: float = 0.0) -> None:
        log_event(
            {
                "ts": time.time(),
                "level": "warn" if status >= 400 else "info",
                "correlation_id": self._correlation_id,
                "status": status,
                "duration_ms": round(duration_ms * 1000.0, 2),
                "client_ip_hash": client_ip_hash(self.client_address[0]),
                "principal": hashed_principal(),
                "error_category": error_category,
                "bytes": self._bytes_read,
            }
        )

    # -- dispatch ----------------------------------------------------------

    def do_GET(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        if self.path == "/health":
            self._health()
        else:
            self._fail(err.not_found())

    def do_HEAD(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        self._fail(err.method_not_allowed())

    def do_PUT(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        self._fail(err.method_not_allowed())

    def do_PATCH(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        self._fail(err.method_not_allowed())

    def do_DELETE(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        self._fail(err.method_not_allowed())

    def do_OPTIONS(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        self._fail(err.method_not_allowed())

    def do_POST(self) -> None:
        self._correlation_id = correlation_id(self.headers)
        self._bytes_read = 0
        if self.path == "/v1/assess":
            self._assess()
        else:
            self._fail(err.not_found())

    # -- health ------------------------------------------------------------

    def _health(self) -> None:
        started = time.monotonic()
        payload = {"status": "ok", "service": SERVICE_NAME, "api_version": API_VERSION}
        self._write_json(200, payload)
        self._log_record(status=200, duration_ms=time.monotonic() - started)

    # -- assessment --------------------------------------------------------

    def _assess(self) -> None:
        started = time.monotonic()
        try:
            # ---- payload size guard ------------------------------------
            try:
                length = int(self.headers.get("Content-Length", "") or "0")
            except ValueError:
                raise err.bad_request(
                    "Content-Length header must be an integer."
                )
            if length < 0:
                raise err.bad_request(
                    "Content-Length header must be non-negative."
                )
            if length > MAX_BODY_BYTES:
                raise err.payload_too_large(MAX_BODY_BYTES)
            self._bytes_read = length

            # ---- content-type guard ------------------------------------
            content_type = (
                self.headers.get("Content-Type") or ""
            ).split(";")[0].strip().lower()
            if content_type != "application/json":
                raise err.unsupported_media_type()

            # ---- JSON parse (NaN/Infinity tokens rejected) -------------
            def _reject_nonfinite_constant(value: str):
                raise ValueError("non-finite JSON constant")

            raw = self.rfile.read(length)

            try:
                payload_obj = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=_reject_nonfinite_constant,
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                raise err.invalid_json()

            # ---- authentication (header only; never from the body) -----
            if not is_configured():
                raise err.server_misconfigured()
            if not verify_api_key(self.headers.get(API_KEY_HEADER, "")):
                raise err.unauthorized()

            # ---- rate limiting (after auth, before inference) ----------
            allowed, retry_after = self.server.limiter.allow(
                self.client_address[0]
            )
            if not allowed:
                raise err.rate_limited(retry_after)

            # ---- strict schema validation --------------------------------
            try:
                telemetry = schemas.validate_request(payload_obj)
            except schemas.SchemaValidationError as exc:
                raise err.validation_failed(exc.field_errors)

            # ---- inference ----------------------------------------------
            # trusted_source is SERVER-side policy only and is always False
            # in this phase: an authenticated caller is not proof of sensor
            # provenance. A client can never set or influence it.
            engine = _get_engine()
            result = engine(telemetry, trusted_source=False)

            envelope = self._envelope(self._correlation_id, result)
            self._write_json(200, envelope)
            self._log_record(status=200, duration_ms=time.monotonic() - started)
        except Exception as exc:
            # ai.ml.artifact_guard (torch/numpy) is imported lazily so the
            # process can accept connections without the ML stack resident.
            from ai.ml.artifact_guard import TelemetryValidationError

            if isinstance(exc, TelemetryValidationError):
                # Defensive: the engine's own guard flagged something the
                # schema layer did not. Never echo engine exception text.
                self._fail(
                    err.validation_failed(
                        {"telemetry": "Telemetry failed engine-level validation."}
                    ),
                    started=started,
                )
            elif isinstance(exc, err.ApiError):
                self._fail(exc, started=started)
            else:
                # Never leak tracebacks, exception text, paths, or state to
                # the client. Full traceback is emitted to the log only.
                logging.getLogger("gridsentinel.api").exception(
                    "unhandled_api_exception"
                )
                self._fail(err.internal_error(), started=started)

    @staticmethod
    def _envelope(request_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        alert_state = result.get("alert_state", "NORMAL")
        trust_boundary_applied = bool(result.get("trust_boundary_applied", False))
        investigation_required = bool(
            alert_state == "INVESTIGATION" or trust_boundary_applied
        )
        return {
            "request_id": request_id,
            "api_version": API_VERSION,
            "status": "ok",
            "result": {
                "risk_classification": result.get("risk_level", "LOW"),
                "prediction": result.get("prediction", "NORMAL"),
                "alert_state": alert_state,
                "investigation_required": investigation_required,
                "failure_probability": result.get("failure_probability"),
                "threshold": result.get("threshold"),
                "risk_score": result.get("risk_score"),
                "trust_boundary_applied": trust_boundary_applied,
                "interpretation": result.get("interpretation"),
                "recommended_action": result.get("recommended_action"),
            },
        }

    def _fail(self, api: err.ApiError, started: float = 0.0) -> None:
        duration_ms = time.monotonic() - started if started > 0 else 0.0
        self._log_record(
            status=api.status,
            error_category=api.code,
            duration_ms=duration_ms,
        )
        envelope = err.error_envelope(self._correlation_id, api)
        self._write_json(api.status, envelope, headers=api.headers)


def create_server(host: str = "127.0.0.1", port: int = 0) -> GridSentinelServer:
    """Build a ready-to-serve inference server bound to (host, port).

    ``port=0`` selects an ephemeral port (useful for tests). Rate-limiting
    configuration is read from the environment at construction time.
    """
    configure_logging()
    limit = int(os.environ.get("GRIDSENTINEL_RATE_LIMIT_REQUESTS", "60"))
    window = int(os.environ.get("GRIDSENTINEL_RATE_LIMIT_WINDOW_SECONDS", "60"))
    httpd = GridSentinelServer((host, port), GridSentinelHandler)
    httpd.limiter = FixedWindowRateLimiter(limit=limit, window_seconds=window)
    return httpd


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument(
        "--host", default=os.environ.get("GRIDSENTINEL_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GRIDSENTINEL_PORT", "8000"))
    )
    args = parser.parse_args(argv)

    if not is_configured():
        logging.getLogger("gridsentinel.api").warning(
            "GRIDSENTINEL_API_KEY is not set: /v1/assess will fail closed "
            "with 503. The key is never read from client input."
        )

    httpd = create_server(args.host, args.port)
    bound_host, bound_port = httpd.server_address[:2]
    logging.getLogger("gridsentinel.api").info(
        f"{SERVICE_NAME} listening on {bound_host}:{bound_port}"
    )

    # Signal registration requires the main thread; keep the default SIGINT
    # (KeyboardInterrupt) behavior for interactive use.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _request_shutdown)

    # shutdown() must not run on the serve_forever() thread, so a dedicated
    # watcher performs it once a shutdown signal has been received.
    def _watch_shutdown() -> None:
        _SHUTDOWN_REQUESTED.wait()
        httpd.shutdown()

    threading.Thread(
        target=_watch_shutdown, daemon=True, name="gridsentinel-shutdown"
    ).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Harmless if already set; guarantees the accept loop has exited.
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()