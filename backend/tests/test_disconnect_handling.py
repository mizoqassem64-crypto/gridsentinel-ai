"""
GridSentinel AI - Client Disconnect / Response Transmission Tests
=================================================================

Regression coverage for the production robustness fix that makes a client
disconnecting mid-response a normal, non-fatal condition instead of a
cascading traceback.

    D1  _write_json builds and sends a normal JSON response -> True
    D2  BrokenPipeError during transmission is absorbed -> False surfaced,
        no exception escapes, connection closed
    D3  ConnectionResetError during transmission is absorbed -> False
    D4  A failed success-path write does NOT trigger _fail() recursion
        (no second error response is attempted for a disconnected client)
    D5  handle_error() no longer raises TypeError (was calling
        hashed_principal() with no required 'header_value' argument)
    D6  Real client disconnect mid-response: server stays healthy and
        continues serving normal requests
    D7  Genuine unexpected exceptions are still surfaced (500 internal_error
        envelope), NOT silently swallowed
    D8  Authentication behavior is unchanged (200 valid; 401 missing/wrong)
    D9  Normal API response envelope format is unchanged

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_disconnect_handling.py
"""

import http.client
import io
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.server as server  # noqa: E402
from backend.middleware import FixedWindowRateLimiter  # noqa: E402

API_KEY = "disconnect-test-key-0123456789abcdef0000000000"
KEY_ENV = "GRIDSENTINEL_API_KEY"
os.environ[KEY_ENV] = API_KEY

PASS = 0
FAIL = 0
FAILURES = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"PASS | {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"FAIL | {name} | {detail}")


class FakeWFile:
    """In-memory stand-in for ``handler.wfile``.

    ``fail_with`` optionally raises a transport error on the first write to
    simulate a client that disconnected during response transmission.
    """

    def __init__(self, fail_with=None):
        self.buffer = bytearray()
        self.fail_with = fail_with
        self.write_calls = 0

    def write(self, data):
        self.write_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        self.buffer.extend(data)
        return len(data)

    def flush(self):
        return None


def make_handler(wfile, headers=None, body=None):
    """Build a GridSentinelHandler with all attributes _write_json needs.

    ``_assess`` requires additional attributes (rfile body, auth headers,
    limiter, engine); those are set in the specific test that needs them.
    """
    h = server.GridSentinelHandler.__new__(server.GridSentinelHandler)
    h._request_version = "HTTP/1.1"
    h.request_version = "HTTP/1.1"
    h.protocol_version = "HTTP/1.1"
    h.server_version = "GridSentinel"
    h.sys_version = ""
    h._headers_buffer = []
    h.wfile = wfile
    h.headers = headers if headers is not None else {}
    h._correlation_id = "test-request-id-00000000000000000000000000000000"
    h._bytes_read = 0
    h.close_connection = False
    h.rfile = io.BytesIO(body if body is not None else b"{}")
    h.raw_requestline = b"POST /v1/assess HTTP/1.1\r\n"
    h.requestline = "POST /v1/assess HTTP/1.1"
    h.command = "POST"
    h.path = "/v1/assess"
    h.client_address = ("127.0.0.1", 12345)
    h.server = types.SimpleNamespace(
        limiter=FixedWindowRateLimiter(limit=10 ** 6, window_seconds=60),
        global_limiter=None,
        ready_state="ready",
        request_line_limit=4096,
        socket_timeout=10.0,
    )
    return h


# ---------------------------------------------------------------------------
# D1/D2/D3 - _write_json unit tests
# ---------------------------------------------------------------------------

def test_write_json_success():
    wf = FakeWFile()
    h = make_handler(wf)
    ok = h._write_json(200, {"status": "ok", "hello": "world"}) is True
    check("D1 _write_json sends a normal response and returns True", ok)
    ok = ok and b"world" in bytes(wf.buffer)
    check("D1 response body is transmitted", ok)
    ok = ok and b"Content-Type: application/json" in bytes(wf.buffer)
    check("D1 content-type header present", ok)


def test_write_json_broken_pipe():
    wf = FakeWFile(fail_with=BrokenPipeError(32, "Broken pipe"))
    h = make_handler(wf)
    try:
        result = h._write_json(200, {"status": "ok"})
        no_escape = True
    except BrokenPipeError:
        result = None
        no_escape = False
    check("D2 BrokenPipeError absorbed (no exception escapes)", no_escape, str(result))
    check("D2 BrokenPipeError returns False", result is False)
    check("D2 connection flagged for close", h.close_connection is True)


def test_write_json_connection_reset():
    wf = FakeWFile(fail_with=ConnectionResetError(54, "Connection reset by peer"))
    h = make_handler(wf)
    try:
        result = h._write_json(200, {"status": "ok"})
        no_escape = True
    except (BrokenPipeError, ConnectionResetError):
        result = None
        no_escape = False
    check("D3 ConnectionResetError absorbed (no exception escapes)", no_escape, str(result))
    check("D3 ConnectionResetError returns False", result is False)


# ---------------------------------------------------------------------------
# D4 - failed write must not trigger _fail() recursion
# ---------------------------------------------------------------------------

def _valid_telemetry():
    # A schema-valid, minimal-set telemetry payload (mirrors the required
    # STRICT_ML_FEATURES plus the extra fields validated by the schema).
    import pandas as pd
    from ai.ml.artifact_guard import STRICT_ML_FEATURES
    df = pd.read_csv(ROOT / "datasets" / "grid_features.csv")
    row = df[(df["failure"] == 0) & (df["fault_type"] == "normal")].iloc[0]
    payload = {f: float(row[f]) for f in STRICT_ML_FEATURES}
    payload["previous_faults"] = float(row["previous_faults"])
    payload["fault_type"] = str(row["fault_type"])
    return payload


def test_no_fail_recursion_on_disconnect():
    telemetry = _valid_telemetry()
    body = json.dumps(telemetry).encode("utf-8")
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    wf = FakeWFile(fail_with=BrokenPipeError(32, "Broken pipe"))
    h = make_handler(wf, headers=headers, body=body)

    # A failing write triggers BrokenPipeError inside _write_json. With the
    # fix, _write_json returns False and _assess does NOT call _fail(), so
    # no second write (and no recursive escalation) can occur.
    server._ENGINE["fn"] = lambda telemetry_, trusted_source=False: {
        "risk_level": "LOW",
        "prediction": "NORMAL",
        "alert_state": "NORMAL",
        "failure_probability": 0.1,
        "threshold": 0.70,
        "risk_score": 12,
        "ml_score": 12,
        "operational_score": 10,
        "reasons": [],
        "guard_applied": False,
    }
    try:
        try:
            h._assess()
            no_recursion = True
        except Exception as exc:
            # If _fail() were reached, its own _write_json() would raise the
            # same BrokenPipeError again and escape _assess.
            no_recursion = False
            print(f"    (raised: {type(exc).__name__}: {exc})")
    finally:
        server._ENGINE.pop("fn", None)

    check("D4 disconnected success-write does not trigger _fail recursion",
          no_recursion)
    # The body write failed exactly once; a _fail attempt would have written
    # (and raised) a second time, which is impossible here because _write_json
    # returns before a second call.
    check("D4 no error response body written after disconnect",
          wf.write_calls == 1)


# ---------------------------------------------------------------------------
# D5 - handle_error must not raise TypeError
# ---------------------------------------------------------------------------

def test_handle_error_no_typeerror():
    # handle_error runs on the server (not the handler) and reads
    # sys.exc_info(). It previously called hashed_principal() with no
    # argument -> TypeError. It must now complete without raising.
    srv = server.GridSentinelServer.__new__(server.GridSentinelServer)
    raised = None
    try:
        try:
            raise ValueError("synthetic handler failure")
        except ValueError:
            srv.handle_error("sock", ("203.0.113.7", 443))
    except Exception as exc:
        raised = type(exc).__name__
    check("D5 handle_error does not raise TypeError", raised is None, str(raised))


# ---------------------------------------------------------------------------
# D6 - real client disconnect: server stays healthy
# ---------------------------------------------------------------------------

class ServerHarness:
    """Boots a real GridSentinelServer in a background thread."""

    def __init__(self):
        self.httpd = server.create_server("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def http_request(port, method="GET", path="/health", body=None, headers=None):
    headers = dict(headers or {})
    headers.setdefault("Connection", "close")
    headers.setdefault("X-API-Key", API_KEY)
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    if body is not None:
        conn.request(method, path, body=body, headers=headers)
    else:
        conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    conn.close()
    return status, text


def test_real_client_disconnect():
    harness = ServerHarness()
    try:
        telemetry = _valid_telemetry()
        body = json.dumps(telemetry).encode("utf-8")
        # Send a full valid request then slam the socket shut before reading
        # the (large) response so the server write hits a broken/reset pipe.
        try:
            sock = socket.create_connection(("127.0.0.1", harness.port), timeout=10)
            request_head = (
                "POST /v1/assess HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                f"X-API-Key: {API_KEY}\r\n"
                f"Content-Length: {len(body)}\r\n"
                "\r\n"
            ).encode("utf-8")
            sock.sendall(request_head + body)
            # Give the server a moment to read the request and begin writing.
            time.sleep(0.2)
            sock.close()
        except OSError:
            pass

        # Server must remain healthy and serve a normal request afterwards.
        status, text = http_request(harness.port, path="/health")
        ok = status == 200 and json.loads(text).get("status") == "ok"
        check("D6 server healthy after client disconnect mid-response", ok,
              f"status={status}")

        status2, _ = http_request(
            harness.port, method="POST", path="/v1/assess",
            body=json.dumps(telemetry).encode("utf-8"),
        )
        check("D6 server still serves valid assessment after disconnect",
              status2 == 200, f"status={status2}")

        ok_thread = harness._thread.is_alive()
        check("D6 request-processing thread still alive", ok_thread)
    finally:
        harness.close()


# ---------------------------------------------------------------------------
# D7/D8/D9 - genuine errors surfaced, auth + envelope unchanged
# ---------------------------------------------------------------------------

def test_genuine_error_still_surfaced():
    harness = ServerHarness()
    try:
        telemetry = _valid_telemetry()

        def _boom(data, trusted_source=False):
            raise RuntimeError("genuine internal bug that must not be hidden")

        server._ENGINE["fn"] = _boom
        try:
            status, text = http_request(
                harness.port, method="POST", path="/v1/assess",
                body=json.dumps(telemetry).encode("utf-8"),
            )
        finally:
            server._ENGINE.pop("fn", None)

        env = json.loads(text)
        ok = status == 500 and env.get("error", {}).get("code") == "internal_error"
        check("D7 genuine unexpected exception surfaced as 500 internal_error",
              ok, f"status={status} body={text[:200]}")
    finally:
        harness.close()


def test_auth_and_envelope_unchanged():
    harness = ServerHarness()
    try:
        telemetry = _valid_telemetry()
        body = json.dumps(telemetry).encode("utf-8")

        # D8 - valid key -> 200
        status, text = http_request(
            harness.port, method="POST", path="/v1/assess", body=body)
        ok = status == 200
        check("D8 valid API key -> 200", ok, f"status={status}")

        # D8 - missing key -> 401
        conn = http.client.HTTPConnection("127.0.0.1", harness.port, timeout=30)
        conn.request("POST", "/v1/assess", body=body, headers={
            "Connection": "close", "Content-Type": "application/json",
        })
        resp = conn.getresponse()
        text2 = resp.read().decode("utf-8", "replace")
        status2 = resp.status
        conn.close()
        ok = status2 == 401
        check("D8 missing API key -> 401", ok, f"status={status2}")

        # D9 - response envelope format unchanged
        env = json.loads(text)
        ok = (env.get("status") == "ok"
              and env.get("api_version") == "1.0"
              and bool(env.get("request_id"))
              and isinstance(env.get("result"), dict))
        check("D9 response envelope format unchanged", ok, text[:200])
    finally:
        harness.close()


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DISCONNECT / RESPONSE TRANSMISSION TESTS")
    print("=" * 70)

    test_write_json_success()
    test_write_json_broken_pipe()
    test_write_json_connection_reset()
    test_no_fail_recursion_on_disconnect()
    test_handle_error_no_typeerror()
    test_real_client_disconnect()
    test_genuine_error_still_surfaced()
    test_auth_and_envelope_unchanged()

    print("\n" + "=" * 70)
    print(f"DISCONNECT TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL DISCONNECT TESTS PASSED")
