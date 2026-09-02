"""
GridSentinel AI - Server Hardening Limit Tests (Phase 6B-B)
===========================================================

Read-only, runnable-from-repo-root checks for the P0/P1 server-hardening
behaviors added in Phase 6B-B (bucket 1):

    H1  Request line over the tightened cap (4096) is rejected with 414
    H2  Request line over the stdlib cap (65536) is rejected with 414
    H3  A stalled client (slowloris) is dropped after the socket timeout
        and the server remains healthy
    H4  A request below the concurrency cap is still served (200)
    H5  Requests beyond the concurrency cap get an immediate 503
        server_overloaded and the server recovers once slots free up
    H6  The rate limiter stays memory-bounded without wholesale-clearing
        live windows, and expired windows are swept

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_hardening_limits.py
"""

import http.client
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.middleware as mw  # noqa: E402
import backend.server as server  # noqa: E402
from backend.middleware import FixedWindowRateLimiter  # noqa: E402

API_KEY = "hardening-limit-test-key-00000000-00000000"
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


class ServerHarness:
    """Boots a real GridSentinelServer in a background thread."""

    def __init__(self, env_overrides=None):
        self._saved = {}
        for key, value in (env_overrides or {}).items():
            self._saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        self.httpd = server.create_server("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def http_request(port, method="GET", path="/health", body=None, headers=None):
    headers = dict(headers or {})
    headers.setdefault("Connection", "close")
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


def raw_request(port, data, read_timeout=6.0):
    out = b""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.settimeout(read_timeout)
            try:
                sock.sendall(data)
            except OSError:
                pass
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    out += chunk
            except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError):
                pass
    except OSError:
        pass
    return out


# ===========================================================================
def test_request_line_caps(harness):
    over = b"GET " + b"a" * 6000 + b" HTTP/1.0\r\nHost: t\r\n\r\n"
    out = raw_request(harness.port, over)
    head = out[:128].decode("utf-8", "replace")
    ok = b"414" in out[:128]
    check("H1 request line over tightened cap (4096) -> 414", ok, head)

    huge = b"GET " + b"b" * 70000 + b" HTTP/1.0\r\nHost: t\r\n\r\n"
    out = raw_request(harness.port, huge)
    head = out[:128].decode("utf-8", "replace")
    ok = b"414" in out[:128]
    check("H2 request line over stdlib cap (65536) -> 414", ok, head)


def test_socket_timeout(harness):
    t0 = time.monotonic()
    dropped = False
    try:
        with socket.create_connection(("127.0.0.1", harness.port), timeout=5) as sock:
            sock.sendall(b"GET /health HTTP/1.0\r\nHost: x\r\n")
            sock.settimeout(6)
            try:
                while sock.recv(4096):
                    pass
                dropped = True
            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                dropped = True
    except OSError:
        dropped = True
    dt = time.monotonic() - t0
    ok = dropped and dt < 5.0 and dt > 0.8
    check("H3 stalled client dropped after socket timeout", ok,
          f"dt={dt:.2f}s dropped={dropped}")

    status, _ = http_request(harness.port, path="/health")
    check("H3 server healthy after slow-client connection", status == 200,
          f"status={status}")


def test_concurrency_cap(harness):
    stalls = []
    sock_a = socket.create_connection(("127.0.0.1", harness.port), timeout=5)
    sock_a.sendall(b"GET /health HTTP/1.0\r\nHost: a\r\n")
    stalls.append(sock_a)
    time.sleep(0.4)

    status_b, _ = http_request(harness.port, path="/health")
    check("H4 request below concurrency cap served (200)", status_b == 200,
          f"status={status_b}")

    sock_c = socket.create_connection(("127.0.0.1", harness.port), timeout=5)
    sock_c.sendall(b"GET /health HTTP/1.0\r\nHost: c\r\n")
    stalls.append(sock_c)
    time.sleep(0.4)

    out_d = raw_request(
        harness.port, b"GET /health HTTP/1.0\r\nHost: d\r\n\r\n"
    )
    ok = b"503" in out_d[:64] and b"server_overloaded" in out_d
    check("H5 requests beyond concurrency cap rejected 503", ok,
          out_d[:128].decode("utf-8", "replace"))

    for sock in stalls:
        sock.close()
    time.sleep(0.4)
    status_e, _ = http_request(harness.port, path="/health")
    check("H5 capacity recovered after slots released", status_e == 200,
          f"status={status_e}")


def test_rate_limiter_memory_bound():
    lim = FixedWindowRateLimiter(limit=10 ** 9, window_seconds=300)
    for i in range(2048):
        lim.allow(f"old-{i}")
    for _ in range(5):
        lim.allow("golden")
    for i in range(2048):
        lim.allow(f"new-{i}")
    lim.allow("new-x")
    lim.allow("golden")
    count = lim._counts.get("golden", (0, 0))[1]
    ok = count == 6 and len(lim._counts) <= mw._RATE_MEMORY_BOUND
    check("H6 limiter memory-bounded, live windows never wholesale-cleared",
          ok, f"golden_count={count} len={len(lim._counts)}")


def test_rate_limiter_sweep():
    real = mw.time.monotonic
    clock = [1000000.0]
    mw.time.monotonic = lambda: clock[0]
    try:
        lim = FixedWindowRateLimiter(limit=2, window_seconds=30)
        lim.allow("a")
        lim.allow("b")
        clock[0] += 1.0
        lim.allow("a")
        clock[0] += 30.0 + 5.0
        lim.allow("a")
        ok = "b" not in lim._counts
        ok = ok and lim._counts.get("a", (0, 0))[1] == 1
        ok = ok and len(lim._counts) == 1
        check("H6 expired windows swept without clearing active state", ok,
              str(lim._counts))
    finally:
        mw.time.monotonic = real


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SERVER HARDENING LIMIT TESTS")
    print("=" * 70)

    harness = ServerHarness()
    try:
        test_request_line_caps(harness)
    finally:
        harness.close()

    harness = ServerHarness({"GRIDSENTINEL_SOCKET_TIMEOUT": 1.0})
    try:
        test_socket_timeout(harness)
    finally:
        harness.close()

    harness = ServerHarness({
        "GRIDSENTINEL_MAX_CONCURRENT": 2,
        "GRIDSENTINEL_SOCKET_TIMEOUT": 5.0,
    })
    try:
        test_concurrency_cap(harness)
    finally:
        harness.close()

    test_rate_limiter_memory_bound()
    test_rate_limiter_sweep()

    print("\n" + "=" * 70)
    print(f"HARDENING LIMIT TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL HARDENING LIMIT TESTS PASSED")