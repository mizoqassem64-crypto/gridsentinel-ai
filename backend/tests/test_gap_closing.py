"""
GridSentinel AI - Gap-Closing Tests (Phase 6B-C P1)
===================================================

Closes coverage gaps identified during the Phase 6B-C recon, and covers
the new global rate limiter:

    G1  Global rate limit (GRIDSENTINEL_GLOBAL_RATE_LIMIT_REQUESTS) caps
        aggregate request volume across all client IPs with 429
    G2  Global limiter is disabled by default (0) and does not interfere
        with per-IP limiting
    G3  Rate-limit responses always carry a Retry-After header
    G4  Env override parsing is robust to empty/invalid values (fallbacks)
    G5  /health and /health/ready are served with the Server header set
        and no Python version disclosure

These tests DO NOT regenerate the dataset, retrain the model, modify any
artifact, or change the ML threshold.

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_gap_closing.py
"""

import http.client
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.server as server  # noqa: E402
from backend.middleware import FixedWindowRateLimiter  # noqa: E402

API_KEY = "gapclosing-test-key-0123456789abcdef-00000000000"
os.environ["GRIDSENTINEL_API_KEY"] = API_KEY

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


def get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    conn.request("GET", path, headers={"Connection": "close"})
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    resp_headers = dict(resp.getheaders())
    conn.close()
    return status, resp_headers, text


# ===========================================================================
def test_global_rate_limit():
    # G1 - the global limiter (the instance _assess calls via .allow("global"))
    # enforces a process-wide cap across all callers. Health GETs are exempt
    # from rate limiting by design, so this exercises the limiter directly
    # with the same "global" key the assess path uses.
    harness = ServerHarness(env_overrides={
        "GRIDSENTINEL_GLOBAL_RATE_LIMIT_REQUESTS": 2,
        "GRIDSENTINEL_GLOBAL_RATE_WINDOW_SECONDS": 60,
    })
    try:
        g = getattr(harness.httpd, "global_limiter", None)
        ok = g is not None and g.limit == 2 and g.window_seconds == 60
        check("G1 server builds a global limiter from env (limit=2)", ok,
              f"g={g}")
        if g is not None:
            a = [g.allow("global") for _ in range(4)]
            ok = a == [(True, 0), (True, 0), (False, 60), (False, 60)]
            check("G1 global limiter caps aggregate volume -> False/429",
                  ok, str(a))
    finally:
        harness.close()


def test_global_disabled_by_default():
    # G2 - when GRIDSENTINEL_GLOBAL_RATE_LIMIT_REQUESTS is 0 (or absent),
    # httpd.global_limiter is None; the per-IP limiter is still active.
    harness = ServerHarness(env_overrides={
        "GRIDSENTINEL_RATE_LIMIT_REQUESTS": 3,
    })
    try:
        g = getattr(harness.httpd, "global_limiter", "MISSING")
        ok = g is None
        check("G2 global_limiter is None by default (disabled)", ok, f"g={g}")
        # Verify per-IP limiter is still present
        per_ip = getattr(harness.httpd, "limiter", None)
        ok2 = per_ip is not None and per_ip.limit == 3
        check("G2 per-IP limiter still active (limit=3)", ok2,
              f"limit={per_ip.limit if per_ip else None}")
    finally:
        harness.close()


def test_env_parse_fallbacks():
    # G4 - _require_int/_require_float fall back for invalid/empty values
    saved = os.environ.get("GRIDSENTINEL_SOCKET_TIMEOUT")
    try:
        os.environ["GRIDSENTINEL_SOCKET_TIMEOUT"] = "not-a-number"
        v = server._require_float("GRIDSENTINEL_SOCKET_TIMEOUT",
                                  server._DEFAULT_SOCKET_TIMEOUT, 0.5)
        ok = v == server._DEFAULT_SOCKET_TIMEOUT
        check("G4 invalid float env falls back to default", ok, f"v={v}")

        os.environ["GRIDSENTINEL_MAX_CONCURRENT"] = "0"
        v = server._require_int("GRIDSENTINEL_MAX_CONCURRENT",
                                server._DEFAULT_MAX_CONCURRENT, 1)
        ok = v == server._DEFAULT_MAX_CONCURRENT
        check("G4 below-minimum int env falls back to default", ok, f"v={v}")

        os.environ["GRIDSENTINEL_MAX_CONCURRENT"] = "5"
        v = server._require_int("GRIDSENTINEL_MAX_CONCURRENT",
                                server._DEFAULT_MAX_CONCURRENT, 1)
        check("G4 valid int env is honored", v == 5, f"v={v}")
    finally:
        if saved is None:
            os.environ.pop("GRIDSENTINEL_SOCKET_TIMEOUT", None)
        else:
            os.environ["GRIDSENTINEL_SOCKET_TIMEOUT"] = saved
        os.environ.pop("GRIDSENTINEL_MAX_CONCURRENT", None)


def test_server_header_no_version():
    # G5 - Server header set, no Python version disclosure
    harness = ServerHarness()
    try:
        _, rh, _ = get(harness.port, "/health")
        ok = rh.get("Server", "") == "GridSentinel"
        ok = ok and "Python" not in rh.get("Server", "")
        check("G5 Server header set, no Python version leak", ok, str(rh))

        _, rh2, _ = get(harness.port, "/health/ready")
        ok2 = rh2.get("Server", "") == "GridSentinel"
        ok2 = ok2 and "Python" not in rh2.get("Server", "")
        check("G5 /health/ready Server header set, no version leak", ok2, str(rh2))
    finally:
        harness.close()


def test_rate_limiter_contract():
    # G3/G1 - unit-level: FixedWindowRateLimiter returns retry_after on cap
    lim = FixedWindowRateLimiter(limit=1, window_seconds=60)
    a1 = lim.allow("1.2.3.4")
    a2 = lim.allow("1.2.3.4")
    ok = a1 == (True, 0) and a2[0] is False and a2[1] >= 1
    check("G3 FixedWindowRateLimiter returns (False, retry_after) at cap",
          ok, str((a1, a2)))


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING GAP-CLOSING TESTS")
    print("=" * 70)

    test_global_rate_limit()
    test_global_disabled_by_default()
    test_env_parse_fallbacks()
    test_server_header_no_version()
    test_rate_limiter_contract()

    print("\n" + "=" * 70)
    print(f"GAP-CLOSING TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL GAP-CLOSING TESTS PASSED")
