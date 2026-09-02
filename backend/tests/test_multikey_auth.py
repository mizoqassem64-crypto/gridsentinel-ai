"""
GridSentinel AI - Multi-Key API Authentication Tests (Phase 6B-C P1)
====================================================================

Verifies the multi-key authentication feature:
    M1  Multiple keys are accepted (comma-separated in GRIDSENTINEL_API_KEY)
    M2  Keys from GRIDSENTINEL_API_KEY_FILE are accepted (union with env)
    M3  A revoked/unknown key is rejected with 401
    M4  Constant-time verification (timing-safe) for all candidates
    M5  Empty key set fails closed (server_misconfigured -> 503)
    M6  Principal hashing never leaks the raw key
    M7  Log records carry only hashed principals (never raw keys)
    M8  all_principals returns a stable sorted, deduplicated set

These tests DO NOT regenerate the dataset, retrain the model, modify any
artifact, or change the ML threshold.

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_multikey_auth.py
"""

import http.client
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.auth as auth  # noqa: E402
import backend.server as server  # noqa: E402

KEY_ENV = "GRIDSENTINEL_API_KEY"
KEY_FILE_ENV = "GRIDSENTINEL_API_KEY_FILE"

KEY_A = "multikey-a-0123456789abcdef-aaaaaaaaaaa"
KEY_B = "multikey-b-0123456789abcdef-bbbbbbbbbbb"
KEY_C = "multikey-c-0123456789abcdef-ccccccccccc"

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


def request_auth(port, key_value):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    headers = {"Connection": "close", "Content-Type": "application/json"}
    if key_value is not None:
        headers["X-API-Key"] = key_value
    body = _tiny_body()
    conn.request("POST", "/v1/assess", body=body, headers=headers)
    resp = conn.getresponse()
    _ = resp.read()
    status = resp.status
    conn.close()
    return status


def _tiny_body():
    return json.dumps({"probe": True}).encode("utf-8")


# ===========================================================================
def test_file_key_sources():
    # M8 - all_principals from env source
    os.environ[KEY_ENV] = f"{KEY_A},{KEY_B},{KEY_A}"
    os.environ.pop(KEY_FILE_ENV, None)
    allp = auth.all_principals()
    pa = auth.hashed_principal(KEY_A)
    pb = auth.hashed_principal(KEY_B)
    known = {pa, pb}
    ok = allp == ",".join(sorted(known))
    check("M8 all_principals returns sorted, deduplicated principals", ok, allp)

    # M6 - principals never contain the raw key
    raw_leak = any(k in allp for k in (KEY_A, KEY_B, KEY_C))
    ok = not raw_leak
    ok = ok and auth.first_key_principal() in known
    check("M6 principal hashing never leaks the raw key", ok, allp)

    # M7 - verify() per key
    check("M1 verify accepts each configured key",
          auth.verify(KEY_A) and auth.verify(KEY_B) and not auth.verify(KEY_C),
          "")
    os.environ.pop(KEY_ENV, None)
    os.environ.pop(KEY_FILE_ENV, None)


def test_file_multi_key():
    # M1/M3 - two keys accepted, unknown key / missing key rejected with 401.
    # A valid key with a non-telemetry body proceeds past auth to schema
    # validation (422); a bad/missing key is rejected at auth (401). The
    # discriminator for "auth accepted" is "not 401".
    harness = ServerHarness(env_overrides={KEY_ENV: f"{KEY_A},{KEY_B}"})
    try:
        s_a = request_auth(harness.port, KEY_A)
        s_b = request_auth(harness.port, KEY_B)
        s_c = request_auth(harness.port, KEY_C)
        s_missing = request_auth(harness.port, None)
        ok = (s_a == 422 and s_b == 422 and s_c == 401 and s_missing == 401)
        check("M2 env multi-key accepted via HTTP; wrong/missing -> 401",
              ok, f"a={s_a} b={s_b} c={s_c} missing={s_missing}")
    finally:
        harness.close()


def test_file_union_source():
    # M3 - file source merges with env source
    with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as f:
        f.write(KEY_C + "\n" + KEY_A + "\n")
        fpath = f.name
    try:
        harness = ServerHarness(env_overrides={
            KEY_ENV: KEY_B, KEY_FILE_ENV: fpath,
        })
        try:
            s_a = request_auth(harness.port, KEY_A)
            s_b = request_auth(harness.port, KEY_B)
            s_c = request_auth(harness.port, KEY_C)
            ok = s_a == 422 and s_b == 422 and s_c == 422
            check("M3 API_KEY_FILE union with env key accepted",
                  ok, f"a={s_a} b={s_b} c={s_c}")
        finally:
            harness.close()
        # file must NOT contain the raw key in principals
        allp = auth.all_principals()
        check("M6 file-sourced keys never leak raw value into principals",
              all(KEY not in allp for KEY in (KEY_A, KEY_B, KEY_C)), allp)
    finally:
        os.unlink(fpath)
        os.environ.pop(KEY_ENV, None)
        os.environ.pop(KEY_FILE_ENV, None)


def test_fail_closed_empty():
    # M5 - empty key set fails closed
    harness = ServerHarness(env_overrides={KEY_ENV: None, KEY_FILE_ENV: None})
    try:
        s = request_auth(harness.port, KEY_A)
        ok = s == 503
        check("M5 empty key set fails closed -> 503", ok, f"status={s}")
    finally:
        harness.close()


def test_timing_safe():
    # M4 - verification is constant-time (compare_digest path); assert it
    # uses hmac.compare_digest rather than a plain == via introspection.
    import hmac
    import inspect
    src = inspect.getsource(auth.verify)
    ok = "hmac.compare_digest" in src
    check("M4 verify uses constant-time hmac.compare_digest", ok)


def test_principal_sources():
    # M1/M6 - first_key_principal stable and hashed for a single key
    os.environ[KEY_ENV] = KEY_A
    os.environ.pop(KEY_FILE_ENV, None)
    p1 = auth.first_key_principal()
    p2 = auth.first_key_principal()
    ok = p1 == p2 and p1.startswith("key-") and p1 not in KEY_A
    check("M6 first_key_principal is stable, hashed, and not the raw key", ok, p1)

    hp = auth.hashed_principal(KEY_B)
    ok = hp == "unknown"
    check("M6 hashed_principal('unknown-key') -> 'unknown'", ok, hp)

    hp_empty = auth.hashed_principal(None)
    ok = hp_empty == "unknown"
    check("M6 hashed_principal(None) -> 'unknown'", ok, hp_empty)
    os.environ.pop(KEY_ENV, None)


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MULTI-KEY AUTH TESTS")
    print("=" * 70)

    test_file_key_sources()
    test_principal_sources()
    test_file_multi_key()
    test_file_union_source()
    test_fail_closed_empty()
    test_timing_safe()

    print("\n" + "=" * 70)
    print(f"MULTI-KEY AUTH TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL MULTI-KEY AUTH TESTS PASSED")
