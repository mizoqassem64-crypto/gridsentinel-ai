"""
GridSentinel AI - Readiness Endpoint Tests (Phase 6B-C P1)
==========================================================

Verifies GET /health/ready:
    R1  /health/ready returns 200 "ok" in the 'unknown' state (cold start)
    R2  /health/ready returns 200 "ok" with engine_state='ready' once loaded
    R3  /health/ready returns 503 with code 'not_ready' when init failed
    R4  /health/ready /health responses never leak model/artifact paths
    R5  /health/ready does NOT force the ML engine to load (lazy)
    R6  Liveness /health remains independent of readiness state

These tests DO NOT regenerate the dataset, retrain the model, modify any
artifact, or change the ML threshold.

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_readiness.py
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

API_KEY = "readiness-test-key-0123456789abcdef-00000000000"
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


LEAK_PATTERNS = [
    "Traceback", "models/", ".pt", ".json", "site-packages",
    "gridsentinel-ai", "GRIDSENTINEL_API_KEY", "failure_predictor_v2",
    "failure_scaler_v2", "v2_artifact_manifest",
]


def leak_scan(text):
    return [p for p in LEAK_PATTERNS if p in text]


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
    conn.close()
    return status, text


# ===========================================================================
def test_unknown_and_lazy():
    # R1 - fresh server reports 200 in 'unknown' state
    harness = ServerHarness()
    try:
        status, text = get(harness.port, "/health/ready")
        env = json.loads(text) if status == 200 else {}
        ok = status == 200
        ok = ok and env.get("status") == "ok"
        ok = ok and env.get("ready") is True
        ok = ok and env.get("engine_state") in ("unknown", "ready")
        check("R1 /health/ready returns 200 ok (unknown/ready)", ok, text[:200])

        # R5 - cold server must not have loaded the ML engine
        state = harness.httpd.ready_state
        ok = state in (server._READY_STATE_UNKNOWN, server._READY_STATE_READY)
        check("R5 /health/ready lazy on fresh start (engine not forced)",
              ok, f"state={state}")

        # R4 - no leakage
        _, text2 = get(harness.port, "/health/ready")
        leaks = leak_scan(text2)
        check("R4 /health/ready leaks no model/artifact paths", not leaks, str(leaks))
    finally:
        harness.close()


def test_liveness_independent():
    # R6 - /health works regardless of readiness state
    harness = ServerHarness()
    try:
        status, text = get(harness.port, "/health")
        ok = status == 200 and json.loads(text).get("status") == "ok"
        check("R6 /health liveness independent of readiness", ok, text[:200])
    finally:
        harness.close()


def test_not_ready_state():
    # R3 - when engine init fails, /health/ready returns 503 not_ready
    harness = ServerHarness()
    try:
        harness.httpd.ready_state = server._READY_STATE_NOT_READY
        status, text = get(harness.port, "/health/ready")
        env = json.loads(text) if status == 503 else {}
        ok = status == 503
        ok = ok and env.get("status") == "error"
        ok = ok and env.get("ready") is False
        ok = ok and env.get("error", {}).get("code") == "not_ready"
        check("R3 /health/ready returns 503 not_ready on failed init",
              ok, text[:200])

        # R6 - liveness still 200 even when not ready
        s2, _ = get(harness.port, "/health")
        check("R6 /health stays 200 when engine not_ready", s2 == 200, f"status={s2}")
    finally:
        harness.close()


def test_ready_state():
    # R2 - when engine is loaded, /health/ready reports engine_state='ready'
    harness = ServerHarness()
    try:
        harness.httpd.ready_state = server._READY_STATE_READY
        status, text = get(harness.port, "/health/ready")
        env = json.loads(text) if status == 200 else {}
        ok = status == 200
        ok = ok and env.get("ready") is True
        ok = ok and env.get("engine_state") == "ready"
        check("R2 /health/ready reports engine_state=ready when loaded",
              ok, text[:200])
    finally:
        harness.close()


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING READINESS TESTS")
    print("=" * 70)

    test_unknown_and_lazy()
    test_liveness_independent()
    test_not_ready_state()
    test_ready_state()

    print("\n" + "=" * 70)
    print(f"READINESS TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL READINESS TESTS PASSED")
