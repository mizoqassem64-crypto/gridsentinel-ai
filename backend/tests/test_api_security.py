"""
GridSentinel AI - Secure Inference API Security Tests (Phase 6A)
================================================================

Runs against a real, in-process instance of the standard-library inference
API (backend.server) over HTTP on an ephemeral loopback port.

Required coverage (19 cases):
    T1  Valid authenticated assessment returns the stable "ok" envelope
    T2  Missing API key                            -> 401 unauthorized
    T3  Invalid API key                            -> 401 unauthorized
    T4  API key in the JSON body cannot bypass header authentication
    T5  trusted_source can never be client-controlled (-> 422)
    T6  Missing required telemetry feature         -> 422
    T7  Wrong value type                           -> 422
    T8  NaN is rejected (non-finite token)         -> 400 invalid_json
    T9  Infinity is rejected (1e999 -> 422; Infinity token -> 400)
    T10 Unknown / extra field                      -> 422
    T11 Out-of-physical-range value                -> 422
    T12 Wrong Content-Type                         -> 415
    T13 Oversized request body                     -> 413
    T14 Malformed JSON                             -> 400
    T15 Internal exceptions never leak tracebacks/paths/secrets (500)
    T16 Successful responses expose no filesystem/model/artifact paths
    T17 Known severe failure remains detectable via the API
    T18 Spoofed/untrusted telemetry cannot silently become LOW/NORMAL
    T19 The existing 46-test V2 hardening suite still passes

Extra coverage:
    T20 GET /health liveness without model disclosure
    T21 Wrong HTTP method                          -> 405
    T22 Unconfigured server API key fails closed   -> 503
    T23 In-process rate limiting returns 429 with Retry-After

These tests DO NOT regenerate the dataset, retrain the model, modify any
artifact, or change the ML threshold.

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_api_security.py
"""

import http.client
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import backend.server as server  # noqa: E402
from ai.ml.artifact_guard import STRICT_ML_FEATURES  # noqa: E402

API_KEY = "test-secret-key-0123456789abcdef-0000000000"
KEY_ENV = "GRIDSENTINEL_API_KEY"
os.environ[KEY_ENV] = API_KEY

DATASET = ROOT / "datasets" / "grid_features.csv"

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


# ---------------------------------------------------------------------------
# Request / payload helpers
# ---------------------------------------------------------------------------

def _features(row):
    return {f: float(row[f]) for f in STRICT_ML_FEATURES}


def load_healthy():
    df = pd.read_csv(DATASET)
    row = df[(df["failure"] == 0) & (df["fault_type"] == "normal")].iloc[0]
    return {
        **_features(row),
        "previous_faults": float(row["previous_faults"]),
        "fault_type": str(row["fault_type"]),
    }


def load_severe_failure():
    """A row with a severe physical condition (temp >= 100C) and failure=1."""
    df = pd.read_csv(DATASET)
    row = df[(df["failure"] == 1) & (df["temperature_c"] >= 100)].iloc[0]
    return {
        **_features(row),
        "previous_faults": float(row["previous_faults"]),
        "fault_type": str(row["fault_type"]),
    }


def spoofed_inconsistent(healthy):
    """Healthy telemetry with an internally contradictory power factor."""
    spoofed = dict(healthy)
    spoofed["power_factor"] = 1.0 - healthy["power_factor"] + 0.5
    return spoofed


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


def request(port, method="POST", path="/v1/assess", body=None, headers=None):
    headers = dict(headers or {})
    headers.setdefault("Connection", "close")
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    if body is not None:
        conn.request(method, path, body=body, headers=headers)
    else:
        conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    resp_headers = dict(resp.getheaders())
    conn.close()
    return status, resp_headers, text


def envelope_of(text):
    return json.loads(text)


HDR = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _replace_token(payload: dict, field: str, token: str) -> bytes:
    """Encode a payload substituting the token for the field's entire number."""
    raw = json.dumps(payload)
    pattern = re.compile(rf'"{re.escape(field)}":\s*[-+0-9.eE]+')
    new, count = pattern.subn(f'"{field}": {token}', raw, count=1)
    assert count == 1, f"field {field} not found in serialized payload"
    return new.encode("utf-8")


# Leakage patterns that must NEVER appear in any API response.
LEAK_PATTERNS = [
    "Traceback",
    "File \"",
    ", line ",
    "/data/",
    "/home/",
    "gridsentinel-ai",
    "site-packages",
    "failure_predictor_v2",
    "failure_scaler_v2",
    "failure_threshold_v2",
    "v2_artifact_manifest",
    "models/",
    ".pt",
    ".json",
    "GRIDSENTINEL_API_KEY",
    re.compile(r"[0-9a-f]{64}"),  # sha256-sized hex blob
]


def leak_scan(text):
    found = []
    for pattern in LEAK_PATTERNS:
        if isinstance(pattern, re.Pattern):
            if pattern.search(text):
                found.append(pattern.pattern)
        elif pattern in text:
            found.append(pattern)
    return found


# ===========================================================================
def test_basic_happypath(harness):
    healthy = load_healthy()

    # T1 - valid authenticated assessment
    status, rh, text = request(harness.port, body=healthy, headers=HDR)
    ok = status == 200
    env = envelope_of(text) if ok else {}
    ok = ok and env.get("status") == "ok"
    ok = ok and env.get("api_version") == "1.0"
    ok = ok and bool(env.get("request_id"))
    result = env.get("result") or {}
    ok = ok and result.get("threshold") == 0.70
    ok = ok and isinstance(result.get("failure_probability"), float)
    ok = ok and result.get("risk_classification") in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    ok = ok and isinstance(result.get("investigation_required"), bool)
    ok = ok and "X-Request-ID" in rh
    ok = ok and rh.get("Content-Type", "").startswith("application/json")
    ok = ok and rh.get("Server", "") == "GridSentinel"
    ok = ok and "Python" not in rh.get("Server", "")
    check("T1 valid authenticated assessment returns stable ok envelope", ok, text[:300])

    # T16 - no filesystem / model / artifact paths or hashes leaked
    leaks = leak_scan(text)
    check("T16 response exposes no filesystem/model paths or hashes", not leaks, str(leaks))

    # T2 - missing API key (valid JSON + correct content type, no auth header)
    status, _, text = request(harness.port, body=healthy,
                              headers={"Content-Type": "application/json"})
    ok = status == 401 and envelope_of(text).get("error", {}).get("code") == "unauthorized"
    check("T2 missing API key -> 401", ok, text[:200])

    # T3 - invalid API key
    status, _, text = request(harness.port, body=healthy,
                              headers={"X-API-Key": "wrong-key-abc", "Content-Type": "application/json"})
    ok = status == 401 and envelope_of(text).get("error", {}).get("code") == "unauthorized"
    check("T3 invalid API key -> 401", ok, text[:200])


def test_t4_t5(harness):
    healthy = load_healthy()

    # T4a - key in the JSON body alone must NOT authenticate (header missing)
    payload = dict(healthy)
    payload["api_key"] = API_KEY
    status, _, text = request(harness.port, body=payload,
                              headers={"Content-Type": "application/json"})
    ok = status == 401 and envelope_of(text).get("error", {}).get("code") == "unauthorized"
    check("T4 API key in JSON body cannot bypass header auth (no header -> 401)", ok, text[:200])

    # T4b - key in the body WITH a valid header must still be rejected (unknown field -> 422)
    status, _, text = request(harness.port, body=payload, headers=HDR)
    ok = status == 422
    detail = envelope_of(text).get("error", {}).get("detail") or {}
    ok = ok and "api_key" in detail
    check("T4 body key with valid header -> 422 unknown field", ok, text[:200])

    # T5 - trusted_source can never be client-controlled
    for value in (True, False):
        payload = dict(healthy)
        payload["trusted_source"] = value
        status, _, text = request(harness.port, body=payload, headers=HDR)
        ok = status == 422
        detail = envelope_of(text).get("error", {}).get("detail") or {}
        ok = ok and "trusted_source" in detail
        check(f"T5 trusted_source={value} rejected as server-controlled field", ok, text[:200])


def test_schema_rejections(harness):
    healthy = load_healthy()

    # T6 - missing required feature
    payload = dict(healthy)
    del payload["rated_mva"]
    status, _, text = request(harness.port, body=payload, headers=HDR)
    ok = status == 422
    detail = envelope_of(text).get("error", {}).get("detail") or {}
    ok = ok and "rated_mva" in detail
    check("T6 missing required feature -> 422", ok, text[:200])

    # T7 - wrong type (string) and wrong type (boolean)
    for field, value in (("current_a", "fast"), ("rated_mva", True)):
        payload = dict(healthy)
        payload[field] = value
        status, _, text = request(harness.port, body=payload, headers=HDR)
        ok = status == 422
        detail = envelope_of(text).get("error", {}).get("detail") or {}
        ok = ok and field in detail
        check(f"T7 wrong type for {field} -> 422", ok, text[:200])

    # T8 - NaN token is rejected before schema (invalid JSON constant -> 400)
    body = _replace_token(healthy, "current_a", "NaN")
    status, _, text = request(harness.port, body=body, headers=HDR)
    ok = status == 400 and envelope_of(text).get("error", {}).get("code") == "invalid_json"
    check("T8 NaN token rejected -> 400", ok, text[:200])

    # T9a - decimal overflow decodes to inf and is rejected as Not-a-Finite -> 422
    body = _replace_token(healthy, "current_a", "1e999")
    status, _, text = request(harness.port, body=body, headers=HDR)
    ok = status == 422
    detail = envelope_of(text).get("error", {}).get("detail") or {}
    ok = ok and ("current_a" in detail or any("finite" in str(v).lower() for v in detail.values()))
    check("T9 1e999 (Infinity) rejected -> 422", ok, text[:200])

    # T9b - Infinity token rejected -> 400
    body = _replace_token(healthy, "current_a", "Infinity")
    status, _, text = request(harness.port, body=body, headers=HDR)
    ok = status == 400 and envelope_of(text).get("error", {}).get("code") == "invalid_json"
    check("T9 Infinity token rejected -> 400", ok, text[:200])

    # T10 - unknown/extra field
    payload = dict(healthy)
    payload["apply_backdoor"] = 1
    payload["hacker_flag"] = 1
    status, _, text = request(harness.port, body=payload, headers=HDR)
    ok = status == 422
    detail = envelope_of(text).get("error", {}).get("detail") or {}
    ok = ok and "apply_backdoor" in detail and "hacker_flag" in detail
    check("T10 unknown/extra fields -> 422", ok, text[:200])

    # T11 - out-of-physical-range
    payload = dict(healthy)
    payload["voltage_pu"] = 2.0
    status, _, text = request(harness.port, body=payload, headers=HDR)
    ok = status == 422
    detail = envelope_of(text).get("error", {}).get("detail") or {}
    ok = ok and "voltage_pu" in detail
    check("T11 out-of-range value -> 422", ok, text[:200])


def test_transport_guards(harness):
    healthy = load_healthy()

    # T12 - wrong Content-Type
    body = json.dumps(healthy).encode("utf-8")
    status, _, text = request(harness.port, body=body, headers={"X-API-Key": API_KEY, "Content-Type": "text/plain"})
    ok = status == 415 and envelope_of(text).get("error", {}).get("code") == "unsupported_media_type"
    check("T12 wrong Content-Type -> 415", ok, text[:200])

    # T13 - oversized body (checked before parse)
    big = b"x" * (server.MAX_BODY_BYTES + 1)
    status, _, text = request(harness.port, body=big, headers=HDR)
    ok = status == 413 and envelope_of(text).get("error", {}).get("code") == "payload_too_large"
    check("T13 oversized body -> 413", ok, text[:200])

    # T14 - malformed JSON
    status, _, text = request(harness.port, body=b"{not-json", headers=HDR)
    ok = status == 400 and envelope_of(text).get("error", {}).get("code") == "invalid_json"
    check("T14 malformed JSON -> 400", ok, text[:200])


def test_t15_no_internal_leak(harness):
    healthy = load_healthy()

    def _boom(data, trusted_source=False):
        raise RuntimeError(
            "TOP-SECRET-BOOM /data/data/com.termux/files/home/"
            "gridsentinel-ai/models/failure_predictor_v2.pt "
            "v2_artifact_manifest.json 1234abc...cba"
        )

    server._ENGINE["fn"] = _boom
    try:
        status, _, text = request(harness.port, body=healthy, headers=HDR)
    finally:
        # Restore real lazy engine for any later calls.
        server._ENGINE.pop("fn", None)

    ok = status == 500
    ok = ok and envelope_of(text).get("error", {}).get("code") == "internal_error"
    leaks = leak_scan(text) + [
        p for p in ("TOP-SECRET-BOOM", "1234abc") if p in text
    ]
    ok = ok and not leaks
    check("T15 internal error leaks no traceback/paths/secrets", ok,
          f"status={status} leaks={leaks} body={text[:200]}")


def test_failure_detection(harness):
    # T17 - known severe failure remains detectable
    payload = load_severe_failure()
    status, _, text = request(harness.port, body=payload, headers=HDR)
    env = envelope_of(text)
    result = env.get("result") or {}
    detected = (
        result.get("prediction") == "FAILURE"
        or result.get("alert_state") in ("FAILURE_ALERT", "INVESTIGATION")
    )
    ok = status == 200
    ok = ok and bool(result.get("investigation_required")) is True
    ok = ok and detected
    ok = ok and not (result.get("risk_classification") in ("LOW", "NORMAL")
                     and not result.get("investigation_required"))
    check("T17 known severe failure remains detectable", ok, text[:400])

    # T18 - spoofed/untrusted inconsistent telemetry cannot silently be LOW/NORMAL
    payload = spoofed_inconsistent(load_healthy())
    status, _, text = request(harness.port, body=payload, headers=HDR)
    result = envelope_of(text).get("result") or {}
    ok = status == 200
    ok = ok and result.get("alert_state") == "INVESTIGATION"
    ok = ok and result.get("investigation_required") is True
    check("T18 spoofed inconsistent telemetry cannot become LOW/NORMAL", ok, text[:400])


def test_extra_security(harness):
    # T20 - /health works and discloses no model/artifact info
    status, rh, text = request(harness.port, method="GET", path="/health")
    env = envelope_of(text)
    ok = status == 200 and env.get("status") == "ok"
    ok = ok and env.get("api_version") == "1.0"
    ok = ok and "service" in env
    leaks = leak_scan(text)
    ok = ok and not leaks
    check("T20 /health liveness without disclosure", ok, str(leaks))

    # T21 - wrong method -> 405
    status, _, text = request(harness.port, method="PUT", path="/v1/assess")
    ok = status == 405 and envelope_of(text).get("error", {}).get("code") == "method_not_allowed"
    check("T21 wrong HTTP method -> 405", ok, text[:200])


def test_rate_limit_edge():
    # T23 - rate limiting with a tight limit returns 429 + Retry-After
    harness = ServerHarness(env_overrides={"GRIDSENTINEL_RATE_LIMIT_REQUESTS": 5})
    try:
        statuses = []
        healthy = load_healthy()
        for _ in range(6):
            status, rh, text = request(harness.port, body=healthy, headers=HDR)
            statuses.append(status)
            if status == 429:
                ok429 = (envelope_of(text).get("error", {}).get("code") == "rate_limited"
                         and "Retry-After" in rh)
        ok = statuses == [200, 200, 200, 200, 200, 429]
        ok = ok and ok429
        check("T23 in-process rate limiting returns 429 + Retry-After", ok, str(statuses))
    finally:
        harness.close()


def test_fail_closed_unconfigured():
    # T22 - without a server-side key, /v1/assess fails closed (503)
    harness = ServerHarness(env_overrides={KEY_ENV: None})
    try:
        healthy = load_healthy()
        status, _, text = request(harness.port, body=healthy, headers=HDR)
        ok = status == 503
        ok = ok and envelope_of(text).get("error", {}).get("code") == "server_misconfigured"
        # The client-supplied key must never be accepted when the server is
        # unconfigured.
        status2, _, text2 = request(harness.port, body=healthy,
                                    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"})
        ok = ok and status2 == 503
        check("T22 unconfigured server API key fails closed -> 503", ok, text[:200])
    finally:
        harness.close()


def test_hardening_suite_still_passes():
    # T19 - the existing V2 hardening suite (46 tests) must still pass
    cmd = [sys.executable, str(ROOT / "ai" / "ml" / "test_safety_hardening.py")]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900
    )
    ok = proc.returncode == 0 and "ALL SECURITY TESTS PASSED" in proc.stdout
    summary = ""
    match = re.search(r"SECURITY TEST RESULTS: (\d+) passed, (\d+) failed", proc.stdout)
    if match:
        summary = f"{match.group(1)} passed / {match.group(2)} failed"
    check("T19 existing hardening suite still passes (46)", ok,
          summary or proc.stdout[-500:])


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CORE API TESTS")
    print("=" * 70)

    harness = ServerHarness()

    try:
        test_basic_happypath(harness)
        test_t4_t5(harness)
        test_schema_rejections(harness)
        test_transport_guards(harness)
        test_t15_no_internal_leak(harness)
        test_failure_detection(harness)
        test_extra_security(harness)
    finally:
        harness.close()

    test_rate_limit_edge()
    test_fail_closed_unconfigured()
    test_hardening_suite_still_passes()

    print("\n" + "=" * 70)
    print(f"API SECURITY TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL API SECURITY TESTS PASSED")