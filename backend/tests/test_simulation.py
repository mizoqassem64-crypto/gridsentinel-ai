"""
GridSentinel AI - /v1/simulate Endpoint Regression Tests
========================================================

Verifies per-asset simulation baselines, fault-type passthrough,
envelope compatibility, and input validation.

    S1  Valid simulation returns 200 with simulation + result blocks
    S2  All four fault types succeed
    S3  Per-asset baselines differ (T01/T02/T03 produce different telemetry)
    S4  T03 overheating at high severity produces higher risk than T01
    S5  Invalid asset_id is rejected (400)
    S6  Invalid fault_type is rejected (400)
    S7  Unauthenticated request is rejected (401)
    S8  Simulation result fields are compatible with assess result fields
    S9  Injected fault_type is preserved in telemetry sent to engine

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_simulation.py
"""

import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.server as server  # noqa: E402

API_KEY = "sim-test-key-0123456789abcdef0000000000"
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


def simulate(port, asset_id, fault_type, severity, api_key=API_KEY):
    body = json.dumps({
        "asset_id": asset_id,
        "fault_type": fault_type,
        "severity": severity,
    }).encode("utf-8")
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Connection": "close",
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/v1/simulate", body=body, headers=headers)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    conn.close()
    return status, json.loads(text) if text else {}


def assess(port, telemetry, api_key=API_KEY):
    body = json.dumps(telemetry).encode("utf-8")
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Connection": "close",
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/v1/assess", body=body, headers=headers)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    conn.close()
    return status, json.loads(text) if text else {}


# ---------------------------------------------------------------------------
# S1 - Valid simulation envelope
# ---------------------------------------------------------------------------

def test_valid_simulation_envelope(harness):
    status, env = simulate(harness.port, "T01", "overload", 0.5)
    check("S1 HTTP 200 for valid simulation", status == 200, f"status={status}")
    check("S1 envelope has 'simulation' block",
          "simulation" in env and env["simulation"]["active"] is True)
    check("S1 envelope has 'result' block",
          "result" in env and isinstance(env["result"], dict))
    check("S1 simulation.asset_id matches request",
          env.get("simulation", {}).get("asset_id") == "T01")
    check("S1 simulation.fault_type matches request",
          env.get("simulation", {}).get("fault_type") == "overload")
    check("S1 simulation.severity matches request",
          env.get("simulation", {}).get("severity") == 0.5)
    result = env.get("result", {})
    check("S1 result has risk_classification",
          "risk_classification" in result)
    check("S1 result has failure_probability",
          "failure_probability" in result)
    check("S1 result has recommended_action",
          "recommended_action" in result)
    check("S1 result has ml_score", "ml_score" in result)
    check("S1 result has operational_score", "operational_score" in result)
    check("S1 result has guard_applied", "guard_applied" in result)
    check("S1 result has trust_boundary_applied",
          "trust_boundary_applied" in result)


# ---------------------------------------------------------------------------
# S2 - All four fault types succeed
# ---------------------------------------------------------------------------

def test_all_fault_types(harness):
    for ft in ("overload", "overheating", "voltage_instability",
               "harmonic_distortion"):
        status, env = simulate(harness.port, "T01", ft, 0.5)
        ok = status == 200 and "result" in env
        check(f"S2 fault_type={ft} returns 200 with result", ok,
              f"status={status}")
        if ok:
            check(f"S2 {ft} simulation block present",
                  env.get("simulation", {}).get("fault_type") == ft)


# ---------------------------------------------------------------------------
# S3 - Per-asset baseline differentiation
# ---------------------------------------------------------------------------

def test_per_asset_differentiation(harness):
    """Same fault/severity across T01/T02/T03 must produce different telemetry
    because each asset starts from a different baseline."""
    results = {}
    for asset_id in ("T01", "T02", "T03"):
        status, env = simulate(harness.port, asset_id, "overload", 0.3)
        check(f"S3 {asset_id} simulate returns 200", status == 200)
        if status == 200:
            sim_telem = env.get("simulation", {}).get("simulated_telemetry", {})
            results[asset_id] = sim_telem

    if len(results) == 3:
        temps = {k: v.get("temperature_c") for k, v in results.items()}
        loads = {k: v.get("load_percent") for k, v in results.items()}
        volts = {k: v.get("voltage_pu") for k, v in results.items()}
        thds = {k: v.get("thd_percent") for k, v in results.items()}

        check("S3 temperatures differ across assets",
              len(set(temps.values())) == 3,
              f"temps={temps}")
        check("S3 loads differ across assets",
              len(set(loads.values())) == 3,
              f"loads={loads}")
        check("S3 voltages differ across assets",
              len(set(volts.values())) == 3,
              f"volts={volts}")
        check("S3 THD values differ across assets",
              len(set(thds.values())) == 3,
              f"thds={thds}")


# ---------------------------------------------------------------------------
# S4 - T03 higher baseline stress vs T01
# ---------------------------------------------------------------------------

def test_t03_higher_baseline_stress(harness):
    """T03 (81.5C baseline, 82% load) should produce a higher risk score
    than T01 (62.6C, 66.9% load) for the same overheating fault."""
    _, env_t01 = simulate(harness.port, "T01", "overheating", 0.9)
    _, env_t03 = simulate(harness.port, "T03", "overheating", 0.9)

    score_t01 = env_t01.get("result", {}).get("risk_score", 0)
    score_t03 = env_t03.get("result", {}).get("risk_score", 0)
    temp_t01 = env_t01.get("simulation", {}).get(
        "simulated_telemetry", {}).get("temperature_c", 0)
    temp_t03 = env_t03.get("simulation", {}).get(
        "simulated_telemetry", {}).get("temperature_c", 0)

    check("S4 T03 post-simulation temperature > T01",
          temp_t03 > temp_t01,
          f"T03={temp_t03} T01={temp_t01}")
    check("S4 T03 risk score >= T01 risk score",
          score_t03 >= score_t01,
          f"T03={score_t03} T01={score_t01}")


# ---------------------------------------------------------------------------
# S5 - Invalid asset_id
# ---------------------------------------------------------------------------

def test_invalid_asset(harness):
    status, env = simulate(harness.port, "T99", "overload", 0.5)
    check("S5 invalid asset_id returns 4xx", 400 <= status < 500,
          f"status={status}")
    check("S5 error code is present", "error" in env)


# ---------------------------------------------------------------------------
# S6 - Invalid fault_type
# ---------------------------------------------------------------------------

def test_invalid_fault_type(harness):
    status, env = simulate(harness.port, "T01", "explosion", 0.5)
    check("S6 invalid fault_type returns 4xx", 400 <= status < 500,
          f"status={status}")
    check("S6 error code is present", "error" in env)


# ---------------------------------------------------------------------------
# S7 - Authentication required
# ---------------------------------------------------------------------------

def test_authentication_required(harness):
    status, _ = simulate(harness.port, "T01", "overload", 0.5, api_key="wrong")
    check("S7 wrong API key returns 401", status == 401, f"status={status}")
    status2, _ = simulate(harness.port, "T01", "overload", 0.5, api_key="")
    check("S7 missing API key returns 401", status2 == 401,
          f"status={status2}")


# ---------------------------------------------------------------------------
# S8 - Envelope compatibility with /v1/assess
# ---------------------------------------------------------------------------

def test_envelope_compatibility(harness):
    """The 'result' block in simulate must contain the same field set as
    the 'result' block in a normal /v1/assess response."""
    from ai.ml.artifact_guard import STRICT_ML_FEATURES
    import pandas as pd

    df = pd.read_csv(ROOT / "datasets" / "grid_features.csv")
    row = df[(df["failure"] == 0) & (df["fault_type"] == "normal")].iloc[0]
    telemetry = {f: float(row[f]) for f in STRICT_ML_FEATURES}
    telemetry["previous_faults"] = float(row["previous_faults"])
    telemetry["fault_type"] = str(row["fault_type"])

    _, assess_env = assess(harness.port, telemetry)
    _, sim_env = simulate(harness.port, "T01", "overload", 0.3)

    assess_keys = set(assess_env.get("result", {}).keys())
    sim_keys = set(sim_env.get("result", {}).keys())
    check("S8 simulate result keys == assess result keys",
          assess_keys == sim_keys,
          f"only_in_assess={assess_keys - sim_keys} "
          f"only_in_sim={sim_keys - assess_keys}")


# ---------------------------------------------------------------------------
# S9 - Injected fault_type reaches the engine
# ---------------------------------------------------------------------------

def test_fault_type_preserved(harness):
    """The fault_type sent to the engine must be the actual injected fault,
    not 'normal'."""
    for ft in ("overload", "overheating", "voltage_instability",
               "harmonic_distortion"):
        status, env = simulate(harness.port, "T01", ft, 0.5)
        check(f"S9 {ft} simulation block has correct fault_type",
              env.get("simulation", {}).get("fault_type") == ft)
        result = env.get("result", {})
        reasons = result.get("reasons", [])
        has_fault_reason = any(f"Active fault: {ft}" in r for r in reasons)
        check(f"S9 {ft} operational risk reasons include fault_type",
              has_fault_reason,
              f"reasons={reasons[:3]}")


# ===========================================================================
if __name__ == "__main__":
    harness = ServerHarness()
    try:
        print("=" * 70)
        print("RUNNING SIMULATION REGRESSION TESTS")
        print("=" * 70)

        test_valid_simulation_envelope(harness)
        test_all_fault_types(harness)
        test_per_asset_differentiation(harness)
        test_t03_higher_baseline_stress(harness)
        test_invalid_asset(harness)
        test_invalid_fault_type(harness)
        test_authentication_required(harness)
        test_envelope_compatibility(harness)
        test_fault_type_preserved(harness)

        print("\n" + "=" * 70)
        print(f"SIMULATION TEST RESULTS: {PASS} passed, {FAIL} failed")
        print("=" * 70)
        if FAILURES:
            print("Failing tests:")
            for f in FAILURES:
                print(f"  - {f}")
            raise SystemExit(1)
        print("ALL SIMULATION TESTS PASSED")
    finally:
        harness.close()
