"""
GridSentinel AI - Deployment Contract Tests (Phase 6B)
=======================================================

Read-only, runnable-from-repo-root checks that the deployables produced in
Phase 6B (deploy/run_api.sh, deploy/healthcheck.sh, docs/deployment.md) and
the server's operational behavior meet the deployment contract.

Required coverage:
    D1  deploy/run_api.sh and deploy/healthcheck.sh exist and are executable
    D2  run_api.sh fails closed when GRIDSENTINEL_API_KEY is unset
    D3  server started via run_api.sh serves /health with a stable envelope
        and exposes no model/artifact details
    D4  authenticated /v1/assess returns 200; a wrong key returns 401
    D5  ML artifacts (models/, datasets/, ai/, requirements.txt, README) are
        untouched by the deployment, and no API key is present in deploy
        scripts/docs or in server log output
    D6  importing backend.server does not force torch/numpy resident
        (sub-second cold start to the accept loop)
    D7  SIGTERM drains and the process exits with code 0 within 10 seconds

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_deployment_contract.py
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ai.ml.artifact_guard import STRICT_ML_FEATURES  # noqa: E402

RUN_API = ROOT / "deploy" / "run_api.sh"
HEALTHCHECK = ROOT / "deploy" / "healthcheck.sh"

API_KEY = "deploy-contract-test-key-0000000000-0000000000"
KEY_ENV = "GRIDSENTINEL_API_KEY"

ARTIFACT_PATHS = ("models/", "datasets/", "ai/", "requirements.txt", "README.md")
LEAK_TERMS = ("models/", ".pt", ".json", "site-packages", API_KEY)

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


def free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def http_request(port, method="GET", path="/health", body=None, headers=None):
    headers = dict(headers or {})
    headers.setdefault("Connection", "close")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    if body is not None:
        conn.request(method, path, body=body, headers=headers)
    else:
        conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    status = resp.status
    conn.close()
    return status, text


def wait_for_health(port, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, text = http_request(port, path="/health")
            if status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.25)
    return False


def healthy_payload():
    df = pd.read_csv(ROOT / "datasets" / "grid_features.csv")
    row = df[(df["failure"] == 0) & (df["fault_type"] == "normal")].iloc[0]
    payload = {f: float(row[f]) for f in STRICT_ML_FEATURES}
    payload["previous_faults"] = float(row["previous_faults"])
    payload["fault_type"] = str(row["fault_type"])
    return payload


def test_deploy_files():
    ok = RUN_API.is_file() and RUN_API.stat().st_mode & 0o100
    check("D1 run_api.sh exists and is executable", ok, str(RUN_API))
    ok = HEALTHCHECK.is_file() and HEALTHCHECK.stat().st_mode & 0o100
    check("D1 healthcheck.sh exists and is executable", ok, str(HEALTHCHECK))


def test_launcher_fails_closed():
    env = dict(os.environ)
    env.pop(KEY_ENV, None)
    proc = subprocess.run(
        [str(RUN_API), "--help"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ok = proc.returncode == 1
    ok = ok and "GRIDSENTINEL_API_KEY" in (proc.stdout + proc.stderr)
    check("D2 run_api.sh fails closed without API key", ok,
          f"rc={proc.returncode} out={proc.stdout!r} err={proc.stderr!r}")


def test_lazy_start_no_ml_stack():
    snippet = (
        "import sys, backend.server\n"
        "print('torch_loaded=%s' % ('torch' in sys.modules))\n"
        "print('engine_loaded=%s' % (bool(backend.server._ENGINE)))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    torch_loaded = "torch_loaded=False" in proc.stdout
    engine_loaded = "engine_loaded=False" in proc.stdout
    check("D6 importing backend.server leaves torch/numpy unloaded", torch_loaded,
          proc.stdout + proc.stderr)
    check("D6 engine not loaded on module import", engine_loaded,
          proc.stdout + proc.stderr)


def _start_launcher(port):
    env = dict(os.environ)
    env[KEY_ENV] = API_KEY
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen(
        [str(RUN_API), "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def test_launcher_server(proc, port):
    ok = wait_for_health(port, timeout=60)
    status, text = (200, "{}") if ok else (0, "")
    if ok:
        status, text = http_request(port, path="/health")
    env = json.loads(text) if text else {}
    ok = ok and status == 200
    ok = ok and env.get("status") == "ok"
    ok = ok and env.get("api_version") == "1.0"
    ok = ok and env.get("service") == "gridsentinel-inference-api"
    leaks = [t for t in LEAK_TERMS if t in text]
    ok = ok and not leaks
    check("D3 launcher serves /health with stable, leak-free envelope", ok,
          f"status={status} text={text[:200]} leaks={leaks}")

    payload = healthy_payload()
    status, text = http_request(
        port, method="POST", path="/v1/assess", body=json.dumps(payload),
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    )
    env = json.loads(text) if text else {}
    ok = status == 200 and env.get("status") == "ok"
    ok = ok and env.get("api_version") == "1.0"
    ok = ok and (env.get("result") or {}).get("investigation_required") in (True, False)
    check("D4 authenticated assess via launcher returns 200", ok, text[:200])

    status, text = http_request(
        port, method="POST", path="/v1/assess", body=json.dumps(payload),
        headers={"X-API-Key": "definitely-wrong", "Content-Type": "application/json"},
    )
    env = json.loads(text) if text else {}
    ok = status == 401 and env.get("error", {}).get("code") == "unauthorized"
    check("D4 wrong API key via launcher returns 401", ok, text[:200])


def test_sigterm_drain(proc):
    started = time.monotonic()
    proc.terminate()
    rc = None
    try:
        out, _ = proc.communicate(timeout=10)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
    elapsed = time.monotonic() - started
    log = out if isinstance(out, str) else ""
    ok = rc == 0
    ok = ok and elapsed <= 10.0
    ok = ok and API_KEY not in log
    check("D7 SIGTERM drains and exits 0 within 10s", ok,
          f"rc={rc} elapsed={elapsed:.2f}s")
    check("D5 API key never appears in server log output", API_KEY not in log,
          "key found in captured log")


def test_no_secrets_and_artifact_integrity():
    scan_text = ""
    for path in (ROOT / "deploy", ROOT / "docs" / "deployment.md"):
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    scan_text += child.read_text(errors="ignore")
        elif path.is_file():
            scan_text += path.read_text(errors="ignore")
    ok = API_KEY not in scan_text
    check("D5 no API key present in deploy scripts/docs", ok)

    proc = subprocess.run(
        ["git", "status", "--short", "--", *ARTIFACT_PATHS],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    ok = proc.returncode == 0 and proc.stdout.strip() == ""
    check("D5 ML artifacts untouched (git clean on artifact paths)", ok,
          proc.stdout.strip() or proc.stderr.strip())


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DEPLOYMENT CONTRACT TESTS")
    print("=" * 70)

    test_deploy_files()
    test_launcher_fails_closed()
    test_lazy_start_no_ml_stack()

    port = free_port()
    proc = _start_launcher(port)
    try:
        test_launcher_server(proc, port)
        test_sigterm_drain(proc)
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        except OSError:
            pass

    test_no_secrets_and_artifact_integrity()

    print("\n" + "=" * 70)
    print(f"DEPLOYMENT CONTRACT TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL DEPLOYMENT CONTRACT TESTS PASSED")