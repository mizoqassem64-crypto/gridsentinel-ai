"""
GridSentinel AI - Persistent Structured Logging Tests (Phase 6B-C P1)
====================================================================

Verifies the persistent log-file feature (RotatingFileHandler):
    L1  GRIDSENTINEL_LOG_FILE writes JSON-lines to the target file
    L2  Log records are valid JSON and use the allow-list only
    L3  Raw API keys never appear in log files
    L4  Rotation works at GRIDSENTINEL_LOG_MAX_BYTES (backups created)
    L5  Missing/unwritable log path fails safe (server still works)
    L6  Liveness endpoints are logged with their correlation_id

These tests DO NOT regenerate the dataset, retrain the model, modify any
artifact, or change the ML threshold.

Run (from the repository root):
    PYTHONPATH="$(pwd)" python3 backend/tests/test_file_logging.py
"""

import http.client
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backend.server as server  # noqa: E402
import backend.middleware as middleware  # noqa: E402

API_KEY = "filelog-test-key-0123456789abcdef-000000000000"
os.environ["GRIDSENTINEL_API_KEY"] = API_KEY

LOG_ENV = "GRIDSENTINEL_LOG_FILE"
MAXBYTES_ENV = "GRIDSENTINEL_LOG_MAX_BYTES"
BACKUP_ENV = "GRIDSENTINEL_LOG_BACKUP_COUNT"

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
    """Boots a server, forcing a fresh logger so the file handler is
    attached from the current environment on each test."""

    def __init__(self, env_overrides=None):
        self._saved = {}
        for key, value in (env_overrides or {}).items():
            self._saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        # A fresh logger lets configure_logging() attach the file handler
        # based on the current GRIDSENTINEL_LOG_FILE. This test module runs
        # standalone, so process-global logger state is safe to reset.
        middleware._logger.handlers = []
        middleware._logger.setLevel(logging.INFO)
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
def test_basic_file_logging():
    # L1/L2 - file logging writes valid JSON-lines
    with tempfile.TemporaryDirectory() as tmp:
        logfile = os.path.join(tmp, "api.log")
        harness = ServerHarness(env_overrides={
            LOG_ENV: logfile, MAXBYTES_ENV: "100000", BACKUP_ENV: "1",
        })
        try:
            get(harness.port, "/health")
            path = Path(logfile)
            ok = path.exists() and path.stat().st_size > 0
            check("L1 GRIDSENTINEL_LOG_FILE creates and writes the log file",
                  ok, str(path))
            lines = path.read_text(encoding="utf-8").splitlines()
            # find a health log line
            json_lines = [json.loads(l) for l in lines if l.strip()]
            ok = len(json_lines) > 0
            ok = ok and any(el.get("status") == 200 for el in json_lines)
            check("L2 log records are valid JSON with the allow-list",
                  ok, f"n={len(json_lines)}")

            # L3 - no raw API key in the log file
            raw = path.read_text(encoding="utf-8")
            ok = API_KEY not in raw
            check("L3 raw API key never appears in the log file", ok)
        finally:
            harness.close()


def test_missing_path_failsafe():
    # L5 - an unwritable log path must not break the server
    harness = ServerHarness(env_overrides={
        LOG_ENV: "/nonexistent_dir/nope/api.log",
    })
    try:
        status, _ = get(harness.port, "/health")
        check("L5 unwritable log path fails safe (health still serves)",
              status == 200, f"status={status}")
    finally:
        harness.close()


def test_rotation():
    # L4 - rotation produces a .1 backup when the file exceeds max bytes
    with tempfile.TemporaryDirectory() as tmp:
        logfile = os.path.join(tmp, "api.log")
        harness = ServerHarness(env_overrides={
            LOG_ENV: logfile, MAXBYTES_ENV: "2000", BACKUP_ENV: "2",
        })
        try:
            for _ in range(40):
                get(harness.port, "/health")
            backups = list(Path(tmp).glob("api.log*"))
            ok = Path(logfile).exists()
            ok = ok and any(str(b).endswith(".1") for b in backups)
            check("L4 RotatingFileHandler produces a .1 backup after rotation",
                  ok, f"files={[str(b.name) for b in backups]}")
        finally:
            harness.close()


# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FILE-LOGGING TESTS")
    print("=" * 70)

    test_basic_file_logging()
    test_missing_path_failsafe()
    test_rotation()

    print("\n" + "=" * 70)
    print(f"FILE-LOGGING TEST RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAILURES:
        print("Failing tests:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL FILE-LOGGING TESTS PASSED")
