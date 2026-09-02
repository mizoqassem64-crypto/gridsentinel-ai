"""
GridSentinel AI - Secure Inference API (Phase 6A, standard-library build)
=========================================================================

Zero-dependency HTTP/JSON inference boundary around the hardened V2 risk
engine (ai.ml.risk_engine_v2.assess_risk_v2). Implemented on the Python
standard library (http.server + json) because FastAPI/Pydantic cannot be
installed or run on this Termux/Python 3.14 environment (pydantic-core has
no Android wheel and requires a Rust toolchain; pydantic v1 is broken on
Python 3.14). All security guarantees of the Phase 6A plan are preserved.

Security model
--------------
* API-key authentication via the ``X-API-Key`` request header. A valid
  key proves that the caller is an authorized API client ONLY. It does
  NOT establish physical telemetry provenance. ``trusted_source`` is
  therefore always False at the server boundary (server-side policy) and
  can never be supplied by the client.
* Strict, deterministic schema validation runs before the ML engine is
  called: unknown fields, missing required features, wrong types,
  NaN/Infinity, and out-of-range values are rejected with stable 4xx
  errors and structured details (single source of truth reused from
  ai.ml.artifact_guard.PHYSICAL_RANGES / STRICT_ML_FEATURES).
* Malformed JSON, wrong content type, and oversized bodies are rejected
  before any inference work happens.
* All failures return a stable JSON error envelope. No tracebacks, file
  paths, model filenames, hashes, environment variables, or secrets are
  ever returned to the client.
* Structured logging records request metadata (correlation id, hashed
  principal, status, latency, error category) but never API keys,
  telemetry payloads, model internals, or secrets.
* A basic in-process fixed-window rate limiter provides per-client
  protection against accidental overload. It is deliberately single-worker
  and is NOT a substitute for a shared, distributed rate limiter / API
  gateway in multi-worker deployments.
* Per-connection socket timeouts and a bounded concurrent-request cap
  (excess connections receive a 503 ``server_overloaded`` response, no
  thread is spawned) prevent slow-client and thread-exhaustion denial of
  service. Request lines are capped to 4096 bytes.

Run (from the repository root):
    PYTHONPATH="$(pwd)" GRIDSENTINEL_API_KEY=... \
        python3 -m backend.server --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health      -> service liveness (no model/artifact disclosure)
    GET  /health/ready -> deployment readiness (engine state; 503 if failed)
    POST /v1/assess   -> authenticated, strictly validated risk assessment
"""

API_VERSION = "1.0"
SERVICE_NAME = "gridsentinel-inference-api"