"""API-key authentication for the GridSentinel inference API.

The ``X-API-Key`` header is compared (constant-time) against the expected
key configured via the ``GRIDSENTINEL_API_KEY`` environment variable. The
key is never accepted in the JSON body and is never logged; only a short
hash of it is used as a hashed principal identifier in logs.

Trust boundary mapping (spec item I)
-----------------------------------
An API key authenticates the CALLER to the API only. It does NOT prove
telemetry provenance: a legitimate client can forward spoofed readings.
Therefore the inference boundary always treats submitted telemetry as
untrusted (``trusted_source=False``) and relies on the V2 engine's
fail-safe investigation escalation for any abnormal signal. No form of
API credential ever upgrades telemetry provenance.
"""

import hashlib
import hmac
import os

API_KEY_HEADER = "X-API-Key"
KEY_ENV = "GRIDSENTINEL_API_KEY"


def expected_key() -> str:
    return os.environ.get(KEY_ENV, "").strip()


def is_configured() -> bool:
    return bool(expected_key())


def verify(header_value: str) -> bool:
    """Constant-time comparison of a header value against the env key."""
    if not isinstance(header_value, str):
        return False
    expected = expected_key()
    candidate = header_value.strip()
    if not expected or not candidate:
        return False
    return hmac.compare_digest(
        candidate.encode("utf-8"),
        expected.encode("utf-8"),
    )


def hashed_principal() -> str:
    """Stable short identifier for logs; never contains the raw key."""
    expected = expected_key()
    if not expected:
        return "unconfigured"
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    return f"key-{digest[:12]}"