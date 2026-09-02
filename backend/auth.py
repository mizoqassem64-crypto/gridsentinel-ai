"""API-key authentication for the GridSentinel inference API.

The ``X-API-Key`` header is compared (constant-time) against the set of
valid keys configured via the ``GRIDSENTINEL_API_KEY`` environment variable
(comma-separated) and/or ``GRIDSENTINEL_API_KEY_FILE`` (one key per line).
Keys are never accepted in the JSON body or query string and are never
logged; only a short hash of the accepted key is used as a principal
identifier in logs.

Multiple active keys allow rotation without restart. Revoked/unknown keys
return the same safe 401 envelope and never reveal which key (if any) was
accepted. The empty set fails closed with 503.

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
_KEY_ENV = "GRIDSENTINEL_API_KEY"
_KEY_FILE_ENV = "GRIDSENTINEL_API_KEY_FILE"


def _get_keys():
    """Return a deduplicated ordered list of valid (non-empty) keys.

    Sources are merged in priority order:
      1. GRIDSENTINEL_API_KEY  (comma-separated)
      2. GRIDSENTINEL_API_KEY_FILE  (one key per line)
    The order of first occurrence is preserved for stable principal hashing.
    """
    keys = []
    seen = set()

    env_value = os.environ.get(_KEY_ENV, "")
    for k in env_value.split(","):
        k = k.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    file_path = os.environ.get(_KEY_FILE_ENV, "").strip()
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    k = line.strip()
                    if k and k not in seen:
                        keys.append(k)
                        seen.add(k)
        except OSError:
            pass

    return keys


def is_configured() -> bool:
    """Return True when at least one valid API key is available."""
    return len(_get_keys()) > 0


def verify(header_value: str) -> bool:
    """Constant-time comparison of a header value against all valid keys."""
    if not isinstance(header_value, str):
        return False
    candidate = header_value.strip()
    if not candidate:
        return False
    for key in _get_keys():
        if hmac.compare_digest(candidate.encode("utf-8"),
                               key.encode("utf-8")):
            return True
    return False


def first_key_principal() -> str:
    """Stable short identifier for the first configured key (logs only).

    Returns ``"unconfigured"`` when no key is set. The returned value
    never contains the raw key.
    """
    keys = _get_keys()
    if not keys:
        return "unconfigured"
    digest = hashlib.sha256(keys[0].encode("utf-8")).hexdigest()
    return f"key-{digest[:12]}"


def hashed_principal(header_value: str) -> str:
    """Return the short hash principal for the *accepted* key.

    Accepts the raw header value and returns ``key-<sha256[:12]>`` for
    the matching key, or ``"unknown"`` when no key matches.  The raw key
    is never included in the return value.
    """
    if not isinstance(header_value, str):
        return "unknown"
    candidate = header_value.strip()
    if not candidate:
        return "unknown"
    for key in _get_keys():
        if hmac.compare_digest(candidate.encode("utf-8"),
                               key.encode("utf-8")):
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            return f"key-{digest[:12]}"
    return "unknown"


def all_principals() -> str:
    """Sorted, comma-separated principals for all configured keys."""
    keys = _get_keys()
    if not keys:
        return ""
    principals = set()
    for key in keys:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        principals.add(f"key-{digest[:12]}")
    return ",".join(sorted(principals))
