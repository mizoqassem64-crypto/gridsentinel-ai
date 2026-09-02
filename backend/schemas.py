"""Strict, deterministic validation of client telemetry (external boundary).

The external request schema is intentionally narrower than the internal
engine contract: only the exact V2 feature set plus a small allow-list of
context fields may be submitted. Every numeric value is type-checked, must
be finite, and must fall inside the PHYSICAL_RANGES contract reused from
ai.ml.artifact_guard (single source of truth - this layer can never be
looser than the engine's own guard).

Fields that encode server-side policy (``trusted_source``) or credentials
(``api_key`` / ``x_api_key`` / ``authorization``) are rejected outright so
a client can never influence the trust boundary or transport a key.
"""

import math
import threading
from typing import Any, Dict

OPTIONAL_TEXT_FIELDS = ("fault_type", "asset_id", "asset_type", "timestamp")
OPTIONAL_NUMERIC_FIELDS = ("previous_faults",)

# Never accepted from a client, regardless of value.
FORBIDDEN_FIELDS = ("trusted_source", "api_key", "x_api_key", "authorization")

# The engine contract constants (STRICT_ML_FEATURES / PHYSICAL_RANGES) live
# in ai.ml.artifact_guard, which pulls in torch/numpy on import. They are
# resolved lazily on the first validation so importing this schema - and
# therefore backend.server - never forces the ML stack resident before the
# first request. Resolution is idempotent and import-lock-safe.
_GUARD_LOCK = threading.Lock()
_GUARD_CONSTANTS: Dict[str, Any] = {}


def _engine_contract() -> Dict[str, Any]:
    with _GUARD_LOCK:
        if "features" not in _GUARD_CONSTANTS:
            from ai.ml.artifact_guard import (
                PHYSICAL_RANGES,
                STRICT_ML_FEATURES,
            )

            _GUARD_CONSTANTS["features"] = STRICT_ML_FEATURES
            _GUARD_CONSTANTS["ranges"] = PHYSICAL_RANGES
            _GUARD_CONSTANTS["allowed"] = (
                set(STRICT_ML_FEATURES)
                | set(OPTIONAL_TEXT_FIELDS)
                | set(OPTIONAL_NUMERIC_FIELDS)
            )
    return _GUARD_CONSTANTS


class SchemaValidationError(ValueError):
    """Raised with a {field: reason} map when the payload is unaccepted."""

    def __init__(self, field_errors: Dict[str, str]) -> None:
        super().__init__("telemetry schema validation failed")
        self.field_errors = dict(field_errors)


def _reject_nonfinite_number(value: Any) -> float:
    """Return float(value) only for genuine finite JSON numbers."""
    # ``bool`` is a subclass of ``int`` in Python; JSON true/false must not
    # be accepted as a telemetry number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("not a number")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        # JSON alone cannot carry NaN/Infinity, but ``1e999`` decodes to
        # ``inf``. Reject it here deterministically.
        raise ValueError("not finite")
    return fvalue


def validate_request(payload: Any) -> Dict[str, Any]:
    """Validate and normalize a decoded JSON request body.

    Returns a telemetry dict safe to hand to the V2 engine, or raises
    SchemaValidationError with a {field: reason} map.
    """
    contract = _engine_contract()
    strict_features = contract["features"]
    allowed_fields = contract["allowed"]
    physical_ranges = contract["ranges"]
    field_errors: Dict[str, str] = {}

    if not isinstance(payload, dict):
        raise SchemaValidationError(
            {"body": "Request body must be a JSON object."}
        )

    for forbidden in FORBIDDEN_FIELDS:
        if forbidden in payload:
            field_errors[forbidden] = (
                f"Field '{forbidden}' is controlled by the server and "
                "cannot be supplied by a client."
            )

    unknown = set(payload) - allowed_fields
    for name in sorted(unknown):
        field_errors[name] = "Unexpected field."

    for name in sorted(set(strict_features) - set(payload)):
        field_errors[name] = "Missing required telemetry feature."

    for name in strict_features:
        if name in field_errors or name not in payload:
            continue
        try:
            fvalue = _reject_nonfinite_number(payload[name])
        except (TypeError, ValueError):
            field_errors[name] = (
                "Number required; NaN, Infinity and non-numeric values "
                "are rejected."
            )
            continue
        # Derived deviation features are finite-checked above but are not
        # in PHYSICAL_RANGES; range-check only the physical fields.
        if name not in physical_ranges:
            continue
        lo, hi = physical_ranges[name]
        if not (lo <= fvalue <= hi):
            field_errors[name] = (
                f"Out of physical range [{lo}, {hi}]."
            )

    for name in OPTIONAL_NUMERIC_FIELDS:
        if name not in payload or name in field_errors:
            continue
        try:
            _reject_nonfinite_number(payload[name])
        except (TypeError, ValueError):
            field_errors[name] = (
                "Number required; NaN, Infinity and non-numeric values "
                "are rejected."
            )

    for name in OPTIONAL_TEXT_FIELDS:
        if name not in payload or name in field_errors:
            continue
        if not isinstance(payload[name], str):
            field_errors[name] = "String required."

    if field_errors:
        raise SchemaValidationError(field_errors)

    normalized: Dict[str, Any] = {}
    for name in strict_features:
        normalized[name] = float(payload[name])
    for name in OPTIONAL_NUMERIC_FIELDS:
        if name in payload:
            normalized[name] = float(payload[name])
    for name in OPTIONAL_TEXT_FIELDS:
        if name in payload:
            normalized[name] = payload[name]
    return normalized