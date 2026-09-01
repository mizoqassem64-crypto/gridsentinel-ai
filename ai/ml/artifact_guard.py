"""
GridSentinel AI - Artifact & Telemetry Guard
=============================================

Production hardening for the V2 failure prediction bundle.

Responsibilities:
    A. SAFE MODEL ARTIFACT LOADING
       - Load torch checkpoints with ``weights_only=True``.
       - Enforce a strict state_dict key/schema contract.
       - Reject malformed artifacts without pickle execution.

    B. ARTIFACT INTEGRITY
       - Verify a versioned manifest of SHA-256 hashes for the model
         bundle (model, scaler, metadata, threshold).
       - Verify schema/version compatibility and exact feature ordering.
       - Validate the threshold is finite and within a safe range.
       - Fail closed on any mismatch.

    C. STRICT INPUT VALIDATION
       - Reject NaN / Infinity / wrong types / missing / unexpected fields.
       - Validate physical ranges derived from project generation rules.

    D. TELEMETRY TRUST / SAFETY BOUNDARY
       - Explicit trusted/untrusted boundary.
       - Cross-measurement consistency checks based on the generator's
         own physical relationships (not invented standards).
       - Fail safely to INVESTIGATION instead of LOW/NORMAL when
         provenance is unverified and a signal is present.

This module never claims local validation provides cryptographic trust.
Its purpose is to fail closed and to surface uncertainty deterministically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ============================================================
# Manifest layout
# ============================================================

MANIFEST_VERSION = "1"


# ============================================================
# Expected V2 architecture contract
# ============================================================
# The V2 state dict is saved from an nn.Sequential nested under a
# ``.network`` attribute, so keys are ``network.<idx>.<param>``.
# The bare ``<idx>.<param>`` form is the raw Sequential equivalent.
# Layer indices refer to: Linear(0) ReLU(1) Linear(2) ReLU(3)
# Dropout(4) Linear(5) ReLU(6) Linear(7). Dropout/ReLU have no params.

EXPECTED_INPUT_SIZE = 16

# Normalized key -> expected tensor shape.
EXPECTED_STATE_DICT_SHAPES: Dict[str, tuple] = {
    "0.weight": (64, 16),
    "0.bias": (64,),
    "2.weight": (32, 64),
    "2.bias": (32,),
    "5.weight": (16, 32),
    "5.bias": (16,),
    "7.weight": (1, 16),
    "7.bias": (1,),
}

MANIFEST_KEYS = (
    "model",
    "scaler",
    "metadata",
    "threshold",
)


# ============================================================
# Physical range contract
# ============================================================
# Ranges follow the project generator's own clamping and failure
# thresholds (see ai/data/generate_dataset.py). Values outside these
# bounds are rejected as implausible telemetry.

PHYSICAL_RANGES: Dict[str, tuple] = {
    "rated_mva": (30.0, 80.0),
    "asset_age_years": (0.0, 40.0),
    "criticality": (0.0, 1.0),
    "voltage_pu": (0.85, 1.15),
    "current_a": (0.0, 1600.0),
    "frequency_hz": (45.0, 55.0),
    "active_power_mw": (0.0, 120.0),
    "reactive_power_mvar": (-20.0, 60.0),
    "power_factor": (0.50, 1.00),
    "temperature_c": (-20.0, 200.0),
    "load_percent": (0.0, 130.0),
    "thd_percent": (0.0, 30.0),
    "previous_faults": (0.0, 20.0),
}

# Features that must be strictly finite and within PHYSICAL_RANGES.
STRICT_ML_FEATURES = [
    "rated_mva",
    "asset_age_years",
    "criticality",
    "voltage_pu",
    "current_a",
    "frequency_hz",
    "active_power_mw",
    "reactive_power_mvar",
    "power_factor",
    "temperature_c",
    "load_percent",
    "thd_percent",
    "voltage_deviation",
    "frequency_deviation",
    "temperature_excess",
    "electrical_stress",
]

# Cross-measurement consistency tolerances (project relationships).
LOAD_FROM_POWER_TOL = 15.0        # load_percent vs active_power/rated_mva
PF_DEVIATION_TOL = 0.06           # power_factor vs active/(active^2+reactive^2)
TEMP_FROM_LOAD_TOL = 12.0         # temperature vs 50 + 0.18*load + 0.15*age


# ============================================================
# Exceptions
# ============================================================

class ArtifactGuardError(Exception):
    """Raised when artifact integrity or schema validation fails."""


class TelemetryValidationError(ValueError):
    """Raised when untrusted telemetry fails strict validation."""


class TelemetryBoundaryError(TelemetryValidationError):
    """Raised when provenance cannot be verified and a signal exists."""


# ============================================================
# Manifest generation helpers
# ============================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    model: Path,
    scaler: Path,
    metadata: Path,
    threshold: Path,
) -> Dict[str, Any]:
    """Compute the current SHA-256 manifest for the V2 bundle."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "bundle": "v2",
        "hashes": {
            "model": _sha256(model),
            "scaler": _sha256(scaler),
            "metadata": _sha256(metadata),
            "threshold": _sha256(threshold),
        },
        "paths": {
            "model": model.name,
            "scaler": scaler.name,
            "metadata": metadata.name,
            "threshold": threshold.name,
        },
    }


def write_manifest(
    manifest_path: Path,
    model: Path,
    scaler: Path,
    metadata: Path,
    threshold: Path,
) -> Dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(model, scaler, metadata, threshold)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


# ============================================================
# Manifest verification (fail closed)
# ============================================================

def verify_manifest(
    manifest_path: Path,
    model: Path,
    scaler: Path,
    metadata: Path,
    threshold: Path,
    expected_manifest_version: str = MANIFEST_VERSION,
) -> None:
    """Verify every bound artifact against the manifest. Fail closed."""
    if not manifest_path.is_file():
        raise ArtifactGuardError(
            f"Artifact manifest missing: {manifest_path}"
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactGuardError(
            f"Artifact manifest unreadable: {exc}"
        ) from exc

    if manifest.get("manifest_version") != expected_manifest_version:
        raise ArtifactGuardError(
            "Manifest version mismatch: expected "
            f"{expected_manifest_version}, got "
            f"{manifest.get('manifest_version')}"
        )

    if manifest.get("bundle") != "v2":
        raise ArtifactGuardError(
            f"Manifest bundle mismatch: expected 'v2', got "
            f"{manifest.get('bundle')}"
        )

    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ArtifactGuardError("Manifest 'hashes' missing or invalid.")

    for key in MANIFEST_KEYS:
        expected = hashes.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ArtifactGuardError(
                f"Manifest hash for '{key}' missing or invalid."
            )

    artifacts = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "threshold": threshold,
    }

    for key, path in artifacts.items():
        if not path.is_file():
            raise ArtifactGuardError(
                f"Artifact missing for '{key}': {path}"
            )
        actual = _sha256(path)
        if actual != hashes[key]:
            raise ArtifactGuardError(
                f"Artifact integrity check FAILED for '{key}': {path}"
            )


# ============================================================
# Threshold validation
# ============================================================

def validate_threshold_config(threshold_config: Dict[str, Any]) -> float:
    """Validate threshold is present, finite and within a safe range."""
    raw = threshold_config.get("threshold")
    if raw is None:
        raise ArtifactGuardError("Threshold missing from config.")
    try:
        threshold = float(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactGuardError(
            f"Threshold not a number: {raw!r}"
        ) from exc

    if not np.isfinite(threshold):
        raise ArtifactGuardError(
            f"Threshold is not finite: {threshold}"
        )

    if not (0.0 < threshold < 1.0):
        raise ArtifactGuardError(
            f"Threshold out of safe range (0,1): {threshold}"
        )

    return threshold


# ============================================================
# Scaler validation
# ============================================================

def validate_scaler(
    scaler: Dict[str, Any],
    expected_features: List[str],
) -> None:
    features = scaler.get("features")
    mean = scaler.get("mean")
    std = scaler.get("std")

    if not isinstance(features, list) or not isinstance(mean, list) \
            or not isinstance(std, list):
        raise ArtifactGuardError("Scaler schema invalid.")

    if len(features) != len(expected_features) or \
            len(mean) != len(expected_features) or \
            len(std) != len(expected_features):
        raise ArtifactGuardError("Scaler feature count mismatch.")

    if features != expected_features:
        raise ArtifactGuardError(
            "Scaler feature order mismatch; expected "
            f"{expected_features}, got {features}."
        )

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if not np.isfinite(mean_arr).all() or not np.isfinite(std_arr).all():
        raise ArtifactGuardError("Scaler contains non-finite values.")

    if (std_arr <= 0).any():
        raise ArtifactGuardError("Scaler contains non-positive std.")


# ============================================================
# Metadata / schema validation
# ============================================================

def validate_metadata(
    metadata: Dict[str, Any],
    expected_features: List[str],
) -> None:
    features = metadata.get("input_features")
    if not isinstance(features, list):
        raise ArtifactGuardError("Metadata 'input_features' missing.")
    if features != expected_features:
        raise ArtifactGuardError(
            "Metadata feature order mismatch; expected "
            f"{expected_features}, got {features}."
        )

    count = metadata.get("input_feature_count")
    if count != len(expected_features):
        raise ArtifactGuardError(
            "Metadata feature count mismatch; expected "
            f"{len(expected_features)}, got {count}."
        )


# ============================================================
# State dict schema validation (safe weights_only load)
# ============================================================

def _normalize_key(key: str) -> str:
    if key.startswith("network."):
        return key[len("network."):]
    return key


def validate_model_state_dict(
    state_dict: Dict[str, Any],
    input_size: int = EXPECTED_INPUT_SIZE,
) -> None:
    """Strictly validate state dict keys and shapes. Reject malformed."""
    if not isinstance(state_dict, dict):
        raise ArtifactGuardError(
            "Model checkpoint is not a state_dict dictionary."
        )

    normalized: Dict[str, torch.Tensor] = {}
    seen = set()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ArtifactGuardError(
                f"State dict key not a string: {key!r}"
            )
        nk = _normalize_key(key)
        if nk in seen:
            raise ArtifactGuardError(
                f"Duplicate state dict key after normalization: {nk}"
            )
        seen.add(nk)
        if not isinstance(value, torch.Tensor):
            raise ArtifactGuardError(
                f"State dict value for '{nk}' is not a torch.Tensor."
            )
        normalized[nk] = value

    if set(normalized.keys()) != set(EXPECTED_STATE_DICT_SHAPES.keys()):
        missing = set(EXPECTED_STATE_DICT_SHAPES) - set(normalized)
        extra = set(normalized) - set(EXPECTED_STATE_DICT_SHAPES)
        raise ArtifactGuardError(
            f"State dict key set mismatch. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    for key, expected_shape in EXPECTED_STATE_DICT_SHAPES.items():
        value = normalized[key]
        if tuple(value.shape) != expected_shape:
            raise ArtifactGuardError(
                f"State dict key '{key}' shape {tuple(value.shape)} "
                f"does not match expected {expected_shape}."
            )
        if not torch.isfinite(value).all():
            raise ArtifactGuardError(
                f"State dict key '{key}' contains non-finite values."
            )

    if input_size != EXPECTED_INPUT_SIZE:
        raise ArtifactGuardError(
            f"Model input size {input_size} does not match "
            f"expected {EXPECTED_INPUT_SIZE}."
        )


def load_v2_weights(
    model_path: Path,
    input_size: int = EXPECTED_INPUT_SIZE,
) -> Dict[str, torch.Tensor]:
    """
    Load the V2 model checkpoint safely using weights_only=True and a
    strict schema contract. Never falls back to unsafe pickle loading.
    """
    if not model_path.is_file():
        raise ArtifactGuardError(f"Model file missing: {model_path}")

    try:
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ArtifactGuardError(
            f"Failed to load model checkpoint safely: {exc}"
        ) from exc

    if isinstance(state_dict, torch.nn.Module):
        raise ArtifactGuardError(
            "Checkpoint contains a full Module; expected raw state_dict."
        )

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    validate_model_state_dict(state_dict, input_size=input_size)
    return state_dict


# ============================================================
# Strict telemetry validation
# ============================================================

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_telemetry(
    data: Dict[str, Any],
    required_features: List[str] = STRICT_ML_FEATURES,
    allow_extra: bool = False,
) -> None:
    """
    Strict, deterministic validation of a telemetry dictionary.

    Rejects (with TelemetryValidationError):
        - non-dict input
        - missing required features
        - NaN / Infinity
        - wrong types (no silent coercion to 0.0)
        - out-of-range physical values
        - unexpected fields (unless allow_extra=True)
    """
    if not isinstance(data, dict):
        raise TelemetryValidationError(
            f"Telemetry must be a dict, got {type(data).__name__}."
        )

    for feature in required_features:
        if feature not in data:
            raise TelemetryValidationError(
                f"Missing required telemetry feature: {feature}"
            )

    if not allow_extra:
        allowed = set(required_features) | {
            "previous_faults",
            "fault_type",
            "asset_id",
            "asset_type",
            "timestamp",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise TelemetryValidationError(
                f"Unexpected telemetry fields: {sorted(unexpected)}"
            )

    # Every required feature must be a finite number.
    for feature in required_features:
        value = data.get(feature)
        if not _is_number(value):
            raise TelemetryValidationError(
                f"Telemetry field '{feature}' has invalid type "
                f"{type(value).__name__}; number required."
            )
        fvalue = float(value)
        if not np.isfinite(fvalue):
            raise TelemetryValidationError(
                f"Telemetry field '{feature}' is not finite "
                f"(NaN/Infinity rejected)."
            )

    # Physical range enforcement on known telemetry fields.
    for feature, (lo, hi) in PHYSICAL_RANGES.items():
        if feature not in data:
            continue
        value = data[feature]
        if not _is_number(value):
            raise TelemetryValidationError(
                f"Telemetry field '{feature}' has invalid type "
                f"{type(value).__name__}; number required."
            )
        fvalue = float(value)
        if not np.isfinite(fvalue):
            raise TelemetryValidationError(
                f"Telemetry field '{feature}' is not finite "
                f"(NaN/Infinity rejected)."
            )
        if not (lo <= fvalue <= hi):
            raise TelemetryValidationError(
                f"Telemetry field '{feature}' out of physical range "
                f"[{lo}, {hi}]: {fvalue}."
            )

    # Derived deviation features must match their source features.
    for base, deviation in (
        ("voltage_pu", "voltage_deviation"),
        ("frequency_hz", "frequency_deviation"),
        ("temperature_c", "temperature_excess"),
    ):
        if deviation not in data or base not in data:
            continue
        raw = data[deviation]
        if not _is_number(raw) or not np.isfinite(float(raw)):
            raise TelemetryValidationError(
                f"Telemetry field '{deviation}' is not a finite number."
            )


def _safe_consistency_float(values: Dict[str, Any], key: str) -> Optional[float]:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not np.isfinite(float(value)):
        return None
    return float(value)


def consistency_checks(
    data: Dict[str, Any],
) -> List[str]:
    """
    Return human-readable reasons describing cross-measurement
    inconsistencies, based on the generator's own physical relationships.
    Empty list means the measurements are mutually consistent.
    """
    reasons: List[str] = []

    power_factor = _safe_consistency_float(data, "power_factor")
    active = _safe_consistency_float(data, "active_power_mw")
    reactive = _safe_consistency_float(data, "reactive_power_mvar")
    load = _safe_consistency_float(data, "load_percent")
    rated = _safe_consistency_float(data, "rated_mva")
    temperature = _safe_consistency_float(data, "temperature_c")
    age = _safe_consistency_float(data, "asset_age_years")

    # power_factor should approximate active/sqrt(active^2+reactive^2).
    if (
        power_factor is not None
        and active is not None
        and reactive is not None
        and active > 0.0
    ):
        expected_pf = active / np.sqrt(active ** 2 + reactive ** 2)
        if abs(power_factor - expected_pf) > PF_DEVIATION_TOL:
            reasons.append(
                "Inconsistent power factor vs active/reactive power "
                f"(pf={power_factor:.4f}, expected~{expected_pf:.4f})"
            )

    # load should approximate active_power/(rated_mva*0.92)*100.
    if (
        load is not None
        and active is not None
        and rated is not None
        and rated > 0.0
    ):
        expected_load = (active / (rated * 0.92)) * 100.0
        if abs(load - expected_load) > LOAD_FROM_POWER_TOL:
            reasons.append(
                "Inconsistent load_percent vs active power/rating "
                f"(load={load:.1f}, expected~{expected_load:.1f})"
            )

    # temperature should approximate 50 + 0.18*load + 0.15*age.
    if (
        temperature is not None
        and load is not None
        and age is not None
    ):
        expected_temp = 50.0 + (load * 0.18) + (age * 0.15)
        if abs(temperature - expected_temp) > TEMP_FROM_LOAD_TOL:
            reasons.append(
                "Inconsistent temperature vs load/age "
                f"(temp={temperature:.1f}, expected~{expected_temp:.1f})"
            )

    return reasons


def is_severe_physical(operational_data: Dict[str, Any]) -> List[str]:
    """Return severe physical condition reasons using project thresholds."""
    reasons: List[str] = []

    temperature = _safe_consistency_float(operational_data, "temperature_c")
    load = _safe_consistency_float(operational_data, "load_percent")
    thd = _safe_consistency_float(operational_data, "thd_percent")
    voltage = _safe_consistency_float(operational_data, "voltage_pu")
    frequency = _safe_consistency_float(operational_data, "frequency_hz")

    if temperature is not None and temperature >= 100:
        reasons.append("Severe temperature condition")
    if load is not None and load >= 90:
        reasons.append("Severe loading condition")
    if thd is not None and thd >= 10:
        reasons.append("Severe harmonic distortion")
    if voltage is not None and abs(voltage - 1.0) >= 0.05:
        reasons.append("Severe voltage deviation")
    if frequency is not None and abs(frequency - 50.0) >= 0.2:
        reasons.append("Severe frequency deviation")

    return reasons
