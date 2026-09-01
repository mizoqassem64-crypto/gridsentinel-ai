import json
from pathlib import Path

import numpy as np
import torch

from ai.ml.safety_guard_v2 import apply_safety_guard
from ai.ml.artifact_guard import (
    ArtifactGuardError,
    load_v2_weights,
    verify_manifest,
    validate_threshold_config,
    validate_scaler,
    validate_metadata,
    validate_telemetry,
    consistency_checks,
    is_severe_physical,
)


# ============================================================
# GridSentinel AI - V2 Risk Intelligence Engine
# ============================================================

print("=" * 70)
print("GridSentinel AI - V2 Risk Intelligence Engine")
print("=" * 70)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "failure_predictor_v2.pt"
SCALER_PATH = ROOT / "models" / "failure_scaler_v2.json"
METADATA_PATH = ROOT / "models" / "failure_model_metadata_v2.json"
THRESHOLD_PATH = ROOT / "models" / "failure_threshold_v2.json"
MANIFEST_PATH = ROOT / "models" / "v2_artifact_manifest.json"


# ============================================================
# Verify artifact integrity (fail closed)
# ============================================================

try:
    verify_manifest(
        manifest_path=MANIFEST_PATH,
        model=MODEL_PATH,
        scaler=SCALER_PATH,
        metadata=METADATA_PATH,
        threshold=THRESHOLD_PATH,
    )
except ArtifactGuardError as exc:
    raise RuntimeError(
        f"V2 artifact integrity verification failed: {exc}"
    ) from exc


# ============================================================
# Load production artifacts
# ============================================================

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

with open(SCALER_PATH, "r", encoding="utf-8") as f:
    SCALER = json.load(f)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    THRESHOLD_CONFIG = json.load(f)


FEATURES = METADATA["input_features"]
FEATURE_COUNT = len(FEATURES)

# Strict schema/version/feature-order validation.
validate_metadata(METADATA, FEATURES)
validate_scaler(SCALER, FEATURES)

PRODUCTION_THRESHOLD = validate_threshold_config(
    THRESHOLD_CONFIG
)


# ============================================================
# Model architecture
# ============================================================

def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.20),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


# ============================================================
# Load model (safe: weights_only=True + strict schema contract)
# ============================================================

STATE_DICT = load_v2_weights(MODEL_PATH)

MODEL = build_model()

MODEL.load_state_dict(
    {
        k.removeprefix("network."): v
        for k, v in STATE_DICT.items()
    }
)

MODEL.eval()


# ============================================================
# Scaler
# ============================================================

SCALER_MEAN = np.asarray(
    SCALER["mean"],
    dtype=np.float32,
)

SCALER_STD = np.asarray(
    SCALER["std"],
    dtype=np.float32,
)


if len(SCALER_MEAN) != FEATURE_COUNT:
    raise RuntimeError(
        "Scaler mean feature count mismatch."
    )

if len(SCALER_STD) != FEATURE_COUNT:
    raise RuntimeError(
        "Scaler std feature count mismatch."
    )


# ============================================================
# Utilities
# ============================================================

def clamp(
    value,
    minimum=0.0,
    maximum=100.0,
):
    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# ML Prediction
# ============================================================

def predict_v2(data):
    """
    Run production FailurePredictorV2.

    Input:
        Dictionary containing the exact 16 V2 features.

    Returns:
        failure probability
        prediction status
        threshold

    Raises:
        TelemetryValidationError on missing/invalid/out-of-range input.
    """

    # allow_extra=True: callers may pass enriched dataset rows containing
    # legitimate context columns (timestamp, asset metadata, derived
    # features). Required ML features are still strictly validated for
    # presence, type and physical range. Use validate_telemetry() with
    # allow_extra=False for a strict external schema boundary.
    validate_telemetry(
        data,
        required_features=FEATURES,
        allow_extra=True,
    )

    values = np.asarray(
        [
            float(data[feature])
            for feature in FEATURES
        ],
        dtype=np.float32,
    )

    scaled = (
        values - SCALER_MEAN
    ) / SCALER_STD

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        logits = MODEL(tensor)

        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    status = (
        "FAILURE"
        if probability >= PRODUCTION_THRESHOLD
        else "NORMAL"
    )

    return {
        "failure_probability": probability,
        "threshold": PRODUCTION_THRESHOLD,
        "status": status,
    }


# ============================================================
# Operational Risk
# ============================================================

def calculate_operational_risk(data):

    score = 0.0
    reasons = []

    # NOTE: telemetry is expected to have been validated by
    # validate_telemetry() before this function is reached. These
    # reads are strict: invalid values raise instead of coercing to 0.
    criticality = float(data["criticality"])
    temperature = float(data["temperature_c"])
    load = float(data["load_percent"])
    thd = float(data["thd_percent"])
    voltage = float(data.get("voltage_pu", 1.0))
    frequency = float(data.get("frequency_hz", 50.0))
    power_factor = float(data.get("power_factor", 1.0))
    previous_faults = float(data.get("previous_faults", 0.0))

    fault_type = str(
        data.get(
            "fault_type",
            "normal",
        )
    )


    # --------------------------------------------------------
    # Criticality
    # --------------------------------------------------------

    if criticality >= 1.0:

        score += 15
        reasons.append(
            "Critical asset"
        )

    elif criticality >= 0.9:

        score += 10
        reasons.append(
            "High asset criticality"
        )

    else:

        score += 5


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if temperature >= 100:

        score += 20
        reasons.append(
            "Severe overheating"
        )

    elif temperature >= 85:

        score += 12
        reasons.append(
            "Elevated temperature"
        )

    elif temperature >= 75:

        score += 6
        reasons.append(
            "Temperature above normal"
        )


    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    if load >= 90:

        score += 20
        reasons.append(
            "Severe overload"
        )

    elif load >= 80:

        score += 12
        reasons.append(
            "High loading"
        )

    elif load >= 70:

        score += 6
        reasons.append(
            "Elevated loading"
        )


    # --------------------------------------------------------
    # Harmonic distortion
    # --------------------------------------------------------

    if thd >= 10:

        score += 15
        reasons.append(
            "Severe harmonic distortion"
        )

    elif thd >= 5:

        score += 8
        reasons.append(
            "Elevated harmonic distortion"
        )


    # --------------------------------------------------------
    # Voltage stability
    # --------------------------------------------------------

    voltage_deviation = abs(
        voltage - 1.0
    )

    if voltage_deviation >= 0.05:

        score += 15
        reasons.append(
            "Significant voltage instability"
        )

    elif voltage_deviation >= 0.03:

        score += 8
        reasons.append(
            "Voltage deviation detected"
        )


    # --------------------------------------------------------
    # Frequency stability
    # --------------------------------------------------------

    frequency_deviation = abs(
        frequency - 50.0
    )

    if frequency_deviation >= 0.2:

        score += 10
        reasons.append(
            "Significant frequency deviation"
        )

    elif frequency_deviation >= 0.1:

        score += 5
        reasons.append(
            "Frequency deviation detected"
        )


    # --------------------------------------------------------
    # Power factor
    # --------------------------------------------------------

    if power_factor < 0.85:

        score += 10
        reasons.append(
            "Low power factor"
        )

    elif power_factor < 0.90:

        score += 5
        reasons.append(
            "Reduced power factor"
        )


    # --------------------------------------------------------
    # Historical faults
    # --------------------------------------------------------

    if previous_faults >= 5:

        score += 10
        reasons.append(
            "High previous fault count"
        )

    elif previous_faults >= 2:

        score += 5
        reasons.append(
            "Previous faults detected"
        )


    # --------------------------------------------------------
    # Active fault type
    # --------------------------------------------------------

    fault_weights = {
        "normal": 0,
        "overload": 8,
        "voltage_instability": 8,
        "harmonic_distortion": 8,
        "overheating": 10,
    }

    fault_score = fault_weights.get(
        fault_type,
        0,
    )

    if fault_score:

        score += fault_score

        reasons.append(
            f"Active fault: {fault_type}"
        )


    return {
        "score": clamp(score),
        "reasons": reasons,
    }


# ============================================================
# Risk Level
# ============================================================

def calculate_risk_level(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    return "LOW"


# ============================================================
# Recommended Action
# ============================================================

def recommended_action(
    level,
    prediction_status,
):

    if level == "CRITICAL":

        return (
            "Immediate inspection and "
            "controlled shutdown assessment."
        )

    if level == "HIGH":

        return (
            "Dispatch maintenance team and "
            "investigate asset condition."
        )

    if level == "MEDIUM":

        return (
            "Increase monitoring frequency and "
            "schedule inspection."
        )

    if prediction_status == "FAILURE":

        return (
            "Investigate predicted failure condition."
        )

    return (
        "Continue normal operation with "
        "routine monitoring."
    )


# ============================================================
# V2 Trust Boundary
# ============================================================

def _apply_trust_boundary(data, trusted_source):
    """
    Determine whether telemetry can be trusted for autonomous
    LOW/NORMAL classification.

    Untrusted telemetry is escalated to INVESTIGATION whenever there is:
        - a severe physical condition
        - a cross-measurement inconsistency
        - an elevated ML failure probability (>= 0.50, WATCH band)
        - a non-trivial previous fault history reported by the client

    Provenance cannot be established cryptographically here; this layer
    only guarantees that unverifiable abnormal telemetry fails safe.
    """
    reasons = []

    if trusted_source:
        return {
            "applied": False,
            "investigate": False,
            "reasons": [],
        }

    boundary_reasons = list(
        is_severe_physical(data)
    )

    inconsistency_reasons = consistency_checks(data)

    if inconsistency_reasons:
        boundary_reasons.extend(
            [
                "Telemetry inconsistency: " + r
                for r in inconsistency_reasons
            ]
        )

    previous_faults = data.get("previous_faults", 0.0)
    if isinstance(previous_faults, (int, float)) \
            and not isinstance(previous_faults, bool) \
            and float(previous_faults) >= 2.0:
        boundary_reasons.append(
            "Client-reported historical fault count "
            "cannot be trusted without provenance"
        )

    investigate = bool(boundary_reasons)

    for r in boundary_reasons:
        reasons.append(
            "Trust boundary: " + r
        )

    return {
        "applied": investigate,
        "investigate": investigate,
        "reasons": reasons,
    }


# ============================================================
# V2 Risk Assessment
# ============================================================


def assess_risk_v2(data, trusted_source=False):
    """
    Complete GridSentinel V2.1 risk assessment.

    Parameters:
        data: validated telemetry dictionary.
        trusted_source: False unless the caller can prove authenticated,
            server-collected provenance. Untrusted telemetry is treated
            as unsafe for autonomous LOW/NORMAL classification whenever
            any abnormal signal, inconsistency, or high ML probability
            is present (fail-safe to INVESTIGATION instead).

    Pipeline:
        1. V2 ML prediction
        2. Operational risk calculation
        3. Weighted risk score
        4. Production failure floor
        5. V2.1 Safety Guard
        6. Final risk classification
        7. Engineering interpretation/action
    """

    # ============================================================
    # 0. STRICT TELEMETRY VALIDATION + TRUST BOUNDARY
    # ============================================================

    validate_telemetry(
        data,
        required_features=FEATURES,
        allow_extra=True,
    )

    trust_boundary = _apply_trust_boundary(
        data=data,
        trusted_source=bool(trusted_source),
    )

    # ============================================================
    # 1. ML PREDICTION
    # ============================================================

    prediction = predict_v2(data)

    probability = float(
        prediction["failure_probability"]
    )

    prediction_status = prediction["status"]

    # ============================================================
    # 2. OPERATIONAL RISK
    # ============================================================

    operational = calculate_operational_risk(data)

    operational_score = float(
        operational["score"]
    )

    # ============================================================
    # 3. WEIGHTED RISK SCORE
    # ============================================================

    # ML contributes 60%
    ml_score = probability * 60.0

    # Operational engine contributes 40%
    operational_component = operational_score * 0.40

    total_score = clamp(
        ml_score + operational_component
    )

    # ============================================================
    # 4. PRODUCTION FAILURE FLOOR
    # ============================================================

    # A production ML failure prediction must never
    # become a LOW risk classification.
    if prediction_status == "FAILURE":
        total_score = max(
            total_score,
            70.0,
        )

    # ============================================================
    # 5. INITIAL RISK LEVEL
    # ============================================================

    initial_risk_level = calculate_risk_level(
        total_score
    )

    # ============================================================
    # 6. SAFETY GUARD V2.1
    # ============================================================

    guard_result = apply_safety_guard(
        risk_score=total_score,
        risk_level=initial_risk_level,
        failure_probability=probability,
        operational_data=data,
    )
    # Safety Guard may adjust the final state.
    final_score = float(
        guard_result.get(
            "risk_score",
            total_score,
        )
    )

    final_risk_level = guard_result.get(
        "risk_level",
        calculate_risk_level(final_score),
    )

    alert_state = guard_result.get(
        "alert_state",
        "NORMAL",
    )

    guard_applied = bool(
        guard_result.get(
            "guard_applied",
            False,
        )
    )

    guard_reasons = guard_result.get(
        "guard_reasons",
        [],
    )

    # ============================================================
    # 7. COMBINE REASONS
    # ============================================================

    reasons = list(
        operational.get(
            "reasons",
            [],
        )
    )

    for reason in guard_reasons:
        if reason not in reasons:
            reasons.append(reason)

    # ============================================================
    # 8. FINAL ACTION
    # ============================================================

    action = recommended_action(
        final_risk_level,
        prediction_status,
    )

    # Safety alert gets priority over ordinary action.
    if alert_state == "FAILURE_ALERT":
        action = (
            "Failure condition detected. "
            "Immediate engineering investigation "
            "and controlled shutdown assessment required."
        )

    # ============================================================
    # 9. INTERPRETATION
    # ============================================================

    if final_risk_level == "CRITICAL":
        interpretation = (
            "Critical operational risk. "
            "Immediate engineering intervention "
            "is recommended."
        )

    elif final_risk_level == "HIGH":
        interpretation = (
            "High operational risk. "
            "Maintenance investigation should "
            "be prioritized."
        )

    elif final_risk_level == "MEDIUM":
        interpretation = (
            "Moderate operational risk. "
            "Increase monitoring and schedule "
            "preventive inspection."
        )

    else:
        interpretation = (
            "Low operational risk under the "
            "current operating conditions."
        )

    # ============================================================
    # 9b. TRUST BOUNDARY ENFORCEMENT
    # ============================================================
    # Untrusted telemetry with any abnormal signal, physical severity,
    # inconsistency, or elevated ML probability is escalated to an
    # INVESTIGATION state and must never silently become LOW/NORMAL.
    # This does not pretend cryptographic trust; it fails safe.

    boundary_reasons = list(
        trust_boundary.get("reasons", [])
    )
    boundary_applied = bool(
        trust_boundary.get("applied", False)
    )

    investigate = bool(
        trust_boundary.get("investigate", False)
    )

    # Elevated ML probability from untrusted telemetry can never be
    # treated as NORMAL; route to investigation instead.
    if (
        not trusted_source
        and probability >= 0.50
        and alert_state != "FAILURE_ALERT"
    ):
        investigate = True
        reason = (
            "Trust boundary: elevated ML failure probability "
            "from unverified telemetry"
        )
        if reason not in boundary_reasons:
            boundary_reasons.append(reason)

    if investigate:
        alert_state = "INVESTIGATION"
        boundary_applied = True

        for reason in boundary_reasons:
            if reason not in guard_reasons:
                guard_reasons.append(reason)
            if reason not in reasons:
                reasons.append(reason)

    # ============================================================
    # 9c. INVESTIGATION ACTION / INTERPRETATION
    # ============================================================

    if alert_state == "INVESTIGATION":
        action = (
            "Telemetry provenance could not be verified while "
            "abnormal or inconsistent signals are present. "
            "Engineering investigation required before relying "
            "on a LOW/NORMAL assessment."
        )
        interpretation = (
            "Unverified telemetry with abnormal signals. "
            "Cannot safely classify as normal; investigation required."
        )

    # ============================================================
    # 10. FINAL RESULT
    # ============================================================

    return {
        "model": "FailurePredictorV2",
        "version": "2.1",

        "failure_probability": round(
            probability,
            6,
        ),

        "threshold": round(
            PRODUCTION_THRESHOLD,
            6,
        ),

        "prediction": prediction_status,

        "risk_score": round(
            final_score,
            2,
        ),

        "risk_level": final_risk_level,

        "ml_score": round(
            ml_score,
            2,
        ),

        "operational_score": round(
            operational_score,
            2,
        ),

        "alert_state": alert_state,

        "guard_applied": guard_applied,

        "guard_reasons": guard_reasons,

        "reasons": reasons,

        "interpretation": interpretation,

        "recommended_action": action,

        "telemetry_trusted": trusted_source,

        "trust_boundary_applied": boundary_applied,

        "trust_boundary_reasons": boundary_reasons,
    }
