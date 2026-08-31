import json
from pathlib import Path

import numpy as np
import torch

from ai.ml.safety_guard_v2 import apply_safety_guard


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


# ============================================================
# Load production artifacts
# ============================================================

with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)

with open(SCALER_PATH, "r") as f:
    SCALER = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD_CONFIG = json.load(f)


FEATURES = METADATA["input_features"]
FEATURE_COUNT = len(FEATURES)

PRODUCTION_THRESHOLD = float(
    THRESHOLD_CONFIG["threshold"]
)


# ============================================================
# Model architecture
# ============================================================

def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.20),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

if isinstance(checkpoint, torch.nn.Module):

    MODEL = checkpoint

else:

    MODEL = build_model()

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

        # V2 training saves keys as:
        # network.0.weight
        #
        # Sequential expects:
        # 0.weight
        #
        # Normalize automatically.

        normalized_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("network."):
                new_key = key.replace(
                    "network.",
                    "",
                    1,
                )
            else:
                new_key = key

            normalized_state_dict[new_key] = value

        MODEL.load_state_dict(
            normalized_state_dict
        )

    else:
        raise RuntimeError(
            "Unsupported V2 model checkpoint format."
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


def safe_float(
    data,
    key,
    default=0.0,
):
    try:
        return float(
            data.get(key, default)
        )
    except (
        TypeError,
        ValueError,
    ):
        return float(default)


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
    """

    missing = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing:
        raise ValueError(
            "Missing V2 features: "
            + ", ".join(missing)
        )

    values = np.asarray(
        [
            safe_float(
                data,
                feature,
            )
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

    criticality = safe_float(
        data,
        "criticality",
    )

    temperature = safe_float(
        data,
        "temperature_c",
    )

    load = safe_float(
        data,
        "load_percent",
    )

    thd = safe_float(
        data,
        "thd_percent",
    )

    voltage = safe_float(
        data,
        "voltage_pu",
        1.0,
    )

    frequency = safe_float(
        data,
        "frequency_hz",
        50.0,
    )

    power_factor = safe_float(
        data,
        "power_factor",
        1.0,
    )

    previous_faults = safe_float(
        data,
        "previous_faults",
    )

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
# V2 Risk Assessment
# ============================================================


def assess_risk_v2(data):
    """
    Complete GridSentinel V2.1 risk assessment.

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
    }
