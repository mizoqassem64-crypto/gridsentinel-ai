import json
from pathlib import Path

from ai.ml.predict import predict


ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_PATH = ROOT / "models" / "failure_threshold.json"

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD_CONFIG = json.load(f)


PRODUCTION_THRESHOLD = float(THRESHOLD_CONFIG["threshold"])


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, value))


def calculate_operational_risk(data):
    """
    Calculate operational risk independently from the ML probability.

    Expected input:
        criticality
        temperature_c
        load_percent
        thd_percent
        voltage_pu
        frequency_hz
        power_factor
        previous_faults
        fault_type
    """

    score = 0.0
    reasons = []

    criticality = float(data.get("criticality", 0.0))
    temperature = float(data.get("temperature_c", 0.0))
    load = float(data.get("load_percent", 0.0))
    thd = float(data.get("thd_percent", 0.0))
    voltage = float(data.get("voltage_pu", 1.0))
    frequency = float(data.get("frequency_hz", 50.0))
    power_factor = float(data.get("power_factor", 1.0))
    previous_faults = float(data.get("previous_faults", 0.0))
    fault_type = str(data.get("fault_type", "normal"))

    # Criticality
    if criticality >= 1.0:
        score += 15
        reasons.append("Critical asset")
    elif criticality >= 0.9:
        score += 10
        reasons.append("High asset criticality")
    else:
        score += 5

    # Temperature
    if temperature >= 100:
        score += 20
        reasons.append("Severe overheating")
    elif temperature >= 85:
        score += 12
        reasons.append("Elevated temperature")
    elif temperature >= 75:
        score += 6
        reasons.append("Temperature above normal")

    # Loading
    if load >= 90:
        score += 20
        reasons.append("Severe overload")
    elif load >= 80:
        score += 12
        reasons.append("High loading")
    elif load >= 70:
        score += 6
        reasons.append("Elevated loading")

    # THD
    if thd >= 10:
        score += 15
        reasons.append("Severe harmonic distortion")
    elif thd >= 5:
        score += 8
        reasons.append("Elevated harmonic distortion")

    # Voltage
    voltage_deviation = abs(voltage - 1.0)

    if voltage_deviation >= 0.05:
        score += 15
        reasons.append("Significant voltage instability")
    elif voltage_deviation >= 0.03:
        score += 8
        reasons.append("Voltage deviation detected")

    # Frequency
    frequency_deviation = abs(frequency - 50.0)

    if frequency_deviation >= 0.2:
        score += 10
        reasons.append("Significant frequency deviation")
    elif frequency_deviation >= 0.1:
        score += 5
        reasons.append("Frequency deviation detected")

    # Power factor
    if power_factor < 0.85:
        score += 10
        reasons.append("Low power factor")
    elif power_factor < 0.90:
        score += 5
        reasons.append("Reduced power factor")

    # Historical faults
    if previous_faults >= 5:
        score += 10
        reasons.append("High previous fault count")
    elif previous_faults >= 2:
        score += 5
        reasons.append("Previous faults detected")

    # Known fault classification
    fault_weights = {
        "normal": 0,
        "overload": 8,
        "voltage_instability": 8,
        "harmonic_distortion": 8,
        "overheating": 10,
    }

    fault_score = fault_weights.get(fault_type, 0)

    if fault_score:
        score += fault_score
        reasons.append(f"Active fault: {fault_type}")

    return {
        "score": clamp(score),
        "reasons": reasons,
    }


def calculate_risk_level(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def recommended_action(level, prediction_status):
    if level == "CRITICAL":
        return "Immediate inspection and controlled shutdown assessment."

    if level == "HIGH":
        return "Dispatch maintenance team and investigate asset condition."

    if level == "MEDIUM":
        return "Increase monitoring frequency and schedule inspection."

    if prediction_status == "FAILURE":
        return "Investigate predicted failure condition."

    return "Continue normal operation with routine monitoring."


def assess_risk(values, operational_data):
    """
    Full GridSentinel risk assessment.

    values:
        ML feature vector in the exact order expected by predict()

    operational_data:
        Original operational/context fields.
    """

    prediction = predict(values)

    probability = float(prediction["failure_probability"])

    operational = calculate_operational_risk(operational_data)

    # ML probability contributes up to 60 points.
    ml_score = probability * 60.0

    # Operational engine contributes up to 40 points.
    operational_score = operational["score"] * 0.40

    total_score = clamp(ml_score + operational_score)

    # A production ML failure prediction should not be hidden
    # by a low operational score.
    if prediction["status"] == "FAILURE":
        total_score = max(total_score, 70.0)

    risk_level = calculate_risk_level(total_score)

    action = recommended_action(
        risk_level,
        prediction["status"],
    )

    return {
        "failure_probability": round(probability, 6),
        "threshold": PRODUCTION_THRESHOLD,
        "prediction": prediction["status"],
        "risk_score": round(total_score, 2),
        "risk_level": risk_level,
        "operational_score": round(operational["score"], 2),
        "reasons": operational["reasons"],
        "recommended_action": action,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("GridSentinel AI - Risk Engine")
    print("=" * 70)

    sample = {
        "rated_mva": 50,
        "asset_age_years": 12,
        "criticality": 0.9,
        "voltage_pu": 0.96,
        "current_a": 650,
        "frequency_hz": 49.9,
        "active_power_mw": 35,
        "reactive_power_mvar": 12,
        "power_factor": 0.87,
        "temperature_c": 95,
        "load_percent": 88,
        "thd_percent": 8,
    }

    operational_data = {
        **sample,
        "previous_faults": 3,
        "fault_type": "overheating",
    }

    values = [
        sample["rated_mva"],
        sample["asset_age_years"],
        sample["criticality"],
        sample["voltage_pu"],
        sample["current_a"],
        sample["frequency_hz"],
        sample["active_power_mw"],
        sample["reactive_power_mvar"],
        sample["power_factor"],
        sample["temperature_c"],
        sample["load_percent"],
        sample["thd_percent"],
    ]

    result = assess_risk(values, operational_data)

    print("\n[RESULT]")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
