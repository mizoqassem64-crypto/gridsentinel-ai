from typing import Dict


def apply_safety_guard(
    risk_score: float,
    risk_level: str,
    failure_probability: float,
    operational_data: Dict,
) -> Dict:
    """
    V2.1 Safety Guard.

    Purpose:
        Prevent the risk engine from underestimating
        severe physical operating conditions.

    This layer does NOT modify the ML probability.
    It only protects the final operational risk decision.
    """

    score = float(risk_score)
    level = str(risk_level)

    temperature = float(
        operational_data.get("temperature_c", 0.0)
    )

    load = float(
        operational_data.get("load_percent", 0.0)
    )

    thd = float(
        operational_data.get("thd_percent", 0.0)
    )

    voltage = float(
        operational_data.get("voltage_pu", 1.0)
    )

    frequency = float(
        operational_data.get("frequency_hz", 50.0)
    )

    previous_faults = float(
        operational_data.get("previous_faults", 0.0)
    )

    guard_reasons = []

    # ============================================================
    # Severe physical conditions
    # ============================================================

    severe_temperature = temperature >= 100
    severe_loading = load >= 90
    severe_thd = thd >= 10
    severe_voltage = abs(voltage - 1.0) >= 0.05
    severe_frequency = abs(frequency - 50.0) >= 0.2

    severe_conditions = sum(
        [
            severe_temperature,
            severe_loading,
            severe_thd,
            severe_voltage,
            severe_frequency,
        ]
    )

    # ============================================================
    # Safety escalation
    # ============================================================

    if severe_temperature:
        guard_reasons.append(
            "Safety guard: severe temperature condition"
        )

    if severe_loading:
        guard_reasons.append(
            "Safety guard: severe loading condition"
        )

    if severe_thd:
        guard_reasons.append(
            "Safety guard: severe harmonic distortion"
        )

    if severe_voltage:
        guard_reasons.append(
            "Safety guard: severe voltage deviation"
        )

    if severe_frequency:
        guard_reasons.append(
            "Safety guard: severe frequency deviation"
        )

    # Two or more severe physical conditions
    # should never remain LOW.

    if severe_conditions >= 2:
        score = max(score, 50.0)

        if level == "LOW":
            level = "HIGH"

    # One severe physical condition
    # should prevent LOW classification.

    elif severe_conditions == 1:
        score = max(score, 30.0)

        if level == "LOW":
            level = "MEDIUM"

    # ============================================================
    # Historical risk protection
    # ============================================================

    if previous_faults >= 5:

        guard_reasons.append(
            "Safety guard: repeated historical faults"
        )

        score = max(score, 35.0)

        if level == "LOW":
            level = "MEDIUM"

    # ============================================================
    # High ML confidence watch
    # ============================================================

    watch_state = "NORMAL"

    if (
        failure_probability >= 0.50
        and level in ("LOW", "MEDIUM")
    ):
        watch_state = "WATCH"

        guard_reasons.append(
            "Safety guard: elevated ML failure probability"
        )

    # ============================================================
    # ML high-confidence failure
    # ============================================================

    if failure_probability >= 0.70:

        score = max(score, 70.0)

        if level not in ("CRITICAL",):
            level = "HIGH"

        watch_state = "FAILURE_ALERT"

        guard_reasons.append(
            "Safety guard: production failure threshold exceeded"
        )

    # ============================================================
    # Critical physical condition
    # ============================================================

    if severe_conditions >= 3:

        score = max(score, 75.0)
        level = "CRITICAL"

        guard_reasons.append(
            "Safety guard: multiple severe operating conditions"
        )

    return {
        "risk_score": round(min(score, 100.0), 2),
        "risk_level": level,
        "alert_state": watch_state,
        "guard_applied": bool(guard_reasons),
        "guard_reasons": guard_reasons,
    }
