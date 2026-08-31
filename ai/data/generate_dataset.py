import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


SEED = 42
random.seed(SEED)

OUTPUT = Path("datasets/grid_operations.csv")

TRANSFORMERS = [
    {"id": "T01", "mva": 40.0, "age": 6, "criticality": 0.90},
    {"id": "T02", "mva": 63.0, "age": 11, "criticality": 1.00},
    {"id": "T03", "mva": 40.0, "age": 17, "criticality": 0.85},
]

FAULTS = [
    "normal",
    "overload",
    "overheating",
    "voltage_instability",
    "harmonic_distortion",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def gaussian(mean, std):
    return random.gauss(mean, std)


def generate_normal(transformer):
    """Generate a healthy operating point."""

    load = clamp(gaussian(68, 8), 35, 82)

    temperature = (
        50
        + (load * 0.18)
        + (transformer["age"] * 0.15)
        + gaussian(0, 2.0)
    )

    voltage = gaussian(1.0, 0.008)
    frequency = gaussian(50.0, 0.025)

    current = (
        transformer["mva"]
        * 1_000_000
        / (math.sqrt(3) *33_000)
    )

    current *= load / 100
    current *= random.uniform(0.97, 1.03)

    active_power = transformer["mva"] * (load / 100) * 0.92
    reactive_power = active_power * random.uniform(0.25, 0.40)

    power_factor = clamp(
        active_power
        / math.sqrt(active_power**2 + reactive_power**2),
        0.85,
        0.99,
    )

    thd = clamp(gaussian(2.5, 0.5), 1.0, 4.0)

    return {
        "load": load,
        "temperature": temperature,
        "voltage": voltage,
        "frequency": frequency,
        "current": current,
        "active_power": active_power,
        "reactive_power": reactive_power,
        "power_factor": power_factor,
        "thd": thd,
    }


def apply_fault(state, fault, severity):
    """Apply a controlled degradation pattern."""

    if fault == "overload":
        state["load"] += 18 * severity
        state["current"] *= 1 + (0.25 * severity)
        state["temperature"] += 30 * severity
        state["voltage"] -= 0.03 * severity
        state["power_factor"] -= 0.04 * severity
        state["thd"] += 2.5 * severity

    elif fault == "overheating":
        state["temperature"] += 45 * severity
        state["power_factor"] -= 0.025 * severity

    elif fault == "voltage_instability":
        state["voltage"] -= 0.08 * severity
        state["frequency"] -= 0.15 * severity
        state["thd"] += 3 * severity

    elif fault == "harmonic_distortion":
        state["thd"] += 12 * severity
        state["temperature"] += 8 * severity
        state["power_factor"] -= 0.05 * severity

    return state


def classify_state(fault, severity):
    if fault == "normal":
        return "healthy"

    if severity < 0.30:
        return "early_degradation"

    if severity < 0.65:
        return "anomaly"

    if severity < 0.90:
        return "high_risk"

    return "failure"


def generate_record(transformer, timestamp):
    normal = generate_normal(transformer)

    # Most records are healthy; a smaller percentage contains degradation.
    fault_probability = random.random()

    if fault_probability < 0.70:
        fault = "normal"
        severity = 0.0
    else:
        fault = random.choice(FAULTS[1:])
        severity = random.uniform(0.10, 1.0)

    state = apply_fault(normal, fault, severity)

    operating_state = classify_state(fault, severity)

    # Failure label is intentionally conservative.
    failure = int(
        operating_state == "failure"
        or (
            state["temperature"] >= 95
            and state["load"] >= 90
        )
    )

    if failure:
        failure_horizon = random.randint(1, 24)
    elif operating_state == "high_risk":
        failure_horizon = random.randint(12, 72)
    elif operating_state == "anomaly":
        failure_horizon = random.randint(72, 240)
    else:
        failure_horizon = 0

    previous_faults = min(
        6,
        max(
            0,
            int(
                gaussian(
                    transformer["age"] / 6,
                    1.5,
                )
            ),
        ),
    )

    return {
        "timestamp": timestamp.isoformat(),
        "asset_id": transformer["id"],
        "asset_type": "transformer",
        "rated_mva": transformer["mva"],
        "asset_age_years": transformer["age"],
        "criticality": transformer["criticality"],
        "voltage_pu": round(state["voltage"], 4),
        "current_a": round(state["current"], 2),
        "frequency_hz": round(state["frequency"], 3),
        "active_power_mw": round(state["active_power"], 3),
        "reactive_power_mvar": round(state["reactive_power"], 3),
        "power_factor": round(
            clamp(state["power_factor"], 0.70, 0.99),
            4,
        ),
        "temperature_c": round(state["temperature"], 2),
        "load_percent": round(
            clamp(state["load"], 30, 115),
            2,
        ),
        "thd_percent": round(
            clamp(state["thd"], 1, 20),
            3,
        ),
        "previous_faults": previous_faults,
        "operating_state": operating_state,
        "fault_type": fault,
        "failure": failure,
        "failure_horizon_hours": failure_horizon,
    }


def generate_dataset(records_per_transformer=17000):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "asset_id",
        "asset_type",
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
        "previous_faults",
        "operating_state",
        "fault_type",
        "failure",
        "failure_horizon_hours",
    ]

    start_time = datetime(2026, 1, 1, 0, 0)

    total = 0

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for transformer in TRANSFORMERS:
            for i in range(records_per_transformer):
                timestamp = start_time + timedelta(minutes=15 * i)

                record = generate_record(
                    transformer,
                    timestamp,
                )

                writer.writerow(record)
                total += 1

    print("=" * 60)
    print("GridSentinel AI - Dataset Generator")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Records: {total}")
    print(f"Transformers: {len(TRANSFORMERS)}")
    print(f"Features: {len(fieldnames)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_dataset()
