import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GridSentinel AI - V2.2 Risk Calibration
# ============================================================

print("=" * 70)
print("GridSentinel AI - V2.2 Risk Calibration Analysis")
print("=" * 70)


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "datasets" / "grid_features.csv"
EVALUATION_PATH = ROOT / "models" / "v2_risk_engine_evaluation.csv"
OUTPUT_PATH = ROOT / "models" / "v2_risk_calibration_v2.json"


# ============================================================
# Configuration
# ============================================================

TARGET = "actual_failure"

PROBABILITY_WEIGHT = 0.60
OPERATIONAL_WEIGHT = 0.40


# ============================================================
# Load data
# ============================================================

print("\nLoading evaluation results...")

df = pd.read_csv(EVALUATION_PATH)

print(f"Records: {len(df):,}")


required = [
    "actual_failure",
    "failure_probability",
    "risk_score",
    "risk_level",
    "operational_score",
    "fault_type",
    "asset_id",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Physical risk signals
# ============================================================

print("\nLoading dataset physical conditions...")

raw = pd.read_csv(DATASET_PATH)

physical_columns = [
    "temperature_c",
    "load_percent",
    "thd_percent",
    "voltage_pu",
    "frequency_hz",
    "power_factor",
    "previous_faults",
    "criticality",
]

missing_physical = [
    column
    for column in physical_columns
    if column not in raw.columns
]

if missing_physical:
    raise RuntimeError(
        f"Missing physical columns: {missing_physical}"
    )


# Align by row if lengths match.
if len(raw) != len(df):
    raise RuntimeError(
        "Dataset and evaluation row counts do not match."
    )

for column in physical_columns:
    df[column] = pd.to_numeric(
        raw[column],
        errors="coerce",
    )


# ============================================================
# Helpers
# ============================================================

def calculate_physical_modifier(row):
    """
    V2.2 physical-condition modifier.

    This does NOT change the ML probability.
    It only adjusts the operational component.
    """

    modifier = 0.0

    temperature = float(row["temperature_c"])
    load = float(row["load_percent"])
    thd = float(row["thd_percent"])
    voltage = float(row["voltage_pu"])
    frequency = float(row["frequency_hz"])
    power_factor = float(row["power_factor"])
    previous_faults = float(row["previous_faults"])

    # Temperature
    if temperature >= 100:
        modifier += 15.0
    elif temperature >= 85:
        modifier += 8.0
    elif temperature >= 80:
        modifier += 4.0

    # Loading
    if load >= 90:
        modifier += 15.0
    elif load >= 80:
        modifier += 8.0
    elif load >= 75:
        modifier += 4.0

    # THD
    if thd >= 10:
        modifier += 12.0
    elif thd >= 7:
        modifier += 6.0
    elif thd >= 5:
        modifier += 3.0

    # Voltage
    voltage_deviation = abs(voltage - 1.0)

    if voltage_deviation >= 0.05:
        modifier += 12.0
    elif voltage_deviation >= 0.03:
        modifier += 6.0
    elif voltage_deviation >= 0.02:
        modifier += 3.0

    # Frequency
    frequency_deviation = abs(frequency - 50.0)

    if frequency_deviation >= 0.2:
        modifier += 8.0
    elif frequency_deviation >= 0.1:
        modifier += 4.0

    # Power factor
    if power_factor < 0.85:
        modifier += 8.0
    elif power_factor < 0.90:
        modifier += 4.0

    # Historical faults
    if previous_faults >= 5:
        modifier += 8.0
    elif previous_faults >= 2:
        modifier += 3.0

    return min(modifier, 40.0)


def calculate_v22_score(row):
    """
    Experimental V2.2 score.

    ML remains 60%.
    Operational component remains 40%.

    The physical modifier is applied only to
    the operational side.
    """

    probability = float(
        row["failure_probability"]
    )

    operational = float(
        row["operational_score"]
    )

    modifier = calculate_physical_modifier(row)

    adjusted_operational = min(
        operational + modifier,
        100.0,
    )

    ml_component = (
        probability * 100.0
        * PROBABILITY_WEIGHT
    )

    operational_component = (
        adjusted_operational
        * OPERATIONAL_WEIGHT
    )

    score = (
        ml_component
        + operational_component
    )

    # Production failure floor
    if probability >= 0.70:
        score = max(
            score,
            70.0,
        )

    return min(
        max(score, 0.0),
        100.0,
    )


def classify(score):
    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    return "LOW"


# ============================================================
# Calculate V2.2
# ============================================================

print("\nCalculating experimental V2.2 scores...")

df["v22_modifier"] = df.apply(
    calculate_physical_modifier,
    axis=1,
)

df["v22_score"] = df.apply(
    calculate_v22_score,
    axis=1,
)

df["v22_level"] = df["v22_score"].apply(
    classify
)


# ============================================================
# Threshold evaluation
# ============================================================

print("\n" + "=" * 70)
print("[V2.2 RISK LEVEL ANALYSIS]")
print("=" * 70)


level_analysis = []

for level in [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]:

    subset = df[
        df["v22_level"] == level
    ]

    failures = int(
        subset[TARGET].sum()
    )

    records = len(subset)

    rate = (
        failures / records
        if records
        else 0.0
    )

    level_analysis.append(
        {
            "risk_level": level,
            "records": records,
            "failures": failures,
            "failure_rate": rate,
        }
    )

    print(
        f"{level:<10}"
        f" records={records:<6}"
        f" failures={failures:<5}"
        f" failure_rate={rate:.6f}"
    )


# ============================================================
# Escaped failures
# ============================================================

escaped = df[
    (df[TARGET] == 1)
    & (df["v22_level"].isin(
        ["LOW", "MEDIUM"]
    ))
]

print("\n" + "=" * 70)
print("[V2.2 ESCAPED FAILURES]")
print("=" * 70)

print(
    f"Count: {len(escaped)}"
)

if len(escaped):

    print(
        escaped[
            [
                "asset_id",
                "fault_type",
                "failure_probability",
                "risk_score",
                "v22_score",
                "v22_level",
            ]
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# Critical false positives
# ============================================================

critical_fp = df[
    (df[TARGET] == 0)
    & (df["v22_level"] == "CRITICAL")
]

print("\n" + "=" * 70)
print("[V2.2 CRITICAL FALSE POSITIVES]")
print("=" * 70)

print(
    f"Count: {len(critical_fp)}"
)


# ============================================================
# Prediction metrics
# ============================================================

def calculate_metrics(
    threshold,
):

    predicted = (
        df["v22_score"] >= threshold
    )

    actual = (
        df[TARGET] == 1
    )

    tp = int(
        (predicted & actual).sum()
    )

    fp = int(
        (predicted & ~actual).sum()
    )

    fn = int(
        (~predicted & actual).sum()
    )

    tn = int(
        (~predicted & ~actual).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


print("\n" + "=" * 70)
print("[V2.2 THRESHOLD ANALYSIS]")
print("=" * 70)

thresholds = [
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
]

threshold_results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        threshold
    )

    threshold_results.append(
        metrics
    )

    print(
        f"{threshold:>3} "
        f"TP={metrics['tp']:<5} "
        f"FP={metrics['fp']:<5} "
        f"FN={metrics['fn']:<4} "
        f"TN={metrics['tn']:<5} "
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


# ============================================================
# Select safest recommended threshold
# ============================================================

# Prefer recall >= 0.97,
# then maximize F1.

eligible = [
    item
    for item in threshold_results
    if item["recall"] >= 0.97
]

if eligible:

    recommended = max(
        eligible,
        key=lambda item: item["f1"],
    )

else:

    recommended = max(
        threshold_results,
        key=lambda item: item["recall"],
    )


# ============================================================
# Compare against V2.1
# ============================================================

baseline = calculate_metrics(70)

print("\n" + "=" * 70)
print("[V2.1 vs V2.2]")
print("=" * 70)

print(
    f"V2.1 threshold 70:"
    f" precision={baseline['precision']:.4f},"
    f" recall={baseline['recall']:.4f},"
    f" F1={baseline['f1']:.4f}"
)

print(
    f"V2.2 threshold {recommended['threshold']}:"
    f" precision={recommended['precision']:.4f},"
    f" recall={recommended['recall']:.4f},"
    f" F1={recommended['f1']:.4f}"
)


# ============================================================
# Save analysis
# ============================================================

result = {
    "version": "2.2",
    "status": "EXPERIMENTAL",
    "records": int(len(df)),
    "weights": {
        "ml": PROBABILITY_WEIGHT,
        "operational": OPERATIONAL_WEIGHT,
    },
    "baseline_v21": baseline,
    "recommended_threshold": recommended,
    "risk_levels": level_analysis,
    "escaped_failures": {
        "count": int(len(escaped)),
        "by_fault": (
            escaped["fault_type"]
            .value_counts()
            .to_dict()
        ),
        "by_asset": (
            escaped["asset_id"]
            .value_counts()
            .to_dict()
        ),
    },
    "critical_false_positives": int(
        len(critical_fp)
    ),
    "threshold_analysis": threshold_results,
    "calibration_policy": {
        "ml_probability_unchanged": True,
        "operational_component_adjusted": True,
        "production_failure_floor": 70.0,
    },
}


with open(
    OUTPUT_PATH,
    "w",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("V2.2 RISK CALIBRATION COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUTPUT_PATH}"
)

print("=" * 70)
