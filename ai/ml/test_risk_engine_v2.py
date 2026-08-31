import json
from pathlib import Path

import pandas as pd

from ai.ml.risk_engine_v2 import assess_risk_v2


print("=" * 70)
print("GridSentinel AI - V2 Risk Intelligence Evaluation")
print("=" * 70)

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "datasets" / "grid_features.csv"
OUTPUT = ROOT / "models" / "v2_risk_engine_evaluation.csv"
SUMMARY = ROOT / "models" / "v2_risk_engine_evaluation_summary.json"


FEATURES = [
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


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print(f"Dataset records: {len(df):,}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# Validation
# ============================================================

required_columns = FEATURES + [
    "failure",
    "asset_id",
    "fault_type",
    "previous_faults",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Build V2 input
# ============================================================

def build_input(row):
    data = {}

    for feature in FEATURES:
        data[feature] = float(row[feature])

    data["previous_faults"] = float(
        row["previous_faults"]
    )

    data["fault_type"] = str(
        row["fault_type"]
    )

    return data


# ============================================================
# Run evaluation
# ============================================================

print("\nRunning V2 risk assessments...")

results = []

for idx, row in df.iterrows():

    data = build_input(row)

    result = assess_risk_v2(data)

    results.append({
        "timestamp": (
            row["timestamp"]
            if "timestamp" in df.columns
            else None
        ),
        "asset_id": str(row["asset_id"]),
        "fault_type": str(row["fault_type"]),
        "actual_failure": int(row["failure"]),
        "failure_probability": float(
            result["failure_probability"]
        ),
        "prediction": str(
            result["prediction"]
        ),
        "risk_score": float(
            result["risk_score"]
        ),
        "risk_level": str(
            result["risk_level"]
        ),
        "ml_score": float(
            result["ml_score"]
        ),
        "operational_score": float(
            result["operational_score"]
        ),
    })

    if (idx + 1) % 5000 == 0:
        print(
            f"Processed: {idx + 1:,}/{len(df):,}"
        )


results_df = pd.DataFrame(results)


# ============================================================
# Risk distribution
# ============================================================

print("\n" + "=" * 70)
print("[RISK LEVEL DISTRIBUTION]")
print("=" * 70)

risk_distribution = (
    results_df["risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        fill_value=0,
    )
)

print(risk_distribution)


# ============================================================
# Risk by actual failure
# ============================================================

print("\n" + "=" * 70)
print("[RISK LEVEL BY ACTUAL FAILURE]")
print("=" * 70)

risk_by_failure = pd.crosstab(
    results_df["risk_level"],
    results_df["actual_failure"],
)

print(risk_by_failure)


# ============================================================
# Average risk score
# ============================================================

print("\n" + "=" * 70)
print("[AVERAGE RISK SCORE]")
print("=" * 70)

average_risk = (
    results_df
    .groupby("actual_failure")["risk_score"]
    .agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
    ])
)

print(average_risk)


# ============================================================
# Risk by fault type
# ============================================================

print("\n" + "=" * 70)
print("[RISK BY FAULT TYPE]")
print("=" * 70)

fault_summary = (
    results_df
    .groupby("fault_type")
    .agg(
        records=("fault_type", "size"),
        actual_failures=("actual_failure", "sum"),
        avg_risk=("risk_score", "mean"),
        avg_probability=(
            "failure_probability",
            "mean",
        ),
        critical=(
            "risk_level",
            lambda x: (x == "CRITICAL").sum(),
        ),
        high=(
            "risk_level",
            lambda x: (x == "HIGH").sum(),
        ),
        medium=(
            "risk_level",
            lambda x: (x == "MEDIUM").sum(),
        ),
        low=(
            "risk_level",
            lambda x: (x == "LOW").sum(),
        ),
    )
    .sort_values(
        "avg_risk",
        ascending=False,
    )
)

print(fault_summary)


# ============================================================
# Risk by asset
# ============================================================

print("\n" + "=" * 70)
print("[RISK BY ASSET]")
print("=" * 70)

asset_summary = (
    results_df
    .groupby("asset_id")
    .agg(
        records=("asset_id", "size"),
        actual_failures=("actual_failure", "sum"),
        avg_risk=("risk_score", "mean"),
        avg_probability=(
            "failure_probability",
            "mean",
        ),
        critical=(
            "risk_level",
            lambda x: (x == "CRITICAL").sum(),
        ),
        high=(
            "risk_level",
            lambda x: (x == "HIGH").sum(),
        ),
        medium=(
            "risk_level",
            lambda x: (x == "MEDIUM").sum(),
        ),
        low=(
            "risk_level",
            lambda x: (x == "LOW").sum(),
        ),
    )
    .sort_values(
        "avg_risk",
        ascending=False,
    )
)

print(asset_summary)


# ============================================================
# Actual failures with low/medium risk
# ============================================================

print("\n" + "=" * 70)
print("[ACTUAL FAILURES WITH LOW/MEDIUM RISK]")
print("=" * 70)

missed = results_df[
    (results_df["actual_failure"] == 1)
    & (
        results_df["risk_level"].isin(
            ["LOW", "MEDIUM"]
        )
    )
]

print(f"Count: {len(missed):,}")

if len(missed):
    print(
        missed
        .sort_values(
            "risk_score",
            ascending=True,
        )
        .head(20)
        .to_string(index=False)
    )
else:
    print("No actual failures were classified as LOW/MEDIUM risk.")


# ============================================================
# Top high-risk records
# ============================================================

print("\n" + "=" * 70)
print("[TOP 20 HIGHEST RISK RECORDS]")
print("=" * 70)

print(
    results_df
    .sort_values(
        "risk_score",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# Failure detection by risk level
# ============================================================

print("\n" + "=" * 70)
print("[FAILURE RATE BY RISK LEVEL]")
print("=" * 70)

failure_rate_by_risk = (
    results_df
    .groupby("risk_level")
    .agg(
        records=("risk_level", "size"),
        failures=("actual_failure", "sum"),
        avg_probability=(
            "failure_probability",
            "mean",
        ),
        avg_risk=("risk_score", "mean"),
    )
)

failure_rate_by_risk["failure_rate"] = (
    failure_rate_by_risk["failures"]
    / failure_rate_by_risk["records"]
)

failure_rate_by_risk = (
    failure_rate_by_risk
    .reindex(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
)

print(failure_rate_by_risk)


# ============================================================
# Save detailed results
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Save JSON summary
# ============================================================

summary = {
    "model": "FailurePredictorV2",
    "version": "2.0",
    "dataset_records": int(len(df)),
    "actual_failures": int(
        results_df["actual_failure"].sum()
    ),
    "risk_distribution": {
        str(key): int(value)
        for key, value
        in risk_distribution.items()
    },
    "missed_low_medium_risk": int(
        len(missed)
    ),
    "average_risk_by_actual_failure": {
        str(index): {
            str(key): float(value)
            for key, value
            in row.items()
        }
        for index, row
        in average_risk.iterrows()
    },
}

with open(
    SUMMARY,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 70)
print("V2 RISK ENGINE EVALUATION COMPLETE")
print("=" * 70)

print(f"Saved: {OUTPUT}")
print(f"Saved: {SUMMARY}")
print("=" * 70)
