import json
import pandas as pd

from ai.ml.risk_engine import assess_risk


DATASET = "datasets/grid_operations.csv"

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
]


def build_values(row):
    return [float(row[f]) for f in FEATURES]


def build_operational_data(row):
    return {
        "rated_mva": row["rated_mva"],
        "asset_age_years": row["asset_age_years"],
        "criticality": row["criticality"],
        "voltage_pu": row["voltage_pu"],
        "current_a": row["current_a"],
        "frequency_hz": row["frequency_hz"],
        "active_power_mw": row["active_power_mw"],
        "reactive_power_mvar": row["reactive_power_mvar"],
        "power_factor": row["power_factor"],
        "temperature_c": row["temperature_c"],
        "load_percent": row["load_percent"],
        "thd_percent": row["thd_percent"],
        "previous_faults": row["previous_faults"],
        "fault_type": row["fault_type"],
    }


print("=" * 70)
print("GridSentinel AI - Full Risk Engine Evaluation")
print("=" * 70)

df = pd.read_csv(DATASET)

print(f"\nDataset records: {len(df):,}")
print("Running risk assessment...")

results = []

for idx, row in df.iterrows():

    values = build_values(row)
    operational_data = build_operational_data(row)

    result = assess_risk(values, operational_data)

    results.append({
        "actual_failure": int(row["failure"]),
        "asset_id": row["asset_id"],
        "fault_type": row["fault_type"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "failure_probability": result["failure_probability"],
        "prediction": result["prediction"],
    })

    if (idx + 1) % 5000 == 0:
        print(f"Processed: {idx + 1:,}/{len(df):,}")


results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("[RISK LEVEL DISTRIBUTION]")
print("=" * 70)

print(
    results_df["risk_level"]
    .value_counts()
    .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
)


print("\n" + "=" * 70)
print("[RISK LEVEL BY ACTUAL FAILURE]")
print("=" * 70)

risk_by_failure = pd.crosstab(
    results_df["risk_level"],
    results_df["actual_failure"],
)

print(risk_by_failure)


print("\n" + "=" * 70)
print("[AVERAGE RISK SCORE]")
print("=" * 70)

print(
    results_df
    .groupby("actual_failure")["risk_score"]
    .agg(["count", "mean", "median", "min", "max"])
)


print("\n" + "=" * 70)
print("[RISK BY FAULT TYPE]")
print("=" * 70)

fault_summary = (
    results_df
    .groupby("fault_type")
    .agg(
        records=("fault_type", "size"),
        avg_risk=("risk_score", "mean"),
        avg_probability=("failure_probability", "mean"),
        critical=("risk_level", lambda x: (x == "CRITICAL").sum()),
        high=("risk_level", lambda x: (x == "HIGH").sum()),
    )
    .sort_values("avg_risk", ascending=False)
)

print(fault_summary)


print("\n" + "=" * 70)
print("[ACTUAL FAILURES WITH LOW RISK]")
print("=" * 70)

missed = results_df[
    (results_df["actual_failure"] == 1)
    & (results_df["risk_level"].isin(["LOW", "MEDIUM"]))
]

print(f"Count: {len(missed)}")

if len(missed) > 0:
    print(missed.head(20).to_string(index=False))


print("\n" + "=" * 70)
print("[TOP 20 HIGHEST RISK RECORDS]")
print("=" * 70)

print(
    results_df
    .sort_values("risk_score", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("RISK ENGINE EVALUATION COMPLETE")
print("=" * 70)
