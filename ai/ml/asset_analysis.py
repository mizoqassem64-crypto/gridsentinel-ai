import json
import numpy as np
import pandas as pd
from ai.ml.predict import predict

DATASET = "datasets/grid_operations.csv"
SCALER = "models/failure_scaler.json"

print("=" * 70)
print("GridSentinel AI - Asset-Aware Error Analysis")
print("=" * 70)

df = pd.read_csv(DATASET)

with open(SCALER) as f:
    scaler = json.load(f)

features = scaler["features"]

results = []

print("\nRunning inference...")

for idx, row in df.iterrows():
    values = [row[f] for f in features]

    result = predict(values)

    results.append({
        "asset_id": row["asset_id"],
        "fault_type": row["fault_type"],
        "actual_failure": int(row["failure"]),
        "probability": float(result["failure_probability"]),
        "prediction": int(result["prediction"]),
    })

    if (idx + 1) % 5000 == 0:
        print(f"Processed: {idx + 1:,}/{len(df):,}")

res = pd.DataFrame(results)

print("\n" + "=" * 70)
print("[ASSET PERFORMANCE]")
print("=" * 70)

for asset in sorted(res["asset_id"].unique()):

    subset = res[res["asset_id"] == asset]

    tp = ((subset.actual_failure == 1) &
          (subset.prediction == 1)).sum()

    tn = ((subset.actual_failure == 0) &
          (subset.prediction == 0)).sum()

    fp = ((subset.actual_failure == 0) &
          (subset.prediction == 1)).sum()

    fn = ((subset.actual_failure == 1) &
          (subset.prediction == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (tp + tn) / len(subset)

    print(f"\nAsset: {asset}")
    print(f"Records    : {len(subset):,}")
    print(f"Failures   : {subset.actual_failure.sum():,}")
    print(f"TP         : {tp}")
    print(f"TN         : {tn}")
    print(f"FP         : {fp}")
    print(f"FN         : {fn}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1         : {f1:.4f}")

print("\n" + "=" * 70)
print("[FAILURE RATE BY ASSET]")
print("=" * 70)

failure_rate = (
    df.groupby("asset_id")["failure"]
    .agg(["count", "sum", "mean"])
)

failure_rate.columns = [
    "records",
    "failures",
    "failure_rate"
]

print(failure_rate)

print("\n" + "=" * 70)
print("[PREDICTED FAILURE RATE BY ASSET]")
print("=" * 70)

print(
    res.groupby("asset_id")["prediction"]
    .mean()
    .rename("predicted_failure_rate")
)

print("\n" + "=" * 70)
print("[PROBABILITY DISTRIBUTION BY ASSET]")
print("=" * 70)

print(
    res.groupby("asset_id")["probability"]
    .describe()[[
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ]]
)

print("\n" + "=" * 70)
print("[FALSE POSITIVES BY ASSET + FAULT]")
print("=" * 70)

fp = res[
    (res.actual_failure == 0) &
    (res.prediction == 1)
]

print(
    pd.crosstab(
        fp["asset_id"],
        fp["fault_type"]
    )
)

print("\n" + "=" * 70)
print("[FALSE NEGATIVES BY ASSET + FAULT]")
print("=" * 70)

fn = res[
    (res.actual_failure == 1) &
    (res.prediction == 0)
]

print(
    pd.crosstab(
        fn["asset_id"],
        fn["fault_type"]
    )
)

print("\n" + "=" * 70)
print("[FAILURE PROBABILITY BY ASSET]")
print("=" * 70)

actual_failures = res[res.actual_failure == 1]

print(
    actual_failures
    .groupby("asset_id")["probability"]
    .describe()
)

print("\n" + "=" * 70)
print("ASSET-AWARE ERROR ANALYSIS COMPLETE")
print("=" * 70)
