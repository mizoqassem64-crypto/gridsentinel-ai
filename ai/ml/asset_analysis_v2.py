import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("=" * 70)
print("GridSentinel AI - V2 Asset-Aware Production Analysis")
print("=" * 70)

BASE = Path(".")

DATASET = BASE / "datasets/grid_features.csv"
MODEL_PATH = BASE / "models/failure_predictor_v2.pt"
SCALER_PATH = BASE / "models/failure_scaler_v2.json"
METADATA_PATH = BASE / "models/failure_model_metadata_v2.json"
THRESHOLD_PATH = BASE / "models/failure_threshold_v2.json"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

with open(SCALER_PATH, "r") as f:
    scaler_data = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)

FEATURES = metadata["input_features"]
THRESHOLD = float(threshold_data["threshold"])

print(f"\nFeatures: {len(FEATURES)}")
print(f"Production threshold: {THRESHOLD:.6f}")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

n = len(df)

val_end = int(n * 0.85)

test_df = (
    df.iloc[val_end:]
    .copy()
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("[TEST DATA]")
print("=" * 70)

print(f"Test samples : {len(test_df):,}")
print(f"Test failures: {int(test_df['failure'].sum()):,}")

print(
    f"Period: {test_df['timestamp'].min()} -> "
    f"{test_df['timestamp'].max()}"
)


# ============================================================
# V2 MODEL
# ============================================================

class FailurePredictorV2(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = torch.nn.Sequential(

            torch.nn.Linear(input_dim, 64),

            torch.nn.ReLU(),

            torch.nn.Linear(64, 32),

            torch.nn.ReLU(),

            torch.nn.Dropout(0.20),

            torch.nn.Linear(32, 16),

            torch.nn.ReLU(),

            torch.nn.Linear(16, 1),
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

from ai.ml.artifact_guard import load_v2_weights

state_dict = load_v2_weights(
    MODEL_PATH,
    input_size=len(FEATURES),
)

model = FailurePredictorV2(
    len(FEATURES)
)

if any(
    str(k).startswith("network.")
    for k in state_dict.keys()
):

    model.load_state_dict(
        state_dict
    )

else:

    normalized = {
        f"network.{k}": v
        for k, v in state_dict.items()
    }

    model.load_state_dict(
        normalized
    )

model.eval()


# ============================================================
# SCALING
# ============================================================

mean = np.asarray(
    scaler_data["mean"],
    dtype=np.float32,
)

std = np.asarray(
    scaler_data["std"],
    dtype=np.float32,
)

scaler_features = scaler_data.get(
    "features",
    [],
)

if len(mean) != len(FEATURES):

    raise ValueError(
        "Scaler mean feature count "
        "does not match V2 metadata."
    )

if len(std) != len(FEATURES):

    raise ValueError(
        "Scaler std feature count "
        "does not match V2 metadata."
    )

if scaler_features:

    if scaler_features != FEATURES:

        raise ValueError(
            "Scaler feature order does not "
            "match V2 metadata."
        )

if np.any(std == 0):

    raise ValueError(
        "Scaler contains zero standard deviation."
    )

X = (
    test_df[FEATURES]
    .astype(np.float32)
    .values
)

X_scaled = (
    X - mean
) / std

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32,
)


# ============================================================
# INFERENCE
# ============================================================

print("Running inference...")

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
        .reshape(-1)
    )

predictions = (
    probabilities >= THRESHOLD
).astype(int)

test_df["failure_probability"] = probabilities

test_df["prediction"] = predictions


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

y_true = (
    test_df["failure"]
    .astype(int)
    .values
)

accuracy = accuracy_score(
    y_true,
    predictions,
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0,
)

print("\n" + "=" * 70)
print("[OVERALL V2 TEST PERFORMANCE]")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# ASSET PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("[ASSET PERFORMANCE]")
print("=" * 70)

asset_rows = []

for asset, group in test_df.groupby(
    "asset_id"
):

    actual = (
        group["failure"]
        .astype(int)
        .values
    )

    pred = (
        group["prediction"]
        .astype(int)
        .values
    )

    tn = int(
        (
            (actual == 0)
            & (pred == 0)
        ).sum()
    )

    fp = int(
        (
            (actual == 0)
            & (pred == 1)
        ).sum()
    )

    fn = int(
        (
            (actual == 1)
            & (pred == 0)
        ).sum()
    )

    tp = int(
        (
            (actual == 1)
            & (pred == 1)
        ).sum()
    )

    acc = accuracy_score(
        actual,
        pred,
    )

    prec = precision_score(
        actual,
        pred,
        zero_division=0,
    )

    rec = recall_score(
        actual,
        pred,
        zero_division=0,
    )

    f1_asset = f1_score(
        actual,
        pred,
        zero_division=0,
    )

    failures = int(
        actual.sum()
    )

    predicted_failures = int(
        pred.sum()
    )

    actual_rate = (
        failures / len(group)
    )

    predicted_rate = (
        predicted_failures
        / len(group)
    )

    print(f"\nAsset: {asset}")
    print(f"Records    : {len(group):,}")
    print(f"Failures   : {failures:,}")
    print(f"TP         : {tp}")
    print(f"TN         : {tn}")
    print(f"FP         : {fp}")
    print(f"FN         : {fn}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1         : {f1_asset:.4f}")

    asset_rows.append({

        "asset_id": asset,

        "records": len(group),

        "actual_failures": failures,

        "predicted_failures":
            predicted_failures,

        "actual_failure_rate":
            actual_rate,

        "predicted_failure_rate":
            predicted_rate,

        "TP": tp,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "accuracy": acc,

        "precision": prec,

        "recall": rec,

        "f1": f1_asset,

        "avg_probability":
            group[
                "failure_probability"
            ].mean(),
    })


asset_summary = pd.DataFrame(
    asset_rows
)


# ============================================================
# FAILURE RATE BY ASSET
# ============================================================

print("\n" + "=" * 70)
print("[FAILURE RATE BY ASSET]")
print("=" * 70)

print(
    asset_summary[
        [
            "asset_id",
            "records",
            "actual_failures",
            "predicted_failures",
            "actual_failure_rate",
            "predicted_failure_rate",
        ]
    ].to_string(index=False)
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

print("\n" + "=" * 70)
print("[FALSE NEGATIVES BY ASSET + FAULT]")
print("=" * 70)

fn_df = test_df[
    (test_df["failure"] == 1)
    & (test_df["prediction"] == 0)
].copy()

if len(fn_df):

    print(
        pd.crosstab(
            fn_df["asset_id"],
            fn_df["fault_type"],
        )
    )

else:

    print("No false negatives.")


# ============================================================
# FALSE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("[FALSE POSITIVES BY ASSET + FAULT]")
print("=" * 70)

fp_df = test_df[
    (test_df["failure"] == 0)
    & (test_df["prediction"] == 1)
].copy()

if len(fp_df):

    print(
        pd.crosstab(
            fp_df["asset_id"],
            fp_df["fault_type"],
        )
    )

else:

    print("No false positives.")


# ============================================================
# FN DETAILED ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[FALSE NEGATIVE DETAILED ANALYSIS]")
print("=" * 70)

if len(fn_df):

    cols = [

        "timestamp",

        "asset_id",

        "fault_type",

        "failure_probability",

        "temperature_c",

        "load_percent",

        "thd_percent",

        "voltage_pu",

        "frequency_hz",

        "power_factor",

        "previous_faults",

        "failure_horizon_hours",
    ]

    print(
        fn_df.sort_values(
            "failure_probability",
            ascending=False,
        )[cols].to_string(
            index=False
        )
    )

else:

    print("No false negatives.")


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("[PROBABILITY DISTRIBUTION BY ASSET]")
print("=" * 70)

prob_stats = (
    test_df
    .groupby("asset_id")
    ["failure_probability"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        median="median",
        max="max",
    )
)

print(prob_stats)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

asset_output = (
    BASE
    / "models/asset_analysis_v2.csv"
)

fn_output = (
    BASE
    / "models/false_negatives_v2.csv"
)

fp_output = (
    BASE
    / "models/false_positives_v2.csv"
)

summary_output = (
    BASE
    / "models/asset_analysis_v2_summary.json"
)

asset_summary.to_csv(
    asset_output,
    index=False,
)

fn_df.to_csv(
    fn_output,
    index=False,
)

fp_df.to_csv(
    fp_output,
    index=False,
)


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {

    "model":
        metadata.get(
            "model",
            "FailurePredictorV2",
        ),

    "version":
        metadata.get(
            "version",
            "2.0",
        ),

    "threshold":
        THRESHOLD,

    "test_samples":
        int(len(test_df)),

    "test_failures":
        int(y_true.sum()),

    "overall_metrics": {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),
    },

    "confusion_matrix": {

        "tn":
            int(
                (
                    (y_true == 0)
                    & (predictions == 0)
                ).sum()
            ),

        "fp":
            int(
                (
                    (y_true == 0)
                    & (predictions == 1)
                ).sum()
            ),

        "fn":
            int(
                (
                    (y_true == 1)
                    & (predictions == 0)
                ).sum()
            ),

        "tp":
            int(
                (
                    (y_true == 1)
                    & (predictions == 1)
                ).sum()
            ),
    },

    "assets":
        asset_rows,
}

with open(
    summary_output,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
default=lambda x: x.item() if isinstance(x, np.generic) else x,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V2 ASSET ANALYSIS COMPLETE")
print("=" * 70)

print(f"Saved: {asset_output}")
print(f"Saved: {fn_output}")
print(f"Saved: {fp_output}")
print(f"Saved: {summary_output}")

print("=" * 70)
