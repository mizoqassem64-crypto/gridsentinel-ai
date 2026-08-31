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
    confusion_matrix,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


print("=" * 70)
print("GridSentinel AI - V2 Model Diagnostics")
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

with open(METADATA_PATH) as f:
    metadata = json.load(f)

with open(SCALER_PATH) as f:
    scaler_data = json.load(f)

with open(THRESHOLD_PATH) as f:
    threshold_data = json.load(f)

FEATURES = metadata["input_features"]
THRESHOLD = float(threshold_data["threshold"])

print(f"\nFeatures: {len(FEATURES)}")
print(f"Production threshold: {THRESHOLD:.6f}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

test_df = df.iloc[val_end:].copy()

print("\n" + "=" * 70)
print("[TEST DATA]")
print("=" * 70)

print(f"Samples : {len(test_df):,}")
print(f"Failures: {int(test_df['failure'].sum()):,}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading V2 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.20),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)

if isinstance(checkpoint, dict):

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint
    )

    # Handle models saved inside a wrapper
    if any(k.startswith("network.") for k in state_dict):

        state_dict = {
            k.replace("network.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)

else:
    model = checkpoint

model.eval()


# ============================================================
# SCALING
# ============================================================

mean = np.asarray(
    scaler_data["mean"],
    dtype=np.float32
)

std = np.asarray(
    scaler_data["std"],
    dtype=np.float32
)

std = np.where(std == 0, 1.0, std)

X = (
    test_df[FEATURES]
    .astype(np.float32)
    .values
)

X_scaled = (X - mean) / std

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
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

y_true = (
    test_df["failure"]
    .astype(int)
    .values
)


# ============================================================
# OVERALL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("[OVERALL DIAGNOSTICS]")
print("=" * 70)

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    probabilities
)

brier = brier_score_loss(
    y_true,
    probabilities
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Brier    : {brier:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    predictions
).ravel()

print("\n" + "=" * 70)
print("[CONFUSION MATRIX]")
print("=" * 70)

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("[PROBABILITY DISTRIBUTION]")
print("=" * 70)

probability_stats = pd.Series(
    probabilities
).describe()

print(probability_stats)


# ============================================================
# POSITIVE VS NEGATIVE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("[PROBABILITY BY TRUE CLASS]")
print("=" * 70)

positive_probs = probabilities[y_true == 1]
negative_probs = probabilities[y_true == 0]

print(
    f"Failure mean probability : "
    f"{positive_probs.mean():.6f}"
)

print(
    f"Healthy mean probability : "
    f"{negative_probs.mean():.6f}"
)

print(
    f"Failure median probability : "
    f"{np.median(positive_probs):.6f}"
)

print(
    f"Healthy median probability : "
    f"{np.median(negative_probs):.6f}"
)


# ============================================================
# FALSE POSITIVES
# ============================================================

fp_mask = (
    (y_true == 0) &
    (predictions == 1)
)

fp_df = test_df.loc[fp_mask].copy()

fp_df["failure_probability"] = probabilities[fp_mask]

print("\n" + "=" * 70)
print("[HIGH-CONFIDENCE FALSE POSITIVES]")
print("=" * 70)

print(
    fp_df.sort_values(
        "failure_probability",
        ascending=False
    )[
        [
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
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

fn_mask = (
    (y_true == 1) &
    (predictions == 0)
)

fn_df = test_df.loc[fn_mask].copy()

fn_df["failure_probability"] = probabilities[fn_mask]

print("\n" + "=" * 70)
print("[FALSE NEGATIVES]")
print("=" * 70)

if len(fn_df):

    print(
        fn_df.sort_values(
            "failure_probability",
            ascending=False
        )[
            [
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
        ].to_string(index=False)
    )

else:
    print("No false negatives.")


# ============================================================
# FAULT TYPE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("[FAULT TYPE DIAGNOSTICS]")
print("=" * 70)

fault_rows = []

for fault, group in test_df.assign(
    failure_probability=probabilities,
    prediction=predictions
).groupby("fault_type"):

    actual = group["failure"].astype(int)
    pred = group["prediction"].astype(int)

    fault_rows.append(
        {
            "fault_type": fault,
            "records": len(group),
            "actual_failures": int(actual.sum()),
            "predicted_failures": int(pred.sum()),
            "avg_probability": float(
                group["failure_probability"].mean()
            ),
            "precision": float(
                precision_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
        }
    )

fault_df = pd.DataFrame(fault_rows)

print(
    fault_df.to_string(index=False)
)


# ============================================================
# ASSET DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("[ASSET DIAGNOSTICS]")
print("=" * 70)

asset_rows = []

analysis_df = test_df.copy()

analysis_df["failure_probability"] = probabilities
analysis_df["prediction"] = predictions

for asset, group in analysis_df.groupby("asset_id"):

    actual = group["failure"].astype(int)
    pred = group["prediction"].astype(int)

    asset_rows.append(
        {
            "asset_id": asset,
            "records": len(group),
            "actual_failures": int(actual.sum()),
            "predicted_failures": int(pred.sum()),
            "false_positives": int(
                ((actual == 0) & (pred == 1)).sum()
            ),
            "false_negatives": int(
                ((actual == 1) & (pred == 0)).sum()
            ),
            "avg_probability": float(
                group["failure_probability"].mean()
            ),
            "precision": float(
                precision_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    actual,
                    pred,
                    zero_division=0
                )
            ),
        }
    )

asset_df = pd.DataFrame(asset_rows)

print(
    asset_df.to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE - PERMUTATION
# ============================================================

print("\n" + "=" * 70)
print("[PERMUTATION FEATURE IMPORTANCE]")
print("=" * 70)

rng = np.random.default_rng(42)

baseline_f1 = f1

importance_rows = []

for i, feature in enumerate(FEATURES):

    X_permuted = X_scaled.copy()

    shuffled = X_permuted[:, i].copy()

    rng.shuffle(shuffled)

    X_permuted[:, i] = shuffled

    X_perm_tensor = torch.tensor(
        X_permuted,
        dtype=torch.float32
    )

    with torch.no_grad():

        perm_logits = model(
            X_perm_tensor
        )

        perm_probs = (
            torch.sigmoid(perm_logits)
            .cpu()
            .numpy()
            .reshape(-1)
        )

    perm_pred = (
        perm_probs >= THRESHOLD
    ).astype(int)

    perm_f1 = f1_score(
        y_true,
        perm_pred,
        zero_division=0
    )

    importance_rows.append(
        {
            "feature": feature,
            "baseline_f1": float(baseline_f1),
            "permuted_f1": float(perm_f1),
            "f1_drop": float(
                baseline_f1 - perm_f1
            ),
        }
    )

importance_df = (
    pd.DataFrame(importance_rows)
    .sort_values(
        "f1_drop",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    importance_df.to_string(index=False)
)


# ============================================================
# CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("[CALIBRATION]")
print("=" * 70)

try:

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="uniform"
        )
    )

    calibration_rows = []

    for actual_rate, predicted_rate in zip(
        fraction_positive,
        mean_predicted
    ):

        calibration_rows.append(
            {
                "actual_rate": float(actual_rate),
                "predicted_rate": float(predicted_rate),
                "absolute_error": float(
                    abs(
                        actual_rate -
                        predicted_rate
                    )
                ),
            }
        )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    print(
        calibration_df.to_string(index=False)
    )

except Exception as e:

    calibration_df = pd.DataFrame()

    print(
        f"Calibration analysis skipped: {e}"
    )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print("\n" + "=" * 70)
print("[SAVING DIAGNOSTIC ARTIFACTS]")
print("=" * 70)

importance_path = (
    BASE /
    "models/v2_feature_importance.csv"
)

fault_path = (
    BASE /
    "models/v2_fault_diagnostics.csv"
)

asset_path = (
    BASE /
    "models/v2_asset_diagnostics.csv"
)

fp_path = (
    BASE /
    "models/v2_high_confidence_fp.csv"
)

fn_path = (
    BASE /
    "models/v2_fn_diagnostics.csv"
)

calibration_path = (
    BASE /
    "models/v2_calibration.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

fault_df.to_csv(
    fault_path,
    index=False
)

asset_df.to_csv(
    asset_path,
    index=False
)

fp_df.to_csv(
    fp_path,
    index=False
)

fn_df.to_csv(
    fn_path,
    index=False
)

calibration_df.to_csv(
    calibration_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "model": "FailurePredictorV2",
    "version": "2.0",
    "threshold": THRESHOLD,
    "test_samples": int(len(test_df)),
    "test_failures": int(y_true.sum()),
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
    "top_features": [
        {
            "feature": str(row["feature"]),
            "f1_drop": float(row["f1_drop"])
        }
        for _, row in importance_df.head(10).iterrows()
    ],
}

summary_path = (
    BASE /
    "models/v2_model_diagnostics_summary.json"
)

with open(summary_path, "w") as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print(f"Saved: {importance_path}")
print(f"Saved: {fault_path}")
print(f"Saved: {asset_path}")
print(f"Saved: {fp_path}")
print(f"Saved: {fn_path}")
print(f"Saved: {calibration_path}")
print(f"Saved: {summary_path}")

print("\n" + "=" * 70)
print("V2 MODEL DIAGNOSTICS COMPLETE")
print("=" * 70)
