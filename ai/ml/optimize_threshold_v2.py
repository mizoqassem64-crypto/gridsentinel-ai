import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# GridSentinel AI - V2 Threshold Optimization
# ============================================================

DATA_PATH = "datasets/grid_features.csv"

MODEL_PATH = "models/failure_predictor_v2.pt"
SCALER_PATH = "models/failure_scaler_v2.json"
METADATA_PATH = "models/failure_model_metadata_v2.json"

THRESHOLD_PATH = "models/failure_threshold_v2.json"

TARGET = "failure"

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class FailurePredictorV2(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

print("=" * 70)
print("GridSentinel AI - V2 Threshold Optimization")
print("=" * 70)


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

print("\nLoading V2 metadata...")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

features = metadata["input_features"]

print("Features:", len(features))

for i, feature in enumerate(features, 1):
    print(f"{i:2}. {feature}")


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading validation dataset...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

val_df = df.iloc[train_end:val_end].copy()

print(f"Validation samples: {len(val_df):,}")
print(f"Validation failures: {int(val_df[TARGET].sum()):,}")


# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------

print("\nLoading V2 scaler...")

with open(SCALER_PATH, "r") as f:
    scaler_data = json.load(f)

scaler_mean = np.array(
    scaler_data["mean"],
    dtype=np.float32
)

scaler_std = np.array(
    scaler_data["std"],
    dtype=np.float32
)

if scaler_data["features"] != features:
    raise ValueError(
        "Scaler features do not match metadata features."
    )


# ------------------------------------------------------------
# Prepare validation data
# ------------------------------------------------------------

X_val = val_df[features].to_numpy(
    dtype=np.float32
)

y_val = val_df[TARGET].to_numpy(
    dtype=np.int64
)

X_val = (
    (X_val - scaler_mean)
    / scaler_std
)

X_val = torch.from_numpy(
    X_val.astype(np.float32)
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading V2 trained model...")

model = FailurePredictorV2(
    input_size=len(features)
)

state = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

print("\nRunning validation inference...")

with torch.no_grad():

    logits = model(X_val).squeeze(1)

    probabilities = torch.sigmoid(
        logits
    ).numpy()


if not np.all(np.isfinite(probabilities)):
    raise ValueError(
        "Model produced non-finite probabilities."
    )


# ------------------------------------------------------------
# Threshold Analysis
# ------------------------------------------------------------

print("\n[THRESHOLD ANALYSIS]")

print(
    "\n"
    f"{'Threshold':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'FP':>7} "
    f"{'FN':>7}"
)

print("-" * 62)


results = []

thresholds = np.arange(
    0.10,
    1.00,
    0.05
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    fp = int(
        ((predictions == 1) & (y_val == 0)).sum()
    )

    fn = int(
        ((predictions == 0) & (y_val == 1)).sum()
    )

    results.append({
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp": fp,
        "fn": fn
    })

    print(
        f"{threshold:10.2f} "
        f"{precision:12.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f} "
        f"{fp:7d} "
        f"{fn:7d}"
    )


# ------------------------------------------------------------
# Best F1
# ------------------------------------------------------------

best_f1 = max(
    results,
    key=lambda x: x["f1"]
)


print("\n" + "=" * 70)
print("[BEST F1 THRESHOLD]")
print("=" * 70)

print(
    f"Threshold : {best_f1['threshold']:.2f}"
)

print(
    f"Precision : {best_f1['precision']:.4f}"
)

print(
    f"Recall    : {best_f1['recall']:.4f}"
)

print(
    f"F1        : {best_f1['f1']:.4f}"
)

print(
    f"FP        : {best_f1['fp']}"
)

print(
    f"FN        : {best_f1['fn']}"
)


# ------------------------------------------------------------
# Recommended Grid Operating Point
# ------------------------------------------------------------

MIN_RECALL = 0.95

eligible = [
    r for r in results
    if r["recall"] >= MIN_RECALL
]

if not eligible:

    raise RuntimeError(
        "No threshold satisfies Recall >= 95%."
    )


recommended = max(
    eligible,
    key=lambda x: (
        x["precision"],
        x["f1"]
    )
)


print("\n" + "=" * 70)
print("[RECOMMENDED GRID OPERATING POINT]")
print("=" * 70)

print(
    "Requirement: Recall >= 95%"
)

print(
    f"Threshold : {recommended['threshold']:.2f}"
)

print(
    f"Precision : {recommended['precision']:.4f}"
)

print(
    f"Recall    : {recommended['recall']:.4f}"
)

print(
    f"F1        : {recommended['f1']:.4f}"
)

print(
    f"FP        : {recommended['fp']}"
)

print(
    f"FN        : {recommended['fn']}"
)


# ------------------------------------------------------------
# Save Threshold
# ------------------------------------------------------------

threshold_config = {

    "threshold": recommended["threshold"],

    "selection_rule":
        "highest_precision_with_recall_at_least_0.95",

    "model":
        "FailurePredictorV2",

    "validation_metrics": {

        "precision":
            recommended["precision"],

        "recall":
            recommended["recall"],

        "f1":
            recommended["f1"],

        "tp":
            int(
                (
                    (probabilities >= recommended["threshold"])
                    & (y_val == 1)
                ).sum()
            ),

        "tn":
            int(
                (
                    (probabilities < recommended["threshold"])
                    & (y_val == 0)
                ).sum()
            ),

        "fp":
            recommended["fp"],

        "fn":
            recommended["fn"]
    }
}


Path("models").mkdir(
    parents=True,
    exist_ok=True
)


with open(
    THRESHOLD_PATH,
    "w"
) as f:

    json.dump(
        threshold_config,
        f,
        indent=2
    )


# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n[ARTIFACT]")

print(
    f"Threshold config: {THRESHOLD_PATH}"
)

print("\n" + "=" * 70)
print("V2 THRESHOLD OPTIMIZATION PASS")
print("=" * 70)
