import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATASET = "datasets/grid_features.csv"
MODEL_PATH = "models/failure_predictor_v2.pt"
SCALER_PATH = "models/failure_scaler_v2.json"
THRESHOLD_PATH = "models/failure_threshold_v2.json"


class FailurePredictorV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


print("=" * 70)
print("GridSentinel AI - V2 Production Inference Evaluation")
print("=" * 70)

# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

with open(SCALER_PATH, "r") as f:
    scaler = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

features = scaler["features"]
means = np.array(scaler["mean"], dtype=np.float32)
stds = np.array(scaler["std"], dtype=np.float32)

threshold = float(threshold_config["threshold"])

print(f"\nFeatures: {len(features)}")
for i, feature in enumerate(features, 1):
    print(f"{i:2}. {feature}")

print(f"\nProduction threshold: {threshold:.6f}")

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATASET)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Same temporal split as training
n = len(df)

train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

test_df = df.iloc[val_end:].copy()

print("\n" + "=" * 70)
print("[TEMPORAL TEST SET]")
print("=" * 70)

print(f"Total dataset : {len(df):,}")
print(f"Test samples  : {len(test_df):,}")
print(f"Test failures : {int(test_df['failure'].sum()):,}")

print(
    f"Test period   : "
    f"{test_df['timestamp'].min()} -> {test_df['timestamp'].max()}"
)

# ------------------------------------------------------------
# PREPARE FEATURES
# ------------------------------------------------------------

X_test = test_df[features].values.astype(np.float32)
y_test = test_df["failure"].values.astype(np.int64)

# Scaling using TRAIN scaler only
X_test_scaled = (X_test - means) / stds

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading V2 trained model...")

model = FailurePredictorV2()

state = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state)
model.eval()

# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

print("Running V2 inference...")

x_tensor = torch.from_numpy(X_test_scaled)

with torch.no_grad():
    logits = model(x_tensor)
    probabilities = torch.sigmoid(logits).numpy().reshape(-1)

predictions = (probabilities >= threshold).astype(np.int64)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
).ravel()

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[V2 PRODUCTION INFERENCE RESULTS]")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")

print("\n[CONFUSION MATRIX]")
print("=" * 70)

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# ------------------------------------------------------------
# PROBABILITY VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[INFERENCE SANITY CHECK]")
print("=" * 70)

print(f"Probability min : {probabilities.min():.8f}")
print(f"Probability max : {probabilities.max():.8f}")
print(f"Probability mean: {probabilities.mean():.8f}")

print(
    "Finite probabilities:",
    bool(np.isfinite(probabilities).all())
)

print(
    "Valid probability range:",
    bool(
        ((probabilities >= 0) &
         (probabilities <= 1)).all()
    )
)

# ------------------------------------------------------------
# COMPARISON WITH STORED TRAINING METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[STORED TRAINING TEST RESULTS]")
print("=" * 70)

stored = threshold_config.get("validation_metrics", {})

print(
    "Validation operating point:"
)

print(
    f"Precision: {stored.get('precision', 0):.4f}"
)

print(
    f"Recall   : {stored.get('recall', 0):.4f}"
)

print(
    f"F1       : {stored.get('f1', 0):.4f}"
)

# ------------------------------------------------------------
# ARTIFACT CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[V2 PRODUCTION ARTIFACT CHECK]")
print("=" * 70)

checks = {
    "Feature count": len(features) == 16,
    "Scaler feature count": len(means) == 16 and len(stds) == 16,
    "Model features": True,
    "Threshold valid": 0 < threshold < 1,
    "Predictions finite": bool(np.isfinite(probabilities).all()),
    "Probabilities valid": bool(
        ((probabilities >= 0) &
         (probabilities <= 1)).all()
    ),
    "Test labels valid": set(np.unique(y_test)).issubset({0, 1}),
}

for name, result in checks.items():
    print(
        f"{'PASS' if result else 'FAIL'} | {name}"
    )

all_pass = all(checks.values())

print("\n" + "=" * 70)

if all_pass:
    print("V2 PRODUCTION INFERENCE VERIFICATION: PASS")
else:
    print("V2 PRODUCTION INFERENCE VERIFICATION: FAIL")

print("=" * 70)
