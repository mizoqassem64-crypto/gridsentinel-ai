import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# GridSentinel AI - V2 Failure Prediction Training
# ============================================================

SEED = 42
EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

DATA_PATH = "datasets/grid_features.csv"

MODEL_PATH = "models/failure_predictor_v2.pt"
SCALER_PATH = "models/failure_scaler_v2.json"
METADATA_PATH = "models/failure_model_metadata_v2.json"

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# V2 Feature Set
# ------------------------------------------------------------
#
# Selected to balance:
# - predictive strength
# - physical meaning
# - low redundancy
# - operational interpretability
#
# We intentionally avoid:
# - apparent_power_mva (highly redundant with active_power_mw)
# - power_utilization (highly redundant with load_percent)
# - harmonic_stress (redundant with thd_percent)
# - power_factor_deviation (same information as power_factor)
# - combined_stress (highly correlated with thermal_stress)
# - overload_stress (zero variance in this dataset)
#

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

TARGET = "failure"


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

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

print("=" * 70)
print("GridSentinel AI - V2 Failure Prediction Training")
print("=" * 70)

print("\nLoading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset:", df.shape)

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("\n[V2 FEATURES]")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i:2}. {feature}")


# ------------------------------------------------------------
# Temporal Split
# ------------------------------------------------------------

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


def print_split(name, data):
    failures = int(data[TARGET].sum())
    rate = failures / len(data)

    print(
        f"{name:<10} {len(data):>6} records | "
        f"Failures: {failures:>4} | "
        f"Rate: {rate:.2%}"
    )

    print(
        f"           {data['timestamp'].min()} -> "
        f"{data['timestamp'].max()}"
    )


print("\n[TEMPORAL SPLIT]")

print_split("Train", train_df)
print_split("Validation", val_df)
print_split("Test", test_df)


# ------------------------------------------------------------
# Prepare Arrays
# ------------------------------------------------------------

X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
X_val = val_df[FEATURES].to_numpy(dtype=np.float32)
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

y_train = train_df[TARGET].to_numpy(dtype=np.float32)
y_val = val_df[TARGET].to_numpy(dtype=np.float32)
y_test = test_df[TARGET].to_numpy(dtype=np.float32)


# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

print("\n[SCALING]")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("Scaler fitted on TRAIN only.")


# ------------------------------------------------------------
# Class Balance
# ------------------------------------------------------------

positive = int(y_train.sum())
negative = len(y_train) - positive

pos_weight_value = negative / positive

print("\n[CLASS BALANCE]")
print(f"Positive failures : {positive:,}")
print(f"Negative healthy  : {negative:,}")
print(f"Positive weight   : {pos_weight_value:.4f}")


# ------------------------------------------------------------
# Tensor Preparation
# ------------------------------------------------------------

X_train_tensor = torch.from_numpy(X_train.copy())
y_train_tensor = torch.from_numpy(y_train.copy())

X_val_tensor = torch.from_numpy(X_val.copy())
y_val_tensor = torch.from_numpy(y_val.copy())

X_test_tensor = torch.from_numpy(X_test.copy())
y_test_tensor = torch.from_numpy(y_test.copy())


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = FailurePredictorV2(
    input_size=len(FEATURES)
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight_value],
        dtype=torch.float32
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ------------------------------------------------------------
# Validation Function
# ------------------------------------------------------------

def evaluate(model, X, y, threshold=0.5):

    model.eval()

    with torch.no_grad():

        logits = model(
            X.to(DEVICE)
        ).squeeze(1)

        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    return accuracy, precision, recall, f1


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\n[TRAINING]")

print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Device     : {DEVICE}")

best_f1 = -1
best_state = None
best_epoch = 0


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * len(batch_x)
        )

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    _, precision, recall, f1 = evaluate(
        model,
        X_val_tensor,
        y_val,
        threshold=0.5
    )

    if f1 > best_f1:

        best_f1 = f1
        best_epoch = epoch

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Val F1: {f1:.4f} | "
            f"Val Recall: {recall:.4f} | "
            f"Val Precision: {precision:.4f}"
        )


# ------------------------------------------------------------
# Restore Best Model
# ------------------------------------------------------------

model.load_state_dict(best_state)

print(f"\nBest validation F1: {best_f1:.4f}")
print(f"Best epoch        : {best_epoch}")


# ------------------------------------------------------------
# Final Test Evaluation
# ------------------------------------------------------------

accuracy, precision, recall, f1 = evaluate(
    model,
    X_test_tensor,
    y_test,
    threshold=0.5
)

with torch.no_grad():

    logits = model(
        X_test_tensor.to(DEVICE)
    ).squeeze(1)

    probabilities = torch.sigmoid(
        logits
    ).cpu().numpy()

predictions = (
    probabilities >= 0.5
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()


print("\n[FINAL TEST EVALUATION]")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n[CONFUSION MATRIX]")

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ------------------------------------------------------------
# Save Scaler
# ------------------------------------------------------------

scaler_data = {
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist(),
    "features": FEATURES
}

with open(
    SCALER_PATH,
    "w"
) as f:

    json.dump(
        scaler_data,
        f,
        indent=2
    )


# ------------------------------------------------------------
# Save Metadata
# ------------------------------------------------------------

metadata = {
    "model": "FailurePredictorV2",
    "version": "2.0",
    "input_features": FEATURES,
    "input_feature_count": len(FEATURES),
    "target": TARGET,
    "architecture": [
        f"Linear({len(FEATURES)},64)",
        "ReLU",
        "Linear(64,32)",
        "ReLU",
        "Dropout(0.20)",
        "Linear(32,16)",
        "ReLU",
        "Linear(16,1)"
    ],
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "pos_weight": True,
    "best_epoch": best_epoch,
    "best_validation_f1": best_f1,
    "test_metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
}

with open(
    METADATA_PATH,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n[ARTIFACTS]")

print(f"Model    : {MODEL_PATH}")
print(f"Scaler   : {SCALER_PATH}")
print(f"Metadata : {METADATA_PATH}")

print("\n" + "=" * 70)
print("V2 TRAINING PASS")
print("=" * 70)
