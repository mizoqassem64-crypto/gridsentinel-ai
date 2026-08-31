from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

DATASET = Path("datasets/grid_operations.csv")
MODEL_DIR = Path("models")

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

TARGET = "failure"

SEED = 42

EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    df = pd.read_csv(DATASET)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Critical: chronological ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def temporal_split(df):
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def make_arrays(df):
    X = df[FEATURES].astype(np.float32).to_numpy()
    y = df[TARGET].astype(np.float32).to_numpy()

    return X, y


# ============================================================
# Scaling
# ============================================================

def fit_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Prevent division by zero
    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def transform(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ============================================================
# Model
# ============================================================

class FailurePredictor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, X, y, threshold=0.5):

    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(DEVICE)

        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= threshold
        ).float().cpu().numpy()

        probabilities = probabilities.cpu().numpy()

    y_int = y.astype(np.int32)
    pred_int = predictions.astype(np.int32)

    tp = int(((pred_int == 1) & (y_int == 1)).sum())
    tn = int(((pred_int == 0) & (y_int == 0)).sum())
    fp = int(((pred_int == 1) & (y_int == 0)).sum())
    fn = int(((pred_int == 0) & (y_int == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    accuracy = (tp + tn) / len(y_int)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "probabilities": probabilities,
    }


# ============================================================
# Training
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val):

    # Handle class imbalance.
    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)

    pos_weight_value = negative / positive

    print("\n[CLASS BALANCE]")
    print(f"Positive failures : {int(positive):,}")
    print(f"Negative healthy  : {int(negative):,}")
    print(f"Positive weight   : {pos_weight_value:.4f}")

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    best_val_f1 = -1.0
    best_state = None

    n = len(X_train)

    print("\n[TRAINING]")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device     : {DEVICE}")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        indices = torch.randperm(n)

        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):

            batch_idx = indices[start:start + BATCH_SIZE]

            xb = X_train_tensor[batch_idx].to(DEVICE)
            yb = y_train_tensor[batch_idx].to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        train_loss = total_loss / n

        val_metrics = evaluate(
            model,
            X_val,
            y_val,
            threshold=0.5,
        )

        val_f1 = val_metrics["f1"]

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f} | "
                f"Val Precision: {val_metrics['precision']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nBest validation F1:", f"{best_val_f1:.4f}")


# ============================================================
# Save artifacts
# ============================================================

def save_artifacts(model, mean, std):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "failure_predictor.pt"
    scaler_path = MODEL_DIR / "failure_scaler.json"
    metadata_path = MODEL_DIR / "failure_model_metadata.json"

    torch.save(
        model.state_dict(),
        model_path,
    )

    scaler_data = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": FEATURES,
    }

    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=2)

    metadata = {
        "model": "FailurePredictor",
        "input_features": FEATURES,
        "target": TARGET,
        "architecture": [
            "Linear(13,64)",
            "ReLU",
            "Linear(64,32)",
            "ReLU",
            "Dropout(0.20)",
            "Linear(32,16)",
            "ReLU",
            "Linear(16,1)",
        ],
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "pos_weight": True,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[ARTIFACTS]")
    print("Model  :", model_path)
    print("Scaler :", scaler_path)
    print("Metadata:", metadata_path)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("GridSentinel AI - Failure Prediction Training")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_dataset()

    train, val, test = temporal_split(df)

    X_train, y_train = make_arrays(train)
    X_val, y_val = make_arrays(val)
    X_test, y_test = make_arrays(test)

    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # Fit scaler ONLY on training data.
    mean, std = fit_scaler(X_train)

    X_train = transform(X_train, mean, std)
    X_val = transform(X_val, mean, std)
    X_test = transform(X_test, mean, std)

    print("\n[SCALING]")
    print("Scaler fitted on TRAIN only.")

    model = FailurePredictor(
        input_dim=len(FEATURES)
    ).to(DEVICE)

    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("\n[FINAL TEST EVALUATION]")

    metrics = evaluate(
        model,
        X_test,
        y_test,
        threshold=0.5,
    )

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\n[CONFUSION MATRIX]")
    print(f"TN: {metrics['tn']:,}")
    print(f"FP: {metrics['fp']:,}")
    print(f"FN: {metrics['fn']:,}")
    print(f"TP: {metrics['tp']:,}")

    save_artifacts(
        model,
        mean,
        std,
    )

    print("\n" + "=" * 60)
    print("TRAINING PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
