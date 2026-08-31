from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from train import FailurePredictor


# ============================================================
# Configuration
# ============================================================

DATASET = Path("datasets/grid_operations.csv")
MODEL_PATH = Path("models/failure_predictor.pt")
SCALER_PATH = Path("models/failure_scaler.json")

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

DEVICE = torch.device("cpu")


# ============================================================
# Data
# ============================================================

def load_validation_data():

    df = pd.read_csv(DATASET)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # IMPORTANT:
    # We only use Validation here.
    val = df.iloc[train_end:val_end].copy()

    X = val[FEATURES].astype(np.float32).to_numpy().copy()
    y = val[TARGET].astype(np.int32).to_numpy().copy()

    return X, y


# ============================================================
# Scaler
# ============================================================

def load_scaler():

    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)

    mean = np.asarray(
        scaler["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        scaler["std"],
        dtype=np.float32,
    )

    return mean, std


def scale_data(X, mean, std):

    return ((X - mean) / std).astype(
        np.float32,
        copy=True,
    )


# ============================================================
# Model
# ============================================================

def load_model():

    model = FailurePredictor(
        input_dim=len(FEATURES)
    ).to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)

    model.eval()

    return model


# ============================================================
# Predictions
# ============================================================

def predict_probabilities(model, X):

    X_tensor = torch.from_numpy(
        np.asarray(X, dtype=np.float32).copy()
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    return probabilities


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tp = int(
        ((predictions == 1) & (y_true == 1)).sum()
    )

    tn = int(
        ((predictions == 0) & (y_true == 0)).sum()
    )

    fp = int(
        ((predictions == 1) & (y_true == 0)).sum()
    )

    fn = int(
        ((predictions == 0) & (y_true == 1)).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("GridSentinel AI - Threshold Optimization")
    print("=" * 70)

    print("\nLoading validation dataset...")

    X_val, y_val = load_validation_data()

    print(f"Validation samples: {len(y_val):,}")
    print(f"Validation failures: {int(y_val.sum()):,}")

    mean, std = load_scaler()

    X_val = scale_data(
        X_val,
        mean,
        std,
    )

    print("\nLoading trained model...")

    model = load_model()

    probabilities = predict_probabilities(
        model,
        X_val,
    )

    print("\n[THRESHOLD ANALYSIS]")

    thresholds = np.arange(
        0.10,
        0.96,
        0.05,
    )

    results = []

    print(
        "\n"
        f"{'Threshold':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>7} "
        f"{'FN':>7}"
    )

    print("-" * 62)

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_val,
            probabilities,
            float(threshold),
        )

        results.append(metrics)

        print(
            f"{metrics['threshold']:>10.2f} "
            f"{metrics['precision']:>11.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['fp']:>7} "
            f"{metrics['fn']:>7}"
        )

    # ========================================================
    # Best F1
    # ========================================================

    best_f1 = max(
        results,
        key=lambda x: x["f1"],
    )

    # ========================================================
    # Best threshold with high recall
    # ========================================================

    high_recall = [
        r for r in results
        if r["recall"] >= 0.95
    ]

    if high_recall:

        best_operating = max(
            high_recall,
            key=lambda x: x["precision"],
        )

    else:

        best_operating = best_f1

    print("\n" + "=" * 70)

    print("\n[BEST F1 THRESHOLD]")

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

    print("\n[RECOMMENDED GRID OPERATING POINT]")

    print(
        "Requirement: Recall >= 95%"
    )

    print(
        f"Threshold : {best_operating['threshold']:.2f}"
    )

    print(
        f"Precision : {best_operating['precision']:.4f}"
    )

    print(
        f"Recall    : {best_operating['recall']:.4f}"
    )

    print(
        f"F1        : {best_operating['f1']:.4f}"
    )

    print(
        f"FP        : {best_operating['fp']}"
    )

    print(
        f"FN        : {best_operating['fn']}"
    )

    # ========================================================
    # Save threshold configuration
    # ========================================================

    output = {
        "threshold": best_operating["threshold"],
        "selection_rule": "highest_precision_with_recall_at_least_0.95",
        "validation_metrics": {
            "precision": best_operating["precision"],
            "recall": best_operating["recall"],
            "f1": best_operating["f1"],
            "tp": best_operating["tp"],
            "tn": best_operating["tn"],
            "fp": best_operating["fp"],
            "fn": best_operating["fn"],
        },
    }

    output_path = Path(
        "models/failure_threshold.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n[ARTIFACT]")

    print(
        f"Threshold config: {output_path}"
    )

    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
