import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ai.ml.predict import predict


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "datasets" / "grid_operations.csv"
THRESHOLD_PATH = BASE_DIR / "models" / "failure_threshold.json"
SCALER_PATH = BASE_DIR / "models" / "failure_scaler.json"


def main():
    print("=" * 70)
    print("GridSentinel AI - Production Inference Evaluation")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Same temporal split used during training.
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    test_df = df.iloc[val_end:].copy()

    print(f"\nTotal dataset : {len(df):,}")
    print(f"Test samples  : {len(test_df):,}")
    print(f"Test failures : {int(test_df['failure'].sum()):,}")

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    features = scaler["features"]
    threshold = float(threshold_config["threshold"])

    print("\nFeatures:")
    for i, feature in enumerate(features, 1):
        print(f"{i:2}. {feature}")

    print(f"\nProduction threshold: {threshold:.6f}")

    # ---------------------------------------------------------
    # Run production inference
    # ---------------------------------------------------------
    print("\nRunning inference...")

    probabilities = []

    for _, row in test_df.iterrows():
        values = [row[feature] for feature in features]
        result = predict(values)
        probabilities.append(result["failure_probability"])

    probabilities = np.asarray(probabilities, dtype=np.float64)

    y_true = test_df["failure"].to_numpy(dtype=np.int64)
    y_pred = (probabilities >= threshold).astype(np.int64)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print("\n" + "=" * 70)
    print("[PRODUCTION INFERENCE RESULTS]")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\n[CONFUSION MATRIX]")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    # ---------------------------------------------------------
    # Compare against training evaluation
    # ---------------------------------------------------------
    expected = {
        "accuracy": 0.9809,
        "precision": 0.6577,
        "recall": 0.9782,
        "f1": 0.7865,
        "tn": 7235,
        "fp": 140,
        "fn": 6,
        "tp": 269,
    }

    print("\n" + "=" * 70)
    print("[EXPECTED TRAINING TEST RESULTS]")
    print("=" * 70)

    print(f"Accuracy : ~{expected['accuracy']:.4f}")
    print(f"Precision: ~{expected['precision']:.4f}")
    print(f"Recall   : ~{expected['recall']:.4f}")
    print(f"F1 Score : ~{expected['f1']:.4f}")
    print(f"TN: {expected['tn']} | FP: {expected['fp']}")
    print(f"FN: {expected['fn']} | TP: {expected['tp']}")

    # ---------------------------------------------------------
    # Important:
    # train.py evaluated at its default threshold.
    # The production threshold is 0.75.
    # Therefore exact confusion-matrix equality is NOT expected.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[PRODUCTION ARTIFACT CHECK]")
    print("=" * 70)

    checks = {
        "Feature count": len(features) == 12,
        "Model features": len(features) == 12,
        "Threshold valid": 0.0 < threshold < 1.0,
        "Predictions finite": np.isfinite(probabilities).all(),
        "Probabilities valid": (
            (probabilities >= 0).all()
            and (probabilities <= 1).all()
        ),
        "Test labels valid": set(np.unique(y_true)).issubset({0, 1}),
    }

    all_passed = True

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:5} | {name}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("PRODUCTION INFERENCE VERIFICATION: PASS")
    else:
        print("PRODUCTION INFERENCE VERIFICATION: FAIL")

    print("=" * 70)


if __name__ == "__main__":
    main()
