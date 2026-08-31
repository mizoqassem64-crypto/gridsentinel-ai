import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "failure_predictor.pt"
SCALER_PATH = MODELS_DIR / "failure_scaler.json"
THRESHOLD_PATH = MODELS_DIR / "failure_threshold.json"


class FailurePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(12, 64),
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


def load_artifacts():
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    model = FailurePredictor()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, threshold_config


def predict(features):
    model, scaler, threshold_config = load_artifacts()

    feature_names = scaler["features"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)

    if len(features) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, got {len(features)}"
        )

    x = np.asarray(features, dtype=np.float32)

    # Prevent division by zero in case a scaler contains zero variance.
    std = np.where(std == 0, 1.0, std)

    # Same standardization used during training.
    x_scaled = (x - mean) / std

    tensor = torch.from_numpy(
        x_scaled.astype(np.float32, copy=True)
    ).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    threshold = float(threshold_config["threshold"])

    prediction = int(probability >= threshold)

    if probability >= threshold:
        risk_level = "HIGH"
    elif probability >= threshold * 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "failure_probability": round(probability, 6),
        "threshold": threshold,
        "prediction": prediction,
        "status": "FAILURE" if prediction else "NORMAL",
        "risk_level": risk_level,
        "features": {
            name: float(value)
            for name, value in zip(feature_names, features)
        },
    }


if __name__ == "__main__":
    # Example representative grid-operating sample.
    sample = [
        50.0,    # rated_mva
        10.0,    # asset_age_years
        0.95,    # criticality
        0.995,   # voltage_pu
        570.0,   # current_a
        50.0,    # frequency_hz
        30.0,    # active_power_mw
        10.0,    # reactive_power_mvar
        0.95,    # power_factor
        67.0,    # temperature_c
        69.0,    # load_percent
        3.2,     # thd_percent
    ]

    result = predict(sample)

    print("=" * 60)
    print("GridSentinel AI - Failure Prediction")
    print("=" * 60)

    print(f"Probability : {result['failure_probability']}")
    print(f"Threshold   : {result['threshold']}")
    print(f"Prediction  : {result['prediction']}")
    print(f"Status      : {result['status']}")
    print(f"Risk Level  : {result['risk_level']}")

    print("=" * 60)
