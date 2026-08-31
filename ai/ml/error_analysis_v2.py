import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
print("GridSentinel AI - V2 Production Error Analysis")
print("=" * 70)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

with open(SCALER_PATH, "r") as f:
    scaler = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

features = scaler["features"]

means = np.array(
    scaler["mean"],
    dtype=np.float32
)

stds = np.array(
    scaler["std"],
    dtype=np.float32
)

threshold = float(
    threshold_config["threshold"]
)

print("\nV2 Features:", len(features))
print("Production threshold:", f"{threshold:.6f}")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# Same temporal split used during training
n = len(df)

train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

test_df = df.iloc[val_end:].copy()

print("\n" + "=" * 70)
print("[TEST DATA]")
print("=" * 70)

print("Test samples :", len(test_df))
print(
    "Test failures:",
    int(test_df["failure"].sum())
)

print(
    "Period:",
    test_df["timestamp"].min(),
    "->",
    test_df["timestamp"].max()
)


# ============================================================
# PREPARE TEST DATA
# ============================================================

X = test_df[
    features
].values.astype(np.float32)

y = test_df[
    "failure"
].values.astype(np.int64)

# TRAIN scaler only
X_scaled = (
    X - means
) / stds


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

model = FailurePredictorV2()

state = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)
model.eval()


# ============================================================
# INFERENCE
# ============================================================

print("Running inference...")

X_tensor = torch.from_numpy(
    X_scaled
)

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = (
        torch.sigmoid(logits)
        .numpy()
        .reshape(-1)
    )


predictions = (
    probabilities >= threshold
).astype(np.int64)


# ============================================================
# BUILD ANALYSIS DATAFRAME
# ============================================================

results = test_df.copy()

results["failure_probability"] = probabilities

results["prediction"] = predictions

results["error_type"] = "TN"

results.loc[
    (results["failure"] == 1) &
    (results["prediction"] == 1),
    "error_type"
] = "TP"

results.loc[
    (results["failure"] == 0) &
    (results["prediction"] == 1),
    "error_type"
] = "FP"

results.loc[
    (results["failure"] == 1) &
    (results["prediction"] == 0),
    "error_type"
] = "FN"


# ============================================================
# CONFUSION GROUPS
# ============================================================

tp = int(
    ((y == 1) & (predictions == 1)).sum()
)

tn = int(
    ((y == 0) & (predictions == 0)).sum()
)

fp = int(
    ((y == 0) & (predictions == 1)).sum()
)

fn = int(
    ((y == 1) & (predictions == 0)).sum()
)


print("\n" + "=" * 70)
print("[CONFUSION GROUPS - V2]")
print("=" * 70)

print("True Positives :", tp)
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

fps = results[
    results["error_type"] == "FP"
].copy()

print("\n" + "=" * 70)
print("[FALSE POSITIVE ANALYSIS]")
print("=" * 70)

print("\nBy asset:")

if len(fps):
    print(
        fps["asset_id"]
        .value_counts()
        .to_string()
    )
else:
    print("None")


print("\nBy fault type:")

if len(fps):
    print(
        fps["fault_type"]
        .value_counts()
        .to_string()
    )
else:
    print("None")


if len(fps):

    print("\nProbability statistics:")

    print(
        fps["failure_probability"]
        .describe()
        .to_string()
    )

    operational_cols = [
        "temperature_c",
        "load_percent",
        "thd_percent",
        "voltage_pu",
        "frequency_hz",
        "power_factor",
        "previous_faults",
    ]

    print("\nOperational statistics:")

    print(
        fps[operational_cols]
        .describe()
        .to_string()
    )


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

fns = results[
    results["error_type"] == "FN"
].copy()

print("\n" + "=" * 70)
print("[FALSE NEGATIVE ANALYSIS]")
print("=" * 70)

print("\nBy asset:")

if len(fns):
    print(
        fns["asset_id"]
        .value_counts()
        .to_string()
    )
else:
    print("None")


print("\nBy fault type:")

if len(fns):
    print(
        fns["fault_type"]
        .value_counts()
        .to_string()
    )
else:
    print("None")


# ============================================================
# ALL FALSE NEGATIVES
# ============================================================

print("\n" + "=" * 70)
print("[FALSE NEGATIVES - ALL]")
print("=" * 70)

if len(fns):

    fn_cols = [
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
        fns[fn_cols]
        .sort_values(
            "failure_probability",
            ascending=False
        )
        .to_string(index=False)
    )

else:
    print("No false negatives.")


# ============================================================
# TOP FALSE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("[TOP FALSE POSITIVES]")
print("=" * 70)

if len(fps):

    fp_cols = [
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

    print(
        fps[fp_cols]
        .sort_values(
            "failure_probability",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

else:
    print("No false positives.")


# ============================================================
# ACTUAL FAILURE PROBABILITY
# ============================================================

actual_failures = results[
    results["failure"] == 1
]

print("\n" + "=" * 70)
print("[ACTUAL FAILURE PROBABILITY]")
print("=" * 70)

print(
    actual_failures[
        "failure_probability"
    ]
    .describe()
    .to_string()
)


# ============================================================
# FAULT TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[FAULT TYPE ANALYSIS]")
print("=" * 70)

fault_analysis = (
    results
    .groupby("fault_type")
    .agg(
        records=("failure", "count"),
        actual_failures=("failure", "sum"),
        avg_probability=(
            "failure_probability",
            "mean"
        ),
        predicted_failures=(
            "prediction",
            "sum"
        ),
    )
)

print(
    fault_analysis
    .to_string()
)


# ============================================================
# ASSET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[ASSET ERROR SUMMARY]")
print("=" * 70)

asset_summary = (
    results
    .groupby("asset_id")
    .agg(
        records=("failure", "count"),
        actual_failures=("failure", "sum"),
        predicted_failures=("prediction", "sum"),
        false_positives=(
            "error_type",
            lambda x: (x == "FP").sum()
        ),
        false_negatives=(
            "error_type",
            lambda x: (x == "FN").sum()
        ),
        avg_probability=(
            "failure_probability",
            "mean"
        ),
    )
)

print(
    asset_summary
    .to_string()
)


# ============================================================
# TOP RISK RECORDS
# ============================================================

print("\n" + "=" * 70)
print("[TOP 20 HIGHEST PROBABILITY RECORDS]")
print("=" * 70)

top_cols = [
    "timestamp",
    "asset_id",
    "fault_type",
    "failure",
    "prediction",
    "failure_probability",
    "temperature_c",
    "load_percent",
    "thd_percent",
]

print(
    results[
        top_cols
    ]
    .sort_values(
        "failure_probability",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# LOW CONFIDENCE FAILURES
# ============================================================

print("\n" + "=" * 70)
print("[ACTUAL FAILURES WITH LOW PROBABILITY]")
print("=" * 70)

low_failures = results[
    (results["failure"] == 1) &
    (results["failure_probability"] < threshold)
].copy()

print(
    "Count:",
    len(low_failures)
)

if len(low_failures):

    print(
        low_failures[
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
        ]
        .sort_values(
            "failure_probability"
        )
        .to_string(index=False)
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V2 ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"TP={tp} | TN={tn} | FP={fp} | FN={fn}"
)

print(
    f"False Positive Rate: "
    f"{fp / (fp + tn):.4%}"
)

print(
    f"False Negative Rate: "
    f"{fn / (fn + tp):.4%}"
)

print("=" * 70)
