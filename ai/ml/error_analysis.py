import pandas as pd

from ai.ml.predict import predict


DATASET = "datasets/grid_operations.csv"

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


df = pd.read_csv(DATASET)

# Same temporal test split used during training
test = df.iloc[-7650:].copy()

probabilities = []

print("=" * 70)
print("GridSentinel AI - Production Error Analysis")
print("=" * 70)

print(f"\nTest samples: {len(test):,}")
print("Running inference...")

for i, (_, row) in enumerate(test.iterrows()):

    values = [float(row[f]) for f in FEATURES]

    result = predict(values)

    probabilities.append(
        float(result["failure_probability"])
    )

    if (i + 1) % 1000 == 0:
        print(f"Processed: {i + 1:,}/{len(test):,}")


test["failure_probability"] = probabilities
test["prediction"] = (test["failure_probability"] >= 0.75).astype(int)

# ------------------------------------------------------------
# Confusion groups
# ------------------------------------------------------------

tp = test[
    (test["failure"] == 1) &
    (test["prediction"] == 1)
]

tn = test[
    (test["failure"] == 0) &
    (test["prediction"] == 0)
]

fp = test[
    (test["failure"] == 0) &
    (test["prediction"] == 1)
]

fn = test[
    (test["failure"] == 1) &
    (test["prediction"] == 0)
]


print("\n" + "=" * 70)
print("[CONFUSION GROUPS]")
print("=" * 70)

print("True Positives :", len(tp))
print("True Negatives :", len(tn))
print("False Positives:", len(fp))
print("False Negatives:", len(fn))


# ------------------------------------------------------------
# False Positive Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[FALSE POSITIVE ANALYSIS]")
print("=" * 70)

if len(fp):

    print("\nBy fault type:")
    print(
        fp["fault_type"]
        .value_counts()
        .to_string()
    )

    print("\nBy asset:")
    print(
        fp["asset_id"]
        .value_counts()
        .to_string()
    )

    print("\nProbability statistics:")
    print(
        fp["failure_probability"]
        .describe()
        .to_string()
    )

    print("\nOperational statistics:")
    print(
        fp[
            [
                "temperature_c",
                "load_percent",
                "thd_percent",
                "voltage_pu",
                "frequency_hz",
                "power_factor",
                "previous_faults",
            ]
        ]
        .describe()
        .to_string()
    )


# ------------------------------------------------------------
# False Negative Analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[FALSE NEGATIVE ANALYSIS]")
print("=" * 70)

if len(fn):

    print("\nFalse negative records:")
    print(
        fn[
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
        .to_string(index=False)
    )

else:
    print("No false negatives.")


# ------------------------------------------------------------
# Fault Type Performance
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[FAULT TYPE ANALYSIS]")
print("=" * 70)

fault_summary = (
    test
    .groupby("fault_type")
    .agg(
        records=("fault_type", "size"),
        actual_failures=("failure", "sum"),
        avg_probability=("failure_probability", "mean"),
        predicted_failures=("prediction", "sum"),
    )
    .sort_values("actual_failures", ascending=False)
)

print(fault_summary.to_string())


# ------------------------------------------------------------
# Probability distribution for actual failures
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[ACTUAL FAILURE PROBABILITY]")
print("=" * 70)

print(
    test[test["failure"] == 1]["failure_probability"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Highest-confidence false positives
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[TOP FALSE POSITIVES]")
print("=" * 70)

if len(fp):

    print(
        fp.sort_values(
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
        ]
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Highest-confidence false negatives
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[FALSE NEGATIVES - HIGHEST PROBABILITY]")
print("=" * 70)

if len(fn):

    print(
        fn.sort_values(
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
        ]
        .to_string(index=False)
    )


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
