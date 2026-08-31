from pathlib import Path
import pandas as pd
import numpy as np


DATASET = Path("datasets/grid_operations.csv")

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
    "previous_faults",
]

TARGET = "failure"


def load_dataset():
    df = pd.read_csv(DATASET)

    # Timestamp is required for chronological splitting
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort strictly by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_data(df):
    X = df[FEATURES].astype(np.float32).to_numpy()
    y = df[TARGET].astype(np.float32).to_numpy()

    return X, y


def temporal_split(df):
    """
    Chronological split:

    70% Train
    15% Validation
    15% Test

    No random shuffling.
    """

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def main():
    print("=" * 60)
    print("GridSentinel AI - ML Preprocessing")
    print("=" * 60)

    df = load_dataset()

    print(f"\nTotal records: {len(df):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Target: {TARGET}")

    train, val, test = temporal_split(df)

    print("\n[TEMPORAL SPLIT]")

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        failures = int(part[TARGET].sum())
        rate = failures / len(part) * 100

        print(
            f"{name:12} "
            f"{len(part):6,} records | "
            f"Failures: {failures:4,} | "
            f"Rate: {rate:.2f}%"
        )

        print(
            f"             "
            f"{part['timestamp'].min()} -> "
            f"{part['timestamp'].max()}"
        )

    X_train, y_train = prepare_data(train)
    X_val, y_val = prepare_data(val)
    X_test, y_test = prepare_data(test)

    print("\n[ARRAYS]")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:  ", X_val.shape)
    print("y_val:  ", y_val.shape)

    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    print("\n[FEATURES]")
    for i, feature in enumerate(FEATURES, 1):
        print(f"{i:2}. {feature}")

    print("\n" + "=" * 60)
    print("PREPROCESSING PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
