import pandas as pd
import numpy as np

INPUT = "datasets/grid_operations.csv"
OUTPUT = "datasets/grid_features.csv"

print("=" * 70)
print("GridSentinel AI - Feature Engineering")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nOriginal shape: {df.shape}")

# ---------------------------------------------------------
# Electrical deviation features
# ---------------------------------------------------------

df["voltage_deviation"] = np.abs(df["voltage_pu"] - 1.0)

df["frequency_deviation"] = np.abs(
    df["frequency_hz"] - 50.0
)

# ---------------------------------------------------------
# Power features
# ---------------------------------------------------------

df["apparent_power_mva"] = np.sqrt(
    df["active_power_mw"] ** 2 +
    df["reactive_power_mvar"] ** 2
)

df["power_utilization"] = (
    df["apparent_power_mva"] /
    df["rated_mva"].replace(0, np.nan)
)

df["reactive_to_active_ratio"] = (
    np.abs(df["reactive_power_mvar"]) /
    np.maximum(np.abs(df["active_power_mw"]), 1e-6)
)

# ---------------------------------------------------------
# Thermal / loading stress
# ---------------------------------------------------------

df["thermal_stress"] = (
    df["temperature_c"] *
    df["load_percent"] / 100.0
)

df["overload_stress"] = np.maximum(
    df["load_percent"] - 100.0,
    0
)

df["temperature_excess"] = np.maximum(
    df["temperature_c"] - 80.0,
    0
)

# ---------------------------------------------------------
# Harmonic stress
# ---------------------------------------------------------

df["harmonic_stress"] = np.maximum(
    df["thd_percent"] - 5.0,
    0
)

# ---------------------------------------------------------
# Power quality stress
# ---------------------------------------------------------

df["power_factor_deviation"] = np.abs(
    1.0 - df["power_factor"]
)

df["electrical_stress"] = (
    df["voltage_deviation"] * 10.0
    + df["frequency_deviation"] * 5.0
    + df["power_factor_deviation"] * 10.0
    + df["harmonic_stress"] / 10.0
)

# ---------------------------------------------------------
# Combined operational stress
# ---------------------------------------------------------

df["combined_stress"] = (
    df["thermal_stress"]
    + df["power_utilization"] * 50.0
    + df["harmonic_stress"]
    + df["voltage_deviation"] * 20.0
)

# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

new_features = [
    "voltage_deviation",
    "frequency_deviation",
    "apparent_power_mva",
    "power_utilization",
    "reactive_to_active_ratio",
    "thermal_stress",
    "overload_stress",
    "temperature_excess",
    "harmonic_stress",
    "power_factor_deviation",
    "electrical_stress",
    "combined_stress",
]

print("\n[NEW FEATURES]")

for i, feature in enumerate(new_features, 1):
    print(
        f"{i:2}. {feature:30} "
        f"min={df[feature].min():.4f} "
        f"max={df[feature].max():.4f}"
    )

print("\n[VALIDATION]")

print(
    "NaN values:",
    int(df[new_features].isna().sum().sum())
)

print(
    "Infinite values:",
    int(
        np.isinf(
            df[new_features].to_numpy()
        ).sum()
    )
)

df.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print(f"New shape: {df.shape}")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING PASS")
print("=" * 70)
