import pandas as pd
import numpy as np

DATASET = "datasets/grid_features.csv"

print("=" * 70)
print("GridSentinel AI - Feature Selection")
print("=" * 70)

df = pd.read_csv(DATASET)

target = "failure"

candidate_features = [
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

    # Engineered
    "voltage_deviation",
    "frequency_deviation",
    "apparent_power_mva",
    "power_utilization",
    "reactive_to_active_ratio",
    "thermal_stress",
    "temperature_excess",
    "harmonic_stress",
    "power_factor_deviation",
    "electrical_stress",
    "combined_stress",
]

print(f"\nRecords: {len(df):,}")
print(f"Candidate features: {len(candidate_features)}")

# ---------------------------------------------------------
# Correlation with target
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[TARGET CORRELATION]")
print("=" * 70)

correlations = (
    df[candidate_features + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .abs()
    .sort_values(ascending=False)
)

for feature, value in correlations.items():
    print(f"{feature:30} {value:.6f}")

# ---------------------------------------------------------
# Feature-feature correlation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[HIGH FEATURE CORRELATIONS]")
print("=" * 70)

corr_matrix = df[candidate_features].corr()

pairs = []

for i in range(len(candidate_features)):
    for j in range(i + 1, len(candidate_features)):
        a = candidate_features[i]
        b = candidate_features[j]

        value = abs(corr_matrix.loc[a, b])

        if value >= 0.90:
            pairs.append((a, b, value))

pairs.sort(key=lambda x: x[2], reverse=True)

if pairs:
    for a, b, value in pairs:
        print(f"{a:30} <-> {b:30} {value:.6f}")
else:
    print("No highly correlated feature pairs found.")

# ---------------------------------------------------------
# Zero variance
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[ZERO / NEAR-ZERO VARIANCE]")
print("=" * 70)

for feature in candidate_features:
    std = df[feature].std()

    if std < 1e-8:
        print(f"{feature:30} std={std:.10f}")

# ---------------------------------------------------------
# Final recommendations
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[FEATURE SELECTION RECOMMENDATION]")
print("=" * 70)

print("""
Keep features based on:

1. Meaningful target correlation
2. Operational interpretability
3. Low redundancy
4. No leakage
5. Physical relevance to grid failure

IMPORTANT:
overload_stress is intentionally excluded because
load_percent never exceeds 100% in this dataset.
""")

print("=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)
