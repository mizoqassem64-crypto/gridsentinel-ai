import json
from pathlib import Path

import pandas as pd


print("=" * 70)
print("GridSentinel AI - V2 Risk Investigation")
print("=" * 70)


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "datasets" / "grid_features.csv"
EVALUATION = (
    ROOT
    / "models"
    / "v2_risk_engine_evaluation.csv"
)

OUTPUT = (
    ROOT
    / "models"
    / "v2_risk_investigation.json"
)


# ============================================================
# Load
# ============================================================

print("\nLoading dataset...")
df = pd.read_csv(DATASET)

print("Loading evaluation...")
evaluation = pd.read_csv(EVALUATION)


# ============================================================
# Merge evaluation with physical features
# ============================================================

KEYS = [
    "timestamp",
    "asset_id",
    "fault_type",
]

feature_columns = [
    "temperature_c",
    "load_percent",
    "thd_percent",
    "voltage_pu",
    "frequency_hz",
    "power_factor",
    "previous_faults",
    "criticality",
]

available_features = [
    c for c in feature_columns
    if c in df.columns
]


merged = evaluation.merge(
    df[
        KEYS + available_features
    ],
    on=KEYS,
    how="left",
)


# ============================================================
# 1. Asset analysis
# ============================================================

asset_analysis = {}

for asset in sorted(
    merged["asset_id"].dropna().unique()
):

    subset = merged[
        merged["asset_id"] == asset
    ]

    failures = subset[
        subset["actual_failure"] == 1
    ]

    escaped = subset[
        (subset["actual_failure"] == 1)
        & (
            subset["risk_level"].isin(
                ["LOW", "MEDIUM"]
            )
        )
    ]

    asset_analysis[asset] = {
        "records": int(len(subset)),
        "actual_failures": int(
            subset["actual_failure"].sum()
        ),
        "failure_rate": round(
            float(
                subset["actual_failure"].mean()
            ),
            6,
        ),
        "mean_risk_score": round(
            float(
                subset["risk_score"].mean()
            ),
            2,
        ),
        "mean_failure_probability": round(
            float(
                subset[
                    "failure_probability"
                ].mean()
            ),
            6,
        ),
        "escaped_failures": int(
            len(escaped)
        ),
        "escaped_failure_rate": round(
            len(escaped)
            / len(failures)
            if len(failures)
            else 0.0,
            6,
        ),
    }


# ============================================================
# 2. Escaped failure analysis
# ============================================================

escaped = merged[
    (merged["actual_failure"] == 1)
    & (
        merged["risk_level"].isin(
            ["LOW", "MEDIUM"]
        )
    )
].copy()


escaped_records = []

for _, row in escaped.iterrows():

    record = {
        "timestamp": row.get(
            "timestamp"
        ),
        "asset_id": row.get(
            "asset_id"
        ),
        "fault_type": row.get(
            "fault_type"
        ),
        "risk_level": row.get(
            "risk_level"
        ),
        "risk_score": float(
            row["risk_score"]
        ),
        "failure_probability": float(
            row["failure_probability"]
        ),
    }

    for feature in available_features:

        value = row.get(feature)

        if pd.notna(value):

            record[feature] = float(value)

    escaped_records.append(record)


# ============================================================
# 3. Escaped failures by fault
# ============================================================

escaped_by_fault = (
    escaped[
        "fault_type"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# 4. Escaped failures by asset
# ============================================================

escaped_by_asset = (
    escaped[
        "asset_id"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# 5. Critical false positives
# ============================================================

critical_fp = merged[
    (merged["actual_failure"] == 0)
    & (
        merged["risk_level"]
        == "CRITICAL"
    )
].copy()


critical_fp_by_asset = (
    critical_fp[
        "asset_id"
    ]
    .value_counts()
    .to_dict()
)


critical_fp_by_fault = (
    critical_fp[
        "fault_type"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# 6. Physical condition comparison
# ============================================================

condition_analysis = {}

for feature in available_features:

    failure_values = merged[
        merged["actual_failure"] == 1
    ][feature].dropna()

    healthy_values = merged[
        merged["actual_failure"] == 0
    ][feature].dropna()

    escaped_values = escaped[
        feature
    ].dropna()

    condition_analysis[feature] = {
        "failure_mean": round(
            float(
                failure_values.mean()
            ),
            4,
        )
        if len(failure_values)
        else None,

        "healthy_mean": round(
            float(
                healthy_values.mean()
            ),
            4,
        )
        if len(healthy_values)
        else None,

        "escaped_failure_mean": round(
            float(
                escaped_values.mean()
            ),
            4,
        )
        if len(escaped_values)
        else None,
    }


# ============================================================
# 7. Escaped failure statistics
# ============================================================

escaped_statistics = {}

if len(escaped):

    escaped_statistics = {
        "count": int(len(escaped)),

        "mean_risk_score": round(
            float(
                escaped[
                    "risk_score"
                ].mean()
            ),
            2,
        ),

        "minimum_risk_score": round(
            float(
                escaped[
                    "risk_score"
                ].min()
            ),
            2,
        ),

        "maximum_risk_score": round(
            float(
                escaped[
                    "risk_score"
                ].max()
            ),
            2,
        ),

        "mean_probability": round(
            float(
                escaped[
                    "failure_probability"
                ].mean()
            ),
            6,
        ),

        "minimum_probability": round(
            float(
                escaped[
                    "failure_probability"
                ].min()
            ),
            6,
        ),

        "maximum_probability": round(
            float(
                escaped[
                    "failure_probability"
                ].max()
            ),
            6,
        ),
    }


# ============================================================
# 8. Investigation report
# ============================================================

report = {
    "engine": "GridSentinel AI",
    "version": "V2.1",

    "asset_analysis": asset_analysis,

    "escaped_failures": {
        "count": int(len(escaped)),
        "by_fault_type": escaped_by_fault,
        "by_asset": escaped_by_asset,
        "statistics": escaped_statistics,
        "records": escaped_records,
    },

    "critical_false_positives": {
        "count": int(len(critical_fp)),
        "by_asset": critical_fp_by_asset,
        "by_fault_type": critical_fp_by_fault,
    },

    "physical_condition_analysis":
        condition_analysis,

    "next_step": (
        "Investigate asset-specific and "
        "fault-specific failure patterns "
        "before modifying production thresholds."
    ),
}


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Console
# ============================================================

print("\n" + "=" * 70)
print("[ASSET ANALYSIS]")
print("=" * 70)

for asset, data in asset_analysis.items():

    print(
        f"{asset}: "
        f"failures={data['actual_failures']} "
        f"failure_rate={data['failure_rate']:.4f} "
        f"escaped={data['escaped_failures']} "
        f"mean_risk={data['mean_risk_score']:.2f}"
    )


print("\n" + "=" * 70)
print("[ESCAPED FAILURES]")
print("=" * 70)

print(f"Count: {len(escaped)}")
print(
    "By fault:",
    escaped_by_fault,
)
print(
    "By asset:",
    escaped_by_asset,
)


print("\n" + "=" * 70)
print("[ESCAPED FAILURE STATISTICS]")
print("=" * 70)

print(
    json.dumps(
        escaped_statistics,
        indent=2,
    )
)


print("\n" + "=" * 70)
print("[CRITICAL FALSE POSITIVES]")
print("=" * 70)

print(
    f"Count: {len(critical_fp)}"
)

print(
    "By asset:",
    critical_fp_by_asset,
)

print(
    "By fault:",
    critical_fp_by_fault,
)


print("\n" + "=" * 70)
print("[PHYSICAL CONDITION ANALYSIS]")
print("=" * 70)

for feature, values in condition_analysis.items():

    print(
        f"{feature:<22} "
        f"failure={values['failure_mean']} "
        f"healthy={values['healthy_mean']} "
        f"escaped={values['escaped_failure_mean']}"
    )


print("\n" + "=" * 70)
print("V2 RISK INVESTIGATION COMPLETE")
print("=" * 70)

print(f"Saved: {OUTPUT}")

print("=" * 70)
