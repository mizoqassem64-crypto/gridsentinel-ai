import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EVALUATION_FILE = (
    ROOT
    / "models"
    / "v2_risk_engine_evaluation.csv"
)

OUTPUT_JSON = (
    ROOT
    / "models"
    / "v2_risk_threshold_validation.json"
)


print("=" * 70)
print("GridSentinel AI - V2 Risk Threshold Validation")
print("=" * 70)

print("\nLoading evaluation results...")

df = pd.read_csv(EVALUATION_FILE)

print(f"Records: {len(df):,}")


# ============================================================
# Basic validation
# ============================================================

required = [
    "actual_failure",
    "risk_score",
    "risk_level",
    "failure_probability",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Score distribution by actual class
# ============================================================

print("\n" + "=" * 70)
print("[RISK SCORE BY ACTUAL CLASS]")
print("=" * 70)

score_stats = (
    df.groupby("actual_failure")["risk_score"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
    )
)

print(score_stats)


# ============================================================
# Risk level safety analysis
# ============================================================

print("\n" + "=" * 70)
print("[RISK LEVEL SAFETY ANALYSIS]")
print("=" * 70)

risk_summary = (
    df.groupby("risk_level")
    .agg(
        records=("risk_level", "size"),
        failures=("actual_failure", "sum"),
        avg_score=("risk_score", "mean"),
        min_score=("risk_score", "min"),
        max_score=("risk_score", "max"),
    )
)

risk_summary["failure_rate"] = (
    risk_summary["failures"]
    / risk_summary["records"]
)

print(
    risk_summary
    .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
)


# ============================================================
# Failures hidden in LOW / MEDIUM
# ============================================================

print("\n" + "=" * 70)
print("[FAILURES IN LOW / MEDIUM]")
print("=" * 70)

missed = df[
    (df["actual_failure"] == 1)
    & (df["risk_level"].isin(["LOW", "MEDIUM"]))
].copy()

print(f"Count: {len(missed):,}")

if len(missed):
    print(
        missed[
            [
                "timestamp",
                "asset_id",
                "fault_type",
                "actual_failure",
                "failure_probability",
                "risk_score",
                "risk_level",
            ]
        ]
        .sort_values("risk_score")
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# Highest healthy risk
# ============================================================

healthy = df[df["actual_failure"] == 0]

highest_healthy = healthy.loc[
    healthy["risk_score"].idxmax()
]

print("\n" + "=" * 70)
print("[HIGHEST RISK HEALTHY RECORD]")
print("=" * 70)

print(highest_healthy.to_dict())


# ============================================================
# Lowest failure risk
# ============================================================

failures = df[df["actual_failure"] == 1]

lowest_failure = failures.loc[
    failures["risk_score"].idxmin()
]

print("\n" + "=" * 70)
print("[LOWEST RISK ACTUAL FAILURE]")
print("=" * 70)

print(lowest_failure.to_dict())


# ============================================================
# Boundary analysis
# ============================================================

print("\n" + "=" * 70)
print("[BOUNDARY ANALYSIS]")
print("=" * 70)

candidate_thresholds = [
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
]

boundary_results = []

for threshold in candidate_thresholds:

    predicted_high_risk = (
        df["risk_score"] >= threshold
    )

    actual_failure = (
        df["actual_failure"] == 1
    )

    tp = int((predicted_high_risk & actual_failure).sum())
    fp = int(
        (predicted_high_risk & ~actual_failure).sum()
    )

    fn = int(
        (~predicted_high_risk & actual_failure).sum()
    )

    tn = int(
        (~predicted_high_risk & ~actual_failure).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    boundary_results.append(
        {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


boundary_df = pd.DataFrame(boundary_results)

print(
    boundary_df.to_string(index=False)
)


# ============================================================
# Recommended safety boundary
# ============================================================

# Lowest threshold where recall remains >= 95%
# while keeping precision as high as possible.

eligible = boundary_df[
    boundary_df["recall"] >= 0.95
]

if len(eligible):

    recommended = (
        eligible
        .sort_values(
            ["precision", "f1"],
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )

else:

    recommended = (
        boundary_df
        .sort_values(
            "f1",
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )


print("\n" + "=" * 70)
print("[RECOMMENDED SAFETY THRESHOLD]")
print("=" * 70)

print(recommended)


# ============================================================
# Save results
# ============================================================

summary = {
    "records": int(len(df)),
    "actual_failures": int(
        df["actual_failure"].sum()
    ),
    "low_medium_failures": int(
        len(missed)
    ),
    "highest_healthy_risk": float(
        highest_healthy["risk_score"]
    ),
    "lowest_failure_risk": float(
        lowest_failure["risk_score"]
    ),
    "recommended_boundary": recommended,
    "risk_summary": (
        risk_summary
        .reset_index()
        .to_dict(orient="records")
    ),
    "threshold_analysis": (
        boundary_df.to_dict(orient="records")
    ),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 70)
print("V2 RISK THRESHOLD VALIDATION COMPLETE")
print("=" * 70)

print(f"Saved: {OUTPUT_JSON}")
