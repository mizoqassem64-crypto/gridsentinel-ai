import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "grid_features.csv"
OPERATIONS = ROOT / "datasets" / "grid_operations.csv"
MODEL = ROOT / "models" / "failure_predictor_v2.pt"
SCALER = ROOT / "models" / "failure_scaler_v2.json"
METADATA = ROOT / "models" / "failure_model_metadata_v2.json"
THRESHOLD = ROOT / "models" / "failure_threshold_v2.json"


PASSED = 0
FAILED = 0


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(name, condition, details=""):
    global PASSED, FAILED

    if condition:
        PASSED += 1
        print(f"[PASS] {name}")
        if details:
            print(f"       {details}")
    else:
        FAILED += 1
        print(f"[FAIL] {name}")
        if details:
            print(f"       {details}")


def main():

    section("GridSentinel AI - FULL V1-V5 INTEGRATION TEST")

    print(f"Project root: {ROOT}")

    # ============================================================
    # 1. PROJECT STRUCTURE
    # ============================================================

    section("PHASE 1 - PROJECT / DATA FOUNDATION")

    required_files = [
        DATASET,
        OPERATIONS,
        MODEL,
        SCALER,
        METADATA,
        THRESHOLD,
        ROOT / "ai" / "ml" / "risk_engine_v2.py",
        ROOT / "ai" / "ml" / "safety_guard_v2.py",
        ROOT / "simulation" / "grid.py",
        ROOT / "simulation" / "fault_injection.py",
    ]

    for path in required_files:
        check(
            f"Required artifact: {path.relative_to(ROOT)}",
            path.exists(),
        )

    # ============================================================
    # 2. DATA VALIDATION
    # ============================================================

    section("PHASE 2 - DATA / FEATURE PIPELINE")

    try:
        df = pd.read_csv(DATASET)

        check(
            "Dataset loads successfully",
            len(df) > 0,
            f"records={len(df)}",
        )

        check(
            "Dataset has no completely empty columns",
            not df.isna().all().any(),
        )

        check(
            "Dataset contains actual_failure",
            "failure" in df.columns,
        )

        check(
            "Dataset contains fault_type",
            "fault_type" in df.columns,
        )

        if "actual_failure" in df.columns:
            failure_rate = df["failure"].mean()

            check(
                "Failure labels are binary",
                set(df["actual_failure"].dropna().unique())
                <= {0, 1},
                f"failure_rate={failure_rate:.4f}",
            )

    except Exception as exc:
        check(
            "Dataset validation",
            False,
            str(exc),
        )

    # ============================================================
    # 3. V2 ML MODEL
    # ============================================================

    section("PHASE 3 - V2 ML PREDICTION")

    try:
        from ai.ml.risk_engine_v2 import predict_v2

        # Use the first valid dataset record.
        row = df.iloc[0].to_dict()

        prediction = predict_v2(row)

        probability = float(
            prediction["failure_probability"]
        )

        status = prediction["status"]

        check(
            "V2 model prediction executes",
            True,
            f"probability={probability:.6f}",
        )

        check(
            "Failure probability is valid",
            0.0 <= probability <= 1.0,
            f"probability={probability:.6f}",
        )

        check(
            "Prediction status is valid",
            status in ("NORMAL", "FAILURE"),
            f"status={status}",
        )

    except Exception as exc:
        check(
            "V2 model prediction",
            False,
            str(exc),
        )
        traceback.print_exc()

    # ============================================================
    # 4. RISK ENGINE + SAFETY GUARD
    # ============================================================

    section("PHASE 4 - RISK INTELLIGENCE + SAFETY GUARD")

    try:
        from ai.ml.risk_engine_v2 import assess_risk_v2

        row = df.iloc[0].to_dict()

        result = assess_risk_v2(row)

        required_keys = [
            "risk_score",
            "risk_level",
            "failure_probability",
            "prediction",
            "alert_state",
            "guard_applied",
            "guard_reasons",
            "reasons",
            "recommended_action",
            "interpretation",
        ]

        for key in required_keys:
            check(
                f"Risk result contains '{key}'",
                key in result,
            )

        score = float(result["risk_score"])
        level = result["risk_level"]

        check(
            "Risk score is bounded",
            0.0 <= score <= 100.0,
            f"risk_score={score}",
        )

        check(
            "Risk level is valid",
            level in (
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ),
            f"risk_level={level}",
        )

        print()
        print("Sample integrated result:")
        print(f"  risk_score:          {score}")
        print(f"  risk_level:          {level}")
        print(
            f"  failure_probability: "
            f"{result['failure_probability']:.6f}"
        )
        print(
            f"  prediction_status:   "
            f"{result['prediction']}"
        )
        print(
            f"  alert_state:         "
            f"{result['alert_state']}"
        )
        print(
            f"  guard_applied:       "
            f"{result['guard_applied']}"
        )

    except Exception as exc:
        check(
            "Risk engine integration",
            False,
            str(exc),
        )
        traceback.print_exc()

    # ============================================================
    # 5. SAFETY GUARD SCENARIOS
    # ============================================================

    section("PHASE 5 - SAFETY / FAILURE SCENARIO TESTS")

    try:
        from ai.ml.safety_guard_v2 import apply_safety_guard

        scenarios = [
            (
                "Normal healthy asset",
                {
                    "temperature_c": 65,
                    "load_percent": 60,
                    "thd_percent": 3,
                    "voltage_pu": 1.0,
                    "frequency_hz": 50.0,
                    "previous_faults": 0,
                },
                0.05,
                "LOW",
            ),
            (
                "Severe temperature",
                {
                    "temperature_c": 105,
                    "load_percent": 60,
                    "thd_percent": 3,
                    "voltage_pu": 1.0,
                    "frequency_hz": 50.0,
                    "previous_faults": 0,
                },
                0.05,
                "MEDIUM",
            ),
            (
                "Multiple severe conditions",
                {
                    "temperature_c": 110,
                    "load_percent": 95,
                    "thd_percent": 12,
                    "voltage_pu": 0.92,
                    "frequency_hz": 49.5,
                    "previous_faults": 0,
                },
                0.10,
                "CRITICAL",
            ),
            (
                "High ML probability",
                {
                    "temperature_c": 70,
                    "load_percent": 65,
                    "thd_percent": 3,
                    "voltage_pu": 1.0,
                    "frequency_hz": 50.0,
                    "previous_faults": 0,
                },
                0.80,
                "HIGH",
            ),
            (
                "Repeated faults",
                {
                    "temperature_c": 70,
                    "load_percent": 65,
                    "thd_percent": 3,
                    "voltage_pu": 1.0,
                    "frequency_hz": 50.0,
                    "previous_faults": 5,
                },
                0.10,
                "MEDIUM",
            ),
        ]

        for name, operational_data, probability, expected_level in scenarios:

            result = apply_safety_guard(
                risk_score=10.0,
                risk_level="LOW",
                failure_probability=probability,
                operational_data=operational_data,
            )

            actual_level = result["risk_level"]

            check(
                name,
                actual_level == expected_level,
                f"expected={expected_level}, actual={actual_level}",
            )

            print(
                f"       score={result['risk_score']} "
                f"alert={result['alert_state']}"
            )

    except Exception as exc:
        check(
            "Safety scenario tests",
            False,
            str(exc),
        )
        traceback.print_exc()

    # ============================================================
    # FINAL
    # ============================================================

    section("FULL INTEGRATION TEST SUMMARY")

    total = PASSED + FAILED

    print(f"Total checks: {total}")
    print(f"Passed:       {PASSED}")
    print(f"Failed:       {FAILED}")

    print()

    if FAILED == 0:
        print("STATUS: PASS")
        print()
        print(
            "GridSentinel AI V1-V5 integration "
            "baseline is operational."
        )
        return 0

    print("STATUS: FAIL")
    print()
    print(
        "Do NOT proceed to Phase 6 deployment "
        "until failed checks are investigated."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
