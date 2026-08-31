import csv
from collections import Counter
from pathlib import Path


DATASET = Path("datasets/grid_operations.csv")


REQUIRED_COLUMNS = [
    "timestamp",
    "asset_id",
    "asset_type",
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
    "operating_state",
    "fault_type",
    "failure",
    "failure_horizon_hours",
]


def check_range(rows, column, minimum, maximum):
    invalid = []

    for index, row in enumerate(rows, start=2):
        try:
            value = float(row[column])
        except (ValueError, TypeError):
            invalid.append(index)
            continue

        if not minimum <= value <= maximum:
            invalid.append(index)

    return invalid


def main():
    print("=" * 65)
    print("GridSentinel AI - Dataset Validation")
    print("=" * 65)

    if not DATASET.exists():
        print(f"ERROR: Dataset not found: {DATASET}")
        return

    with DATASET.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    print(f"\nRecords: {len(rows):,}")
    print(f"Columns: {len(columns)}")

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    print("\n[SCHEMA]")
    if missing_columns:
        print("FAIL - Missing columns:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("PASS - All required columns present")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_values = Counter()

    for row in rows:
        for column in columns:
            if row.get(column, "").strip() == "":
                missing_values[column] += 1

    print("\n[MISSING VALUES]")

    if not missing_values:
        print("PASS - No missing values")
    else:
        for column, count in missing_values.items():
            print(f"WARNING - {column}: {count}")

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    unique_rows = {
        tuple(row.get(column, "") for column in columns)
        for row in rows
    }

    duplicate_count = len(rows) - len(unique_rows)

    print("\n[DUPLICATES]")

    if duplicate_count == 0:
        print("PASS - No duplicate rows")
    else:
        print(f"WARNING - {duplicate_count} duplicate rows")

    # ---------------------------------------------------------
    # Physical plausibility checks
    # ---------------------------------------------------------

    checks = [
        ("Voltage", "voltage_pu", 0.85, 1.15),
        ("Frequency", "frequency_hz", 48.0, 52.0),
        ("Power Factor", "power_factor", 0.70, 1.00),
        ("Temperature", "temperature_c", 0.0, 150.0),
        ("Load", "load_percent", 0.0, 120.0),
        ("THD", "thd_percent", 0.0, 25.0),
        ("Current", "current_a", 0.0, 2000.0),
    ]

    print("\n[PHYSICAL PLAUSIBILITY]")

    physical_failures = 0

    for label, column, minimum, maximum in checks:
        invalid = check_range(
            rows,
            column,
            minimum,
            maximum,
        )

        if invalid:
            physical_failures += len(invalid)
            print(
                f"FAIL - {label}: "
                f"{len(invalid)} invalid records"
            )
        else:
            print(
                f"PASS - {label}: "
                f"{minimum} <= value <= {maximum}"
            )

    # ---------------------------------------------------------
    # Failure statistics
    # ---------------------------------------------------------

    failure_counter = Counter(
        int(row["failure"])
        for row in rows
    )

    failure_count = failure_counter.get(1, 0)
    healthy_count = failure_counter.get(0, 0)

    failure_rate = (
        failure_count / len(rows) * 100
        if rows
        else 0
    )

    print("\n[FAILURE DISTRIBUTION]")
    print(f"Healthy: {healthy_count:,}")
    print(f"Failure: {failure_count:,}")
    print(f"Failure rate: {failure_rate:.2f}%")

    # ---------------------------------------------------------
    # Fault distribution
    # ---------------------------------------------------------

    faults = Counter(
        row["fault_type"]
        for row in rows
    )

    print("\n[FAULT DISTRIBUTION]")

    for fault, count in faults.most_common():
        percentage = count / len(rows) * 100
        print(
            f"{fault:25s} "
            f"{count:6,} "
            f"({percentage:5.2f}%)"
        )

    # ---------------------------------------------------------
    # Asset distribution
    # ---------------------------------------------------------

    assets = Counter(
        row["asset_id"]
        for row in rows
    )

    print("\n[ASSET DISTRIBUTION]")

    for asset, count in assets.items():
        print(f"{asset}: {count:,}")

    # ---------------------------------------------------------
    # Data leakage warning
    # ---------------------------------------------------------

    leakage_columns = [
        "operating_state",
        "fault_type",
        "failure",
        "failure_horizon_hours",
    ]

    print("\n[DATA LEAKAGE CHECK]")
    print("The following columns must NOT be used as model features:")

    for column in leakage_columns:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 65)

    if (
        not missing_columns
        and not missing_values
        and duplicate_count == 0
        and physical_failures == 0
    ):
        print("DATASET STATUS: PASS")
        print("Dataset is ready for ML preprocessing.")
    else:
        print("DATASET STATUS: REVIEW REQUIRED")

    print("=" * 65)


if __name__ == "__main__":
    main()
