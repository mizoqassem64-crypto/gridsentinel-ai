from ai.ml.safety_guard_v2 import apply_safety_guard


print("=" * 70)
print("GridSentinel AI - V2.1 Safety Guard Test")
print("=" * 70)


tests = [
    {
        "name": "Normal healthy asset",
        "risk_score": 10,
        "risk_level": "LOW",
        "failure_probability": 0.01,
        "data": {
            "temperature_c": 65,
            "load_percent": 50,
            "thd_percent": 2,
            "voltage_pu": 1.0,
            "frequency_hz": 50.0,
            "previous_faults": 0,
        },
    },
    {
        "name": "Severe temperature",
        "risk_score": 15,
        "risk_level": "LOW",
        "failure_probability": 0.10,
        "data": {
            "temperature_c": 105,
            "load_percent": 60,
            "thd_percent": 2,
            "voltage_pu": 1.0,
            "frequency_hz": 50.0,
            "previous_faults": 0,
        },
    },
    {
        "name": "Multiple severe conditions",
        "risk_score": 20,
        "risk_level": "LOW",
        "failure_probability": 0.15,
        "data": {
            "temperature_c": 105,
            "load_percent": 95,
            "thd_percent": 12,
            "voltage_pu": 0.94,
            "frequency_hz": 49.7,
            "previous_faults": 2,
        },
    },
    {
        "name": "High ML probability",
        "risk_score": 40,
        "risk_level": "MEDIUM",
        "failure_probability": 0.85,
        "data": {
            "temperature_c": 70,
            "load_percent": 60,
            "thd_percent": 3,
            "voltage_pu": 1.0,
            "frequency_hz": 50.0,
            "previous_faults": 0,
        },
    },
    {
        "name": "Repeated faults",
        "risk_score": 15,
        "risk_level": "LOW",
        "failure_probability": 0.10,
        "data": {
            "temperature_c": 70,
            "load_percent": 60,
            "thd_percent": 3,
            "voltage_pu": 1.0,
            "frequency_hz": 50.0,
            "previous_faults": 6,
        },
    },
]


for test in tests:

    result = apply_safety_guard(
        risk_score=test["risk_score"],
        risk_level=test["risk_level"],
        failure_probability=test["failure_probability"],
        operational_data=test["data"],
    )

    print("\n" + "-" * 70)
    print(test["name"])
    print("-" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")


print("\n" + "=" * 70)
print("V2.1 SAFETY GUARD TEST COMPLETE")
print("=" * 70)
