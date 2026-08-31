from dataclasses import dataclass
from typing import Dict


@dataclass
class OperatingState:
    voltage_pu: float
    current_a: float
    frequency_hz: float
    active_power_mw: float
    reactive_power_mvar: float
    power_factor: float
    temperature_c: float
    load_percent: float
    thd_percent: float


def apply_overload(state: OperatingState, severity: float) -> OperatingState:
    """
    Simulates transformer overload.

    severity:
        0.0 = no additional stress
        1.0 = severe stress
    """
    severity = max(0.0, min(1.0, severity))

    state.load_percent += 20.0 * severity
    state.current_a *= 1.0 + (0.25 * severity)
    state.temperature_c += 30.0 * severity
    state.voltage_pu -= 0.03 * severity
    state.thd_percent += 2.5 * severity
    state.power_factor -= 0.04 * severity

    return state


def apply_overheating(state: OperatingState, severity: float) -> OperatingState:
    """
    Simulates abnormal thermal behavior.
    """
    severity = max(0.0, min(1.0, severity))

    state.temperature_c += 40.0 * severity
    state.power_factor -= 0.02 * severity

    return state


def apply_voltage_instability(
    state: OperatingState, severity: float
) -> OperatingState:
    """
    Simulates voltage instability.
    """
    severity = max(0.0, min(1.0, severity))

    state.voltage_pu -= 0.08 * severity
    state.frequency_hz -= 0.15 * severity
    state.thd_percent += 3.0 * severity

    return state


def apply_harmonic_distortion(
    state: OperatingState, severity: float
) -> OperatingState:
    """
    Simulates increasing harmonic distortion.
    """
    severity = max(0.0, min(1.0, severity))

    state.thd_percent += 10.0 * severity
    state.temperature_c += 8.0 * severity

    return state


def state_to_dict(state: OperatingState) -> Dict:
    return {
        "voltage_pu": round(state.voltage_pu, 4),
        "current_a": round(state.current_a, 2),
        "frequency_hz": round(state.frequency_hz, 3),
        "active_power_mw": round(state.active_power_mw, 3),
        "reactive_power_mvar": round(state.reactive_power_mvar, 3),
        "power_factor": round(state.power_factor, 4),
        "temperature_c": round(state.temperature_c, 2),
        "load_percent": round(state.load_percent, 2),
        "thd_percent": round(state.thd_percent, 3),
    }


if __name__ == "__main__":
    normal = OperatingState(
        voltage_pu=1.0,
        current_a=650.0,
        frequency_hz=50.0,
        active_power_mw=32.0,
        reactive_power_mvar=10.0,
        power_factor=0.95,
        temperature_c=62.0,
        load_percent=72.0,
        thd_percent=2.5,
    )

    print("NORMAL STATE")
    print(state_to_dict(normal))

    overloaded = apply_overload(
        OperatingState(**normal.__dict__),
        severity=1.0,
    )

    print("\nAFTER OVERLOAD")
    print(state_to_dict(overloaded))
