from dataclasses import dataclass
from typing import Dict


@dataclass
class Transformer:
    asset_id: str
    rated_mva: float
    voltage_kv: float
    age_years: int
    base_temperature: float = 55.0


@dataclass
class Feeder:
    asset_id: str
    rated_current_a: float
    voltage_kv: float
    length_km: float
    age_years: int


TRANSFORMERS = [
    Transformer("T01", 40.0, 33.0, 6),
    Transformer("T02", 63.0, 33.0, 11),
    Transformer("T03", 40.0, 33.0, 17),
]

FEEDERS = [
    Feeder("F01", 800.0, 33.0, 12.5, 8),
    Feeder("F02", 1000.0, 33.0, 18.0, 13),
    Feeder("F03", 630.0, 33.0, 9.5, 5),
]


def transformer_snapshot(transformer: Transformer) -> Dict:
    """
    Returns the static engineering metadata for a transformer.
    Dynamic operating measurements will be generated separately.
    """
    return {
        "asset_id": transformer.asset_id,
        "asset_type": "transformer",
        "rated_mva": transformer.rated_mva,
        "voltage_kv": transformer.voltage_kv,
        "age_years": transformer.age_years,
        "base_temperature_c": transformer.base_temperature,
    }


def feeder_snapshot(feeder: Feeder) -> Dict:
    """
    Returns the static engineering metadata for a feeder.
    """
    return {
        "asset_id": feeder.asset_id,
        "asset_type": "feeder",
        "rated_current_a": feeder.rated_current_a,
        "voltage_kv": feeder.voltage_kv,
        "length_km": feeder.length_km,
        "age_years": feeder.age_years,
    }


if __name__ == "__main__":
    print("GridSentinel AI - Electrical Grid")
    print("=" * 45)

    print("\nTransformers:")
    for transformer in TRANSFORMERS:
        print(transformer_snapshot(transformer))

    print("\nFeeders:")
    for feeder in FEEDERS:
        print(feeder_snapshot(feeder))
