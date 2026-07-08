"""Generate synthetic shipment data.

Produces:
  data/shipments_train.csv   — training set (baseline distribution)
  data/shipments_batch.csv   — nightly batch input (same distribution)
  data/shipments_drifted.csv — post-holiday scenario: port congestion shifted up
                               (used to demo drift detection firing)
  data/baseline_stats.csv    — raw baseline feature sample kept for PSI

The label has real signal: delay probability rises with congestion, weather
risk, transshipments and historical delay rate, falls with lead time and
carrier reliability, plus noise — so the trained model is non-trivial but
learnable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402

PORTS = ["SGN", "SIN", "HKG", "SHA", "PUS", "LAX", "RTM", "HAM"]
CONTAINERS = ["20GP", "40GP", "40HC", "45HC"]
CUSTOMERS = ["CUS_A", "CUS_B", "CUS_C", "CUS_D"]

rng = np.random.default_rng(42)


def make_shipments(n: int, congestion_shift: float = 0.0, id_prefix: str = "SHP") -> pd.DataFrame:
    congestion = np.clip(rng.beta(2, 3, n) + congestion_shift, 0, 1)
    weather = np.clip(rng.beta(2, 4, n), 0, 1)
    lead_days = rng.integers(1, 30, n)
    transship = rng.integers(0, 4, n)
    hist_delay = np.clip(rng.beta(2, 6, n), 0, 1)
    reliability = np.clip(rng.beta(6, 2, n), 0, 1)

    origins = rng.choice(PORTS, n)
    dests = rng.choice(PORTS, n)

    # Ground-truth delay mechanism
    logit = (
        -1.2
        + 3.0 * congestion
        + 1.8 * weather
        + 0.5 * transship
        + 2.0 * hist_delay
        - 2.2 * reliability
        - 0.05 * lead_days
        + rng.normal(0, 0.6, n)
    )
    delayed = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    return pd.DataFrame({
        "shipment_id": [f"{id_prefix}_{i:05d}" for i in range(n)],
        "customer_id": rng.choice(CUSTOMERS, n),
        "origin_port": origins,
        "destination_port": dests,
        "container_type": rng.choice(CONTAINERS, n),
        "booking_lead_days": lead_days,
        "transshipment_count": transship,
        "port_congestion_score": congestion.round(3),
        "weather_risk_score": weather.round(3),
        "historical_delay_rate": hist_delay.round(3),
        "carrier_reliability_score": reliability.round(3),
        "delayed": delayed,
    })


def main() -> None:
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    train = make_shipments(5000, id_prefix="TRN")
    batch = make_shipments(500, id_prefix="SHP").drop(columns=["delayed"])
    drifted = make_shipments(500, congestion_shift=0.30, id_prefix="DRF").drop(columns=["delayed"])

    train.to_csv(data_dir / "shipments_train.csv", index=False)
    batch.to_csv(data_dir / "shipments_batch.csv", index=False)
    drifted.to_csv(data_dir / "shipments_drifted.csv", index=False)
    # Baseline sample for PSI = the training feature distribution
    train.drop(columns=["delayed"]).to_csv(data_dir / "baseline_stats.csv", index=False)

    print(f"train={len(train)} rows (delay rate {train['delayed'].mean():.2%})")
    print(f"batch={len(batch)} rows, drifted={len(drifted)} rows -> {data_dir}")


if __name__ == "__main__":
    main()
