"""Single source of truth for the feature schema.

Both training (scripts/train_model.py) and serving (src/api) import FEATURE_COLUMNS
from here. This is the cheapest possible defense against training/serving skew:
if the schema changes, both sides change together or fail loudly.
"""
import pandas as pd

FEATURE_COLUMNS = [
    "booking_lead_days",
    "transshipment_count",
    "port_congestion_score",
    "weather_risk_score",
    "historical_delay_rate",
    "carrier_reliability_score",
]

TARGET_COLUMN = "delayed"

# Features monitored for drift (chosen because they are external-world signals
# that shift with seasonality/holidays, unlike e.g. container_type).
DRIFT_FEATURES = [
    "port_congestion_score",
    "weather_risk_score",
    "booking_lead_days",
]


def to_feature_frame(records: list[dict]) -> pd.DataFrame:
    """Build a model-ready frame in canonical column order."""
    df = pd.DataFrame(records)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return df[FEATURE_COLUMNS]
