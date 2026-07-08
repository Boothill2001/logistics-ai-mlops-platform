"""Prediction service — the single execution path for delay predictions.

Used by the real-time API, the batch scorer, and the copilot's
MLInferenceNode. One code path = one set of bugs = consistent behavior
everywhere (and the LLM never computes a probability itself).
"""
from dataclasses import dataclass

from src.config import settings
from src.ml.canary import route_to_canary
from src.ml.features import to_feature_frame
from src.ml.registry import ModelRegistry, registry as default_registry

MODEL_NAME = "shipment_delay_model"


@dataclass
class Prediction:
    shipment_id: str
    delay_probability: float
    risk_level: str
    model_name: str
    model_version: str


def risk_level(probability: float) -> str:
    if probability >= settings.risk_high:
        return "high"
    if probability >= settings.risk_medium:
        return "medium"
    return "low"


class DelayPredictor:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or default_registry

    def ready(self) -> bool:
        return self.registry.get_by_stage(MODEL_NAME, "production") is not None

    def predict(self, record: dict, force_version: str | None = None) -> Prediction:
        """Predict for one shipment dict (must contain FEATURE_COLUMNS + shipment_id)."""
        if force_version:
            loaded = self.registry.get_version(MODEL_NAME, force_version)
        else:
            canary = self.registry.get_by_stage(MODEL_NAME, "canary")
            use_canary = canary is not None and route_to_canary(record["shipment_id"])
            loaded = canary if use_canary else self.registry.get_by_stage(MODEL_NAME, "production")
        if loaded is None:
            raise RuntimeError("No production model registered")

        X = to_feature_frame([record])
        probability = float(loaded.model.predict_proba(X)[0, 1])
        return Prediction(
            shipment_id=record["shipment_id"],
            delay_probability=round(probability, 4),
            risk_level=risk_level(probability),
            model_name=loaded.name,
            model_version=loaded.version,
        )


predictor = DelayPredictor()
