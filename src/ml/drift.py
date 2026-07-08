"""Data drift detection with PSI (Population Stability Index).

PSI compares the binned distribution of a feature in a *baseline* dataset
(what the model was trained on) against a *current* dataset (last night's
batch). Rule of thumb used across the industry:
  PSI < 0.10        no significant change
  0.10 - 0.20       moderate change, watch
  > 0.20            significant shift -> investigate / consider retraining

Why PSI over KS-test: PSI is threshold-friendly, insensitive to sample size
inflation (KS p-values go to zero on large N even for tiny shifts), and is
what most model-monitoring vendors implement — so it demos the real workflow.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import settings
from src.ml.features import DRIFT_FEATURES


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    status: str  # "ok" | "warning" | "alert"


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples, binned on baseline quantiles."""
    # Quantile bins on the baseline so every bin has baseline mass
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)  # collapse duplicate quantiles (low-cardinality features)

    base_pct = np.histogram(baseline, bins=edges)[0] / len(baseline)
    curr_pct = np.histogram(current, bins=edges)[0] / len(current)

    # Avoid log(0) — standard epsilon substitution
    eps = 1e-4
    base_pct = np.clip(base_pct, eps, None)
    curr_pct = np.clip(curr_pct, eps, None)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                 features: list[str] | None = None) -> list[FeatureDrift]:
    results = []
    for feature in features or DRIFT_FEATURES:
        value = psi(baseline_df[feature].to_numpy(), current_df[feature].to_numpy())
        if value >= settings.psi_alert:
            status = "alert"
        elif value >= settings.psi_warning:
            status = "warning"
        else:
            status = "ok"
        results.append(FeatureDrift(feature=feature, psi=round(value, 4), status=status))
    return results
