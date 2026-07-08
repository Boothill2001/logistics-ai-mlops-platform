"""Nightly batch scoring job.

Reads a shipment CSV, scores every row with the *production* model
(batch never uses canary — canary is an online-serving experiment; mixing it
into batch would make nightly outputs non-comparable day over day),
writes predictions CSV, and enforces an input-rows == output-rows invariant.

Also runs drift detection: baseline (training data) vs tonight's batch.

Usage:
    python scripts/run_batch_scoring.py [--input data/shipments_batch.csv]
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import settings  # noqa: E402
from src.ml.drift import detect_drift  # noqa: E402
from src.ml.features import FEATURE_COLUMNS  # noqa: E402
from src.ml.predictor import MODEL_NAME, risk_level  # noqa: E402
from src.ml.registry import registry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("batch")


def score(input_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(input_path)
    n_input = len(df)

    loaded = registry.get_by_stage(MODEL_NAME, "production")
    if loaded is None:
        raise RuntimeError("No production model registered — run scripts/train_model.py first")

    proba = loaded.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    out = df[["shipment_id", "customer_id"]].copy()
    out["delay_probability"] = proba.round(4)
    out["risk_level"] = [risk_level(p) for p in proba]
    out["model_name"] = loaded.name
    out["model_version"] = loaded.version
    out["scored_at"] = datetime.now(timezone.utc).isoformat()

    # Invariant: every input row must be scored. A silent row drop in batch
    # is how shipments "disappear" from ops dashboards.
    if len(out) != n_input:
        raise RuntimeError(f"Row count mismatch: input={n_input} output={len(out)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_latest.csv"
    out.to_csv(output_path, index=False)

    logger.info(json.dumps({
        "job": "batch_scoring", "model_version": loaded.version,
        "rows_in": n_input, "rows_out": len(out),
        "high_risk": int((out["risk_level"] == "high").sum()),
        "output": str(output_path),
    }))

    # Drift check against training baseline
    baseline = pd.read_csv(settings.data_dir / "baseline_stats.csv")
    for result in detect_drift(baseline, df):
        logger.info(json.dumps({
            "job": "drift_check", "feature": result.feature,
            "psi": result.psi, "status": result.status,
        }))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.data_dir / "shipments_batch.csv"))
    args = parser.parse_args()
    score(Path(args.input), settings.data_dir / "batch_output")
