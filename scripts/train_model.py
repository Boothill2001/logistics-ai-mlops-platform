"""Train and register two versions of the delay model.

v1 — LogisticRegression  -> registered as *production*
v2 — GradientBoosting    -> registered as *canary* (gets 10% traffic)

Both are evaluated on the same held-out split and their metrics stored in the
registry, so promotion decisions (v2 -> production) can be made from data.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import settings  # noqa: E402
from src.ml.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from src.ml.registry import ModelRegistry  # noqa: E402

MODEL_NAME = "shipment_delay_model"


def main() -> None:
    df = pd.read_csv(settings.data_dir / "shipments_train.csv")
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    # stratify keeps class balance identical in both splits
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "v1": ("production", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "v2": ("canary", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)),
    }

    registry = ModelRegistry()
    now = datetime.now(timezone.utc).isoformat()
    for version, (stage, model) in candidates.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        metrics = {
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "f1": round(float(f1_score(y_te, proba > 0.5)), 4),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
        registry.register(MODEL_NAME, version, model, stage, metrics, now)
        print(f"{MODEL_NAME} {version} [{stage}] {metrics}")


if __name__ == "__main__":
    main()
