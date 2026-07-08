"""File-based model registry.

Demo stand-in for MLflow/SageMaker Model Registry. The contract is the same:
models are immutable versioned artifacts; a *stage* pointer (production/canary)
decides which version serves traffic; rollback = move the pointer, never retrain.

registry.json shape:
{
  "shipment_delay_model": {
    "versions": {
      "v1": {"path": "shipment_delay_model_v1.joblib", "stage": "production",
              "metrics": {...}, "trained_at": "..."},
      "v2": {"path": "...", "stage": "canary", ...}
    }
  }
}
"""
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.config import settings

REGISTRY_FILE = "registry.json"


@dataclass
class LoadedModel:
    name: str
    version: str
    stage: str
    model: Any


class ModelRegistry:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or settings.models_dir)
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], LoadedModel] = {}

    # -- registry file ---------------------------------------------------
    def _registry_path(self) -> Path:
        return self.models_dir / REGISTRY_FILE

    def _read(self) -> dict:
        path = self._registry_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self._registry_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- write API (used by training script) ------------------------------
    def register(self, name: str, version: str, model: Any, stage: str,
                 metrics: dict, trained_at: str) -> None:
        with self._lock:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            artifact = f"{name}_{version}.joblib"
            joblib.dump(model, self.models_dir / artifact)
            data = self._read()
            entry = data.setdefault(name, {"versions": {}})
            entry["versions"][version] = {
                "path": artifact,
                "stage": stage,
                "metrics": metrics,
                "trained_at": trained_at,
            }
            self._write(data)

    def set_stage(self, name: str, version: str, stage: str) -> None:
        """Promote/demote a version. Rollback is just set_stage(v1, 'production')."""
        with self._lock:
            data = self._read()
            versions = data[name]["versions"]
            if version not in versions:
                raise KeyError(f"{name}/{version} not registered")
            if stage == "production":  # only one production version at a time
                for v, meta in versions.items():
                    if meta["stage"] == "production":
                        meta["stage"] = "archived"
            versions[version]["stage"] = stage
            self._write(data)

    # -- read API (used by serving) ---------------------------------------
    def get_by_stage(self, name: str, stage: str) -> LoadedModel | None:
        data = self._read()
        versions = data.get(name, {}).get("versions", {})
        for version, meta in versions.items():
            if meta["stage"] == stage:
                return self._load(name, version, meta)
        return None

    def get_version(self, name: str, version: str) -> LoadedModel:
        meta = self._read()[name]["versions"][version]
        return self._load(name, version, meta)

    def list_versions(self, name: str) -> dict:
        return self._read().get(name, {}).get("versions", {})

    def _load(self, name: str, version: str, meta: dict) -> LoadedModel:
        key = (name, version)
        if key not in self._cache:
            model = joblib.load(self.models_dir / meta["path"])
            self._cache[key] = LoadedModel(name, version, meta["stage"], model)
        cached = self._cache[key]
        cached.stage = meta["stage"]  # stage may change without reload
        return cached


registry = ModelRegistry()
