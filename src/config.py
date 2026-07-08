"""Central configuration. Everything tunable lives here, loaded from env / .env.

Production note: in a real deployment these would come from a secret manager
(Vault, AWS SSM) — .env is a local-dev convenience only.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # LLM
    llm_provider: str = "mock"  # "mock" | "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Canary
    canary_fraction: float = 0.10

    # Paths
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Risk thresholds on delay probability
    risk_medium: float = 0.40
    risk_high: float = 0.70

    # Drift (PSI) thresholds
    psi_warning: float = 0.20
    psi_alert: float = 0.25


settings = Settings()
