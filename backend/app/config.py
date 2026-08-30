"""Runtime configuration, read from environment / backend/.env (see .env.example)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / "backend" / ".env", env_prefix="SENTINEL_", extra="ignore"
    )

    # storage
    db_url: str = f"sqlite:///{REPO_ROOT / 'backend' / 'sentinel.db'}"

    # model artifacts
    model_file: Path = REPO_ROOT / "models" / "baseline_lgb.txt"
    feature_spec: Path = REPO_ROOT / "models" / "baseline_lgb_feature_spec.json"
    metrics_file: Path = REPO_ROOT / "report" / "metrics.json"
    test_split: Path = REPO_ROOT / "data" / "splits" / "test.parquet"

    # replay engine: how many held-out days pass per real second
    replay_days_per_sec: float = 2.0
    replay_autostart: bool = True

    # ring engine
    ring_cap: int = 25

    # Razorpay (test mode). Integration endpoints 503 until key_id is set.
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    @property
    def razorpay_ready(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)


settings = Settings()
