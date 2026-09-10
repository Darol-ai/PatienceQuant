from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PatienceQuant"
    api_prefix: str = "/api"
    database_url: str = "sqlite:///./data/patience_quant.db"
    data_mode: str = "demo"
    demo_seed: int = 20260909
    demo_as_of: date = Field(default_factory=date.today)
    # The offline universe is intentionally large enough to exercise realistic
    # cross-sectional ranking without requiring a network connection.  It can
    # be lowered for constrained CI machines via DEMO_UNIVERSE_SIZE.
    demo_universe_size: int = Field(default=1000, ge=50, le=2000)
    auto_rebalance_poll_seconds: int = Field(default=60, ge=10, le=3600)
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("demo_as_of", mode="before")
    @classmethod
    def empty_demo_date_uses_today(cls, value):
        return date.today() if value in (None, "") else value

    @property
    def cors_origin_list(self) -> List[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def ensure_data_dir(self) -> None:
        if self.database_url.startswith("sqlite:///./"):
            path = Path(self.database_url.removeprefix("sqlite:///./"))
            path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_data_dir()
    return settings
