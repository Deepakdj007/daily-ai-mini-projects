import logging
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),  # look in backend/ then project root
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Retell AI
    RETELL_API_KEY: str
    RETELL_AGENT_ID: str = ""        # set by setup.py after agent creation
    RETELL_FROM_NUMBER: str = ""     # set by setup.py after number purchase
    RETELL_WEBHOOK_SECRET: str = ""

    # Sarvam AI
    SARVAM_API_KEY: str

    # App config
    APP_ENV: Literal["development", "production"] = "development"
    WEBHOOK_BASE_URL: str = "http://localhost:8000"
    DATABASE_URL: str = "voicebot.db"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
