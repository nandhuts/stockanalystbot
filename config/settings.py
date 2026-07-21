import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    Validates configuration types and provides safe fallbacks.
    """
    model_config = SettingsConfigDict(
        # Look for .env file at the workspace root
        env_file=os.path.join(str(BASE_DIR), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core settings
    APP_ENV: str = Field(default="development", description="Application runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Global log level output")
    BASE_DIR: Path = Field(default=BASE_DIR, description="Root path of the project")

    # LLM Settings
    LLM_PROVIDER: str = Field(default="gemini", description="Selected LLM client wrapper")
    GEMINI_API_KEY: str | None = Field(default=None, description="Gemini developer API key")
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")

    # Market Data settings
    MARKET_DATA_PROVIDER: str = Field(default="yfinance", description="Market data API vendor")
    MARKET_DATA_API_KEY: str | None = Field(default=None, description="Vendor specific API key")

    # App serving details
    APP_PORT: int = Field(default=8000, description="Web API serving port")
    APP_HOST: str = Field(default="127.0.0.1", description="Web API listening host")


# Instantiate settings singleton
settings = Settings()
