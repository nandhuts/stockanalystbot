import os
from config.settings import Settings


def test_settings_load_from_env() -> None:
    """Validate that Settings parses custom env variables correctly."""
    # Instantiating Settings with overrides
    custom_settings = Settings(
        APP_ENV="staging",
        LOG_LEVEL="DEBUG",
        GEMINI_API_KEY="test_gemini",
        OPENAI_API_KEY="test_openai",
        MARKET_DATA_API_KEY="test_md"
    )
    
    assert custom_settings.APP_ENV == "staging"
    assert custom_settings.LOG_LEVEL == "DEBUG"
    assert custom_settings.GEMINI_API_KEY == "test_gemini"
    assert custom_settings.OPENAI_API_KEY == "test_openai"
    assert custom_settings.MARKET_DATA_API_KEY == "test_md"
