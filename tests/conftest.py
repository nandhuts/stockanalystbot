import os
import sys
from pathlib import Path
import pytest

# Ensure the 'src' directory is in Python path for test execution
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

# Also ensure 'config' is in Python path
config_path = Path(__file__).resolve().parent.parent
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically mock environment variables for test safety.
    Prevents tests from modifying real files or hitting production endpoints.
    """
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_123")
    monkeypatch.setenv("MARKET_DATA_API_KEY", "test_market_key_123")
