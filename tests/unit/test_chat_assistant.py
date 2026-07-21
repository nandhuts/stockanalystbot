from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.exceptions import LLMServiceError
from ai_stock_advisor.services.llm.chat_assistant import AIChatAssistantClient


@pytest.fixture
def mock_indicators_df() -> pd.DataFrame:
    """Generates 25 rows of daily price logs."""
    dates = pd.date_range(start="2026-07-01", periods=25, freq="D")
    data = {
        "Open": [100.0] * 25,
        "High": [105.0] * 25,
        "Low": [95.0] * 25,
        "Close": [101.5] * 25,
        "Volume": [1000] * 25,
        "EMA_20": [100.0] * 25,
        "EMA_50": [99.0] * 25,
        "RSI_14": [55.0] * 25,
        "MACD": [1.5] * 25,
        "MACD_Signal": [1.0] * 25,
        "ATR_14": [2.5] * 25
    }
    return pd.DataFrame(data, index=dates)


def test_extract_ticker_calls_openai(mock_indicators_df: pd.DataFrame) -> None:
    """Ensure extract_ticker queries OpenAI and returns clean string."""
    client = AIChatAssistantClient(api_key="test_key_123")
    
    mock_choice = MagicMock()
    mock_choice.message.content = "RELIANCE.NS"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch.object(client.client.chat.completions, "create", return_value=mock_completion) as mock_create:
        ticker = client.extract_ticker("Should I buy Reliance stock?")
        
        assert ticker == "RELIANCE.NS"
        mock_create.assert_called_once()


def test_compile_context_packaging(
    mock_indicators_df: pd.DataFrame
) -> None:
    """Verify compile_context gathers technical indicators, support/resistance, options and news metadata."""
    client = AIChatAssistantClient(api_key="test_key_123")
    
    # Mock data downloaders
    with patch("ai_stock_advisor.services.market_data.client.MarketDataClient.fetch_ohlcv", return_value=mock_indicators_df):
        with patch("ai_stock_advisor.core.indicators.TechnicalIndicatorEngine.compute_all_indicators", return_value=mock_indicators_df):
            # Bypass options and news analyzer for clean unit focus
            with patch("ai_stock_advisor.core.options.OptionAnalyzer.analyze_options", side_effect=Exception("Bypass Options")):
                with patch("ai_stock_advisor.services.llm.news_analyzer.NewsAnalyzer.analyze_news", side_effect=Exception("Bypass News")):
                    
                    context = client.compile_context("RELIANCE.NS")
                    
                    assert context["Ticker"] == "RELIANCE.NS"
                    assert context["Spot_Price"] == 101.5
                    # Support is low (95.0), Resistance is high (105.0)
                    assert context["Support"] == 95.0
                    assert context["Resistance"] == 105.0
                    assert context["Option_Chain"] == {}
                    assert context["News_Sentiment"] == {}


def test_ask_advisor_returns_analysis_report() -> None:
    """Verify ask_advisor queries OpenAI and handles response text."""
    client = AIChatAssistantClient(api_key="test_key_123")
    
    mock_choice = MagicMock()
    mock_choice.message.content = "### 📊 RELIANCE.NS AI Investment Advice\n* **Trend**: Bullish alignment..."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_context = {"Ticker": "RELIANCE.NS", "Spot_Price": 2400.0}

    with patch.object(client.client.chat.completions, "create", return_value=mock_completion):
        response = client.ask_advisor("Should I buy Reliance?", mock_context)
        
        assert "### 📊 RELIANCE.NS AI" in response
        assert "Trend" in response
