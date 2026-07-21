from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.exceptions import LLMServiceError
from ai_stock_advisor.services.llm.client import (
    LLMAnalysisClient,
    RecommendationEnum,
    TechnicalAnalysisReport,
)


@pytest.fixture
def mock_indicators_df() -> pd.DataFrame:
    """Generates mock indicators row."""
    dates = pd.date_range(start="2026-07-01", periods=1, freq="D")
    data = {
        "Open": [150.0],
        "High": [155.0],
        "Low": [148.0],
        "Close": [152.0],
        "Volume": [10000],
        "EMA_20": [150.0],
        "EMA_50": [148.0],
        "EMA_200": [145.0],
        "RSI_14": [55.0],
        "MACD": [1.5],
        "MACD_Signal": [1.0],
        "MACD_Hist": [0.5],
        "ADX_14": [28.0],
        "Vol_MA20": [9000.0]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def expected_report() -> TechnicalAnalysisReport:
    """Mock report parsed output matching target schema."""
    return TechnicalAnalysisReport(
        trend_analysis="Price exhibits support above the EMA50 and is trending upwards.",
        rsi_analysis="RSI is neutral at 55, indicating healthy bullish momentum.",
        macd_analysis="MACD is above the signal line with positive histogram bars.",
        ema_analysis="EMA 20 lies above EMA 50, which is above EMA 200, validating a golden alignment.",
        recommendation=RecommendationEnum.BUY,
        confidence_percentage=85,
        investment_thesis="Strong technical setup across trend alignments and momentum indicators supports a long position."
    )


def test_generate_analysis_success(
    mock_indicators_df: pd.DataFrame,
    expected_report: TechnicalAnalysisReport,
) -> None:
    """Ensure generate_analysis parses successful OpenAI completions into target report schema."""
    client = LLMAnalysisClient(api_key="test_key_123")
    
    # Mock completion object response structure
    mock_parsed_choice = MagicMock()
    mock_parsed_choice.message.parsed = expected_report
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_parsed_choice]

    # Patch OpenAI completions parse method
    with patch.object(client.client.beta.chat.completions, "parse", return_value=mock_completion) as mock_parse:
        report = client.generate_analysis("RELIANCE.NS", mock_indicators_df, 85.0)
        
        # Verify structure
        assert isinstance(report, TechnicalAnalysisReport)
        assert report.recommendation == RecommendationEnum.BUY
        assert report.confidence_percentage == 85
        assert "golden alignment" in report.ema_analysis
        
        # Assert parameters sent to OpenAI
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[1]
        assert call_args["model"] == "gpt-4o-mini"
        assert len(call_args["messages"]) == 2
        assert "system" == call_args["messages"][0]["role"]


def test_generate_analysis_failure_raises_custom_exception(
    mock_indicators_df: pd.DataFrame,
) -> None:
    """Ensure connection exceptions are mapped to custom LLMServiceError."""
    client = LLMAnalysisClient(api_key="test_key_123")
    
    # Patch OpenAI completions to raise connection error
    with patch.object(
        client.client.beta.chat.completions, 
        "parse", 
        side_effect=Exception("API Key revoked or network down")
    ):
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate_analysis("RELIANCE.NS", mock_indicators_df, 85.0)
            
        assert "Failed generating AI analysis report" in exc_info.value.message
        assert exc_info.value.details["ticker"] == "RELIANCE.NS"
        assert exc_info.value.details["model"] == "gpt-4o-mini"
