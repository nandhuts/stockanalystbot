from unittest.mock import MagicMock, patch
import pytest

from ai_stock_advisor.core.exceptions import LLMServiceError
from ai_stock_advisor.services.llm.option_advisor import (
    AIOptionAdvisorClient,
    OptionSentimentEnum,
    OptionTradeSuggestion,
)


@pytest.fixture
def mock_technicals() -> dict:
    """Mock technical indicator scanner dictionary."""
    return {
        "Close": 1500.0,
        "Score": 85.0,
        "Above_EMA20": True,
        "Above_EMA50": True,
        "EMA_Crossover": True,
        "RSI": 58.4,
        "MACD_Bullish": True,
        "Volume_Spike": True,
        "ADX": 28.5,
    }


@pytest.fixture
def mock_options_data() -> dict:
    """Mock options analysis data dictionary."""
    return {
        "Spot_Price": 1500.0,
        "ATM_Strike": 1500.0,
        "PCR": 1.35,
        "Max_Pain": 1490.0,
        "Highest_OI_Call": {"strike": 1550.0, "oi": 25000},
        "Highest_OI_Put": {"strike": 1450.0, "oi": 35000},
        "ITM_Calls_Count": 5,
        "OTM_Calls_Count": 15,
        "ITM_Puts_Count": 4,
        "OTM_Puts_Count": 16,
        "ATR": 15.0,
    }


@pytest.fixture
def expected_suggestion() -> OptionTradeSuggestion:
    """Mock advisor suggestion output matching target schema."""
    return OptionTradeSuggestion(
        suggested_strategy="Bull Call Spread",
        strike_price=1500.0,
        target_price=1530.0,
        stop_loss=1485.0,
        probability_score=82.0,
        sentiment=OptionSentimentEnum.BULLISH,
        thesis_reasoning="Technicals indicate trend breakout with PCR > 1.3 validating solid put writing support at 1450 strike."
    )


def test_generate_option_trade_success(
    mock_technicals: dict,
    mock_options_data: dict,
    expected_suggestion: OptionTradeSuggestion,
) -> None:
    """Verify generate_option_trade parses successful OpenAI completions into option suggestion schema."""
    client = AIOptionAdvisorClient(api_key="test_key_123")
    
    # Mock completion object response structure
    mock_parsed_choice = MagicMock()
    mock_parsed_choice.message.parsed = expected_suggestion
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_parsed_choice]

    # Patch OpenAI completions parse method
    with patch.object(client.client.beta.chat.completions, "parse", return_value=mock_completion) as mock_parse:
        suggestion = client.generate_option_trade("INFY.NS", mock_technicals, mock_options_data)
        
        # Verify structure
        assert isinstance(suggestion, OptionTradeSuggestion)
        assert suggestion.sentiment == OptionSentimentEnum.BULLISH
        assert suggestion.probability_score == 82.0
        assert suggestion.suggested_strategy == "Bull Call Spread"
        
        # Assert parameters sent to OpenAI
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[1]
        assert call_args["model"] == "gpt-4o-mini"
        assert len(call_args["messages"]) == 2


def test_generate_option_trade_failure_raises_custom_exception(
    mock_technicals: dict,
    mock_options_data: dict,
) -> None:
    """Ensure connection exceptions are mapped to custom LLMServiceError."""
    client = AIOptionAdvisorClient(api_key="test_key_123")
    
    # Patch OpenAI completions to raise connection error
    with patch.object(
        client.client.beta.chat.completions, 
        "parse", 
        side_effect=Exception("API connection timeout")
    ):
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate_option_trade("INFY.NS", mock_technicals, mock_options_data)
            
        assert "Failed generating AI options trade recommendation" in exc_info.value.message
        assert exc_info.value.details["ticker"] == "INFY.NS"
