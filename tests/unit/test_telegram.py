from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.telegram.bot import (
    generate_morning_report_text,
    handle_market_summary,
    handle_top_stocks,
)


@pytest.fixture
def mock_scan_results() -> pd.DataFrame:
    """Mock scan results DataFrame."""
    return pd.DataFrame([
        {"Ticker": "INFY.NS", "Score": 85.0, "Close": 1500.0},
        {"Ticker": "TCS.NS", "Score": 50.0, "Close": 3800.0},
        {"Ticker": "RELIANCE.NS", "Score": 20.0, "Close": 2400.0}
    ])


@pytest.fixture
def mock_rankings_results() -> pd.DataFrame:
    """Mock rankings results DataFrame."""
    return pd.DataFrame([
        {
            "Ticker": "INFY.NS",
            "Probability_Score": 85.0,
            "Close": 1500.0,
            "Above_EMA20": True,
            "EMA_Crossover": True,
            "Volume_Spike": False,
            "RSI": 55.0,
            "ADX": 30.0
        },
        {
            "Ticker": "TCS.NS",
            "Probability_Score": 60.0,
            "Close": 3800.0,
            "Above_EMA20": True,
            "EMA_Crossover": False,
            "Volume_Spike": True,
            "RSI": 62.0,
            "ADX": 22.0
        }
    ])


def test_generate_morning_report_text(
    mock_rankings_results: pd.DataFrame
) -> None:
    """Ensure pre-market morning report generation parses rankings and outputs correct markdown summary."""
    with patch("pandas.read_csv", return_value=mock_rankings_results):
        with patch("pathlib.Path.exists", return_value=True):
            report = generate_morning_report_text()
            
            assert "PRE-MARKET AI STOCK ADVISOR REPORT" in report
            assert "Total Checked: 2" in report
            # INFY.NS probability is 85% which is >= 75% (Strong Bullish: 1)
            assert "Strong Bullish (≥ 75%): 1" in report
            # TCS.NS probability is 60% which is 50-74% (Moderate Bullish: 1)
            assert "Moderate Bullish (50-74%): 1" in report
            assert "INFY.NS" in report
            assert "TCS.NS" in report


def test_handle_market_summary_command(
    mock_scan_results: pd.DataFrame
) -> None:
    """Verify `/market` command formats scan distribution stats correctly."""
    mock_message = MagicMock()
    mock_message.text = "/market"
    
    with patch("pandas.read_csv", return_value=mock_scan_results):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ai_stock_advisor.telegram.bot.bot.reply_to") as mock_reply:
                handle_market_summary(mock_message)
                
                mock_reply.assert_called_once()
                reply_text = mock_reply.call_args[0][1]
                assert "Market Scan Summary" in reply_text
                # total scanned should be 3
                assert "Total Securities: 3" in reply_text
                # 1 bullish (Score >= 70, INFY)
                assert "Bullish (Score ≥ 70): 1" in reply_text
                # 1 bearish (Score <= 30, RELIANCE)
                assert "Bearish (Score ≤ 30): 1" in reply_text


def test_handle_top_stocks_command(
    mock_rankings_results: pd.DataFrame
) -> None:
    """Verify `/topstocks` command formats ranked list correctly."""
    mock_message = MagicMock()
    mock_message.text = "/topstocks"
    
    with patch("pandas.read_csv", return_value=mock_rankings_results):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ai_stock_advisor.telegram.bot.bot.reply_to") as mock_reply:
                handle_top_stocks(mock_message)
                
                mock_reply.assert_called_once()
                reply_text = mock_reply.call_args[0][1]
                assert "Top 10 Bullish Stock Rankings" in reply_text
                assert "INFY.NS" in reply_text
                assert "TCS.NS" in reply_text


def test_handle_opportunities_command() -> None:
    """Verify `/opportunities` command loads pre-market json and formats correctly."""
    from ai_stock_advisor.telegram.bot import handle_opportunities
    from unittest.mock import mock_open
    import json
    
    mock_message = MagicMock()
    mock_message.text = "/opportunities"
    mock_message.chat.id = 12345
    
    mock_json_data = [
        {
            "Ticker": "INFY.NS",
            "Price": 1500.0,
            "Trend": "BULLISH",
            "Entry": 1500.0,
            "Stop_Loss": 1450.0,
            "Target_1": 1550.0,
            "Target_2": 1600.0,
            "Target_3": 1650.0,
            "Risk_Reward": 2.0,
            "Probability": 85.0,
            "Confidence": 90.0,
            "Risk_Level": "Low",
            "Position_Size": "4% of capital",
            "Option_Type": "CALL",
            "Strike": 1500.0,
            "Expiry": "26-Jul-2026",
            "Premium_Range": "₹30 - ₹35",
            "Explanation": "Breakout setup"
        }
    ]

    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
            with patch("ai_stock_advisor.telegram.bot.bot.send_chat_action") as mock_action:
                with patch("ai_stock_advisor.telegram.bot.bot.reply_to") as mock_reply:
                    handle_opportunities(mock_message)
                    mock_reply.assert_called_once()
                    reply_text = mock_reply.call_args[0][1]
                    assert "TOP 10 F&O TRADING OPPORTUNITIES" in reply_text
                    assert "INFY.NS" in reply_text
                    assert "CALL" in reply_text


def test_handle_option_recommendation_command() -> None:
    """Verify `/option` command runs OptionAnalyzer and outputs target strikes."""
    from ai_stock_advisor.telegram.bot import handle_option_recommendation
    
    mock_message = MagicMock()
    mock_message.text = "/option INFY.NS"
    mock_message.chat.id = 12345
    
    mock_report = {
        "Ticker": "INFY.NS",
        "Spot_Price": 1500.0,
        "ATM_Strike": 1500.0,
        "PCR": 1.35,
        "Max_Pain": 1500.0,
        "Sentiment": "BULLISH",
        "Suggestion": "BUY CALL",
        "Suggested_Strike": 1500.0,
        "Target": 1550.0,
        "Stop_Loss": 1450.0,
        "ATR": 25.0,
        "Highest_OI_Call": {"strike": 1520.0, "oi": 50000},
        "Highest_OI_Put": {"strike": 1480.0, "oi": 40000}
    }

    with patch("ai_stock_advisor.telegram.bot.OptionAnalyzer.analyze_options", return_value=mock_report):
        with patch("ai_stock_advisor.telegram.bot.bot.send_chat_action") as mock_action:
            with patch("ai_stock_advisor.telegram.bot.bot.reply_to") as mock_reply:
                handle_option_recommendation(mock_message)
                mock_reply.assert_called_once()
                reply_text = mock_reply.call_args[0][1]
                assert "Option Analysis: INFY.NS" in reply_text
                assert "Derivative Outlook" in reply_text
                assert "BULLISH" in reply_text
                assert "Max Pain Strike" in reply_text
