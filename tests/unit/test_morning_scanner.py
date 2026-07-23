"""
Unit Tests for Morning F&O Opportunity Scanner.
Validates Supertrend, news sentiment mapping, option chain analysis, risk engines, rankers, and persistence.
"""
from datetime import datetime
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.analysis.news_sentiment import MorningNewsSentimentAnalyzer
from ai_stock_advisor.analysis.option_chain_analyzer import MorningOptionChainAnalyzer
from ai_stock_advisor.analysis.risk_engine import RiskEngine
from ai_stock_advisor.analysis.opportunity_ranker import OpportunityRanker
from ai_stock_advisor.scanner.morning_scanner import MorningOpportunityScanner
from ai_stock_advisor.db.database import DailyScan


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generates synthetic stock history DataFrame."""
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    data = {
        "Open": np.linspace(100, 130, 30),
        "High": np.linspace(102, 132, 30),
        "Low": np.linspace(99, 129, 30),
        "Close": np.linspace(101, 131, 30),
        "Volume": np.linspace(1000, 3000, 30),
    }
    return pd.DataFrame(data, index=dates)


def test_supertrend_calculation(sample_ohlcv):
    """Verifies Supertrend indicator values and directions."""
    df = TechnicalIndicatorEngine.calculate_supertrend(sample_ohlcv, period=10, multiplier=3.0)
    assert "Supertrend" in df.columns
    assert "Supertrend_Direction" in df.columns
    assert len(df) == len(sample_ohlcv)
    # Check directions are integer flags (+1 or -1)
    assert set(df["Supertrend_Direction"].unique()).issubset({1, -1})


def test_news_sentiment_score_mapping():
    """Validates mapping from polarity (-1.0 to +1.0) to (0 to 100) scores."""
    from ai_stock_advisor.services.llm.news_analyzer import SentimentEnum
    
    mock_report = MagicMock()
    mock_report.average_sentiment_score = 0.5  # Bullish
    mock_report.overall_sentiment = SentimentEnum.POSITIVE
    
    art1 = MagicMock()
    art1.headline = "Earnings Surge"
    art1.summary = "Profits grew 20%."
    art1.sentiment = SentimentEnum.POSITIVE
    
    mock_report.articles = [art1]

    mock_news_analyzer = MagicMock()
    mock_news_analyzer.analyze_news.return_value = mock_report

    analyzer = MorningNewsSentimentAnalyzer(news_analyzer=mock_news_analyzer)
    result = analyzer.analyze_overnight_news("TCS.NS")

    assert result["Score"] == 75  # (0.5 + 1.0) * 50 = 75
    assert result["Sentiment"] == "POSITIVE"
    assert "Positive News: Earnings Surge" in result["Reasons"][0]


def test_option_chain_sentiment():
    """Validates derivative PCR, Max Pain, and option sentiment logic."""
    from ai_stock_advisor.services.llm.news_analyzer import SentimentEnum
    
    mock_calls = pd.DataFrame({
        "strike": [100.0, 105.0, 110.0],
        "openInterest": [100, 500, 200],
        "volume": [10, 50, 20],
        "lastPrice": [5.0, 2.0, 0.5],
        "impliedVolatility": [0.22, 0.20, 0.18],
    })
    mock_puts = pd.DataFrame({
        "strike": [100.0, 105.0, 110.0],
        "openInterest": [200, 100, 50],
        "volume": [20, 10, 5],
        "lastPrice": [0.5, 1.5, 4.0],
        "impliedVolatility": [0.24, 0.21, 0.19],
    })

    mock_opt_analyzer = MagicMock()
    mock_opt_analyzer.fetch_option_chain.return_value = (mock_calls, mock_puts, 105.0)
    mock_opt_analyzer.calculate_max_pain.return_value = 105.0

    analyzer = MorningOptionChainAnalyzer(mock_opt_analyzer)
    result = analyzer.analyze_option_chain("INFY.NS")

    assert result["Spot_Price"] == 105.0
    assert result["ATM_Strike"] == 105.0
    assert result["Max_Pain"] == 105.0
    assert result["Total_Call_OI"] == 800
    assert result["Total_Put_OI"] == 350
    # PCR_OI = 350 / 800 = 0.4375 (round to 0.44)
    assert result["PCR_OI"] == 0.44
    assert result["Sentiment"] == "BEARISH"  # PCR <= 0.8 is Bearish


def test_risk_engine_classification(sample_ohlcv):
    """Verifies that risk profiles resolve to correct categorical classes."""
    engine = RiskEngine()
    
    # 1. Test Low Risk configuration
    result_low = engine.assess_risk(
        df=sample_ohlcv,
        atr=1.5,
        iv=0.12,  # Low IV
        news_sentiment="POSITIVE",
        news_score=80
    )
    assert result_low["Risk_Level"] in ("Very Low", "Low", "Medium")

    # 2. Test High Risk configuration
    # Let's create a dataframe with high gap risk and high volatility
    high_vol_data = {
        "Open": [100, 200, 150, 300, 100, 200, 150, 300, 100, 200],
        "High": [110, 210, 160, 310, 110, 210, 160, 310, 110, 210],
        "Low": [90, 190, 140, 290, 90, 190, 140, 290, 90, 190],
        "Close": [95, 195, 145, 295, 95, 195, 145, 295, 95, 195],
        "Volume": [1000] * 10,
    }
    df_high = pd.DataFrame(high_vol_data)
    result_high = engine.assess_risk(
        df=df_high,
        atr=50.0,  # High ATR
        iv=0.85,   # High IV
        news_sentiment="NEGATIVE",
        news_score=5
    )
    assert result_high["Risk_Level"] in ("High", "Very High")


def test_opportunity_ranker_scoring():
    """Validates weighted scoring and trade recommendation bounds."""
    ranker = OpportunityRanker()
    
    indicators = {
        "Close": 105.0,
        "EMA_20": 108.0,
        "EMA_50": 106.0,
        "EMA_200": 100.0,
        "VWAP": 104.0,
        "RSI_14": 58.0,
        "MACD_Hist": 0.5,
        "ADX_14": 28.0,
        "Volume_Ratio": 2.2,
        "Supertrend_Direction": 1
    }
    options_data = {
        "PCR_OI": 1.35,
        "Sentiment": "BULLISH",
        "ATM_Strike": 105.0,
        "Call_Premium": 3.5,
        "Put_Premium": 1.2
    }
    news_data = {
        "Score": 85,
        "Sentiment": "BULLISH"
    }
    risk_profile = {
        "Total_Risk_Score": 0.20,  # Very Low Risk
        "Risk_Level": "Very Low"
    }

    # Verify score calculation
    scores = ranker.calculate_scores("RELIANCE.NS", indicators, options_data, news_data, risk_profile)
    assert scores["Direction"] == "BULLISH"
    assert 0 <= scores["Final_Score"] <= 100

    # Verify trade compiling
    rec = ranker.compile_recommendation(
        "RELIANCE.NS", 105.0, "BULLISH", 3.0, scores["Final_Score"], risk_profile, options_data, news_data
    )
    assert rec["Ticker"] == "RELIANCE.NS"
    assert rec["Trend"] == "BULLISH"
    assert rec["Stop_Loss"] == 105.0 - (1.5 * 3.0)  # 100.5
    assert rec["Target_1"] == 105.0 + 3.0          # 108.0
    assert rec["Position_Size"] == "5% of capital"  # Very Low risk gets 5%


@patch("ai_stock_advisor.scanner.morning_scanner.SessionLocal")
def test_morning_scanner_db_persistence(mock_session_factory, sample_ohlcv):
    """Verifies that the pre-market scanner successfully serializes and saves results."""
    mock_db = MagicMock()
    mock_session_factory.return_value = mock_db

    # Instantiate scanner with mocked dependencies
    mock_service = MagicMock()
    mock_service.fetch_fo_tickers.return_value = ["INFY.NS"]
    mock_service.fetch_all_candles.return_value = {
        "INFY.NS": {
            "daily": sample_ohlcv,
            "15m": sample_ohlcv,
            "5m": sample_ohlcv
        }
    }

    mock_news_analyzer = MagicMock()
    mock_news_analyzer.analyze_overnight_news.return_value = {
        "Score": 80,
        "Sentiment": "BULLISH",
        "Summary": "Good news summary",
        "Reasons": ["Reason 1"]
    }

    mock_opt_analyzer = MagicMock()
    mock_opt_analyzer.analyze_option_chain.return_value = {
        "Spot_Price": 105.0,
        "ATM_Strike": 105.0,
        "PCR_OI": 1.35,
        "PCR_Vol": 1.20,
        "Max_Pain": 105.0,
        "IV": 0.22,
        "Total_Call_OI": 800,
        "Total_Put_OI": 1100,
        "Sentiment": "BULLISH",
        "Explanation": "Bullish options"
    }

    scanner = MorningOpportunityScanner(
        scanner_service=mock_service,
        db_session_factory=mock_session_factory
    )
    scanner.news_sentiment_analyzer = mock_news_analyzer
    scanner.option_chain_analyzer = mock_opt_analyzer

    # Run scan
    results = scanner.run_scan()

    # Verify results and DB interactions
    assert len(results) == 1
    assert results[0]["Ticker"] == "INFY.NS"
    assert mock_db.add.called
    assert mock_db.commit.called
    
    # Check that added object is a DailyScan and is serialized correctly
    added_obj = mock_db.add.call_args[0][0]
    assert isinstance(added_obj, DailyScan)
    assert added_obj.ticker == "INFY.NS"
    assert json.loads(added_obj.indicators)["Close"] > 0
    assert json.loads(added_obj.options)["Sentiment"] == "BULLISH"
