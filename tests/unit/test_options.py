from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.options import OptionAnalyzer
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient


@pytest.fixture
def mock_market_client() -> MagicMock:
    """Mock MarketDataClient."""
    return MagicMock(spec=MarketDataClient)


@pytest.fixture
def indicator_engine() -> TechnicalIndicatorEngine:
    """Instance of TechnicalIndicatorEngine."""
    return TechnicalIndicatorEngine()


@pytest.fixture
def mock_options_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock Calls and Puts DataFrames."""
    calls_data = {
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "openInterest": [100, 200, 1500, 400, 100]
    }
    puts_data = {
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "openInterest": [50, 400, 800, 150, 50]
    }
    return pd.DataFrame(calls_data), pd.DataFrame(puts_data)


def test_resolve_ticker(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
) -> None:
    """Verify ticker resolving maps NIFTY strings to yfinance NSE symbols."""
    analyzer = OptionAnalyzer(mock_market_client, indicator_engine)
    
    assert analyzer.resolve_ticker("Nifty 50") == "^NSEI"
    assert analyzer.resolve_ticker("banknifty") == "^NSEBANK"
    assert analyzer.resolve_ticker("TCS.NS") == "TCS.NS"


def test_calculate_max_pain(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    mock_options_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Ensure Max Pain minimizes buyer losses correctly."""
    analyzer = OptionAnalyzer(mock_market_client, indicator_engine)
    calls, puts = mock_options_data
    
    # Calculate Max Pain
    max_pain = analyzer.calculate_max_pain(calls, puts)
    
    # Check that Max Pain maps to a strike in the list
    assert max_pain in [90.0, 95.0, 100.0, 105.0, 110.0]
    # For this synthetic dataset, the highest open interest is at 100 Call and 100 Put, which usually gravitates max pain towards 100
    assert max_pain == 100.0


def test_analyze_options_data_calculations(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    mock_options_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Ensure option analysis calculates correct PCR, sentiment suggestion, and zones."""
    analyzer = OptionAnalyzer(mock_market_client, indicator_engine)
    calls, puts = mock_options_data
    
    # Mock return values for fetch_option_chain
    # Spot price is 101.5 (making 100 the ATM strike)
    mock_fetch = MagicMock(return_value=(calls, puts, 101.5))
    
    # Mock history returns for ATR calculations
    dates = pd.date_range(start="2026-07-01", periods=20, freq="D")
    hist_data = {
        "Open": [100.0] * 20,
        "High": [103.0] * 20,
        "Low": [98.0] * 20,
        "Close": [101.5] * 20,
        "Volume": [1000] * 20
    }
    mock_hist_df = pd.DataFrame(hist_data, index=dates)
    mock_market_client.fetch_ohlcv.return_value = mock_hist_df

    with patch.object(analyzer, "fetch_option_chain", mock_fetch):
        res = analyzer.analyze_options("NIFTY")
        
        assert res["Spot_Price"] == 101.5
        assert res["ATM_Strike"] == 100.0
        
        # PCR = total puts OI (1450) / total calls OI (2300) = 0.63
        assert res["PCR"] == 0.63
        assert res["Sentiment"] == "BEARISH"  # PCR < 0.75
        assert res["Suggestion"] == "BUY PUT"
        
        # Check Highest OI details
        assert res["Highest_OI_Call"]["strike"] == 100.0
        assert res["Highest_OI_Call"]["oi"] == 1500
        assert res["Highest_OI_Put"]["strike"] == 100.0
        assert res["Highest_OI_Put"]["oi"] == 800
        
        # Check volatility adjusted stop loss and target for Bearish Suggestion
        # High-Low is 5.0, so ATR is around 5.0
        assert res["ATR"] > 0.0
        assert res["Target"] < 101.5  # Bearish target is below spot
        assert res["Stop_Loss"] > 101.5  # Bearish stop loss is above spot
