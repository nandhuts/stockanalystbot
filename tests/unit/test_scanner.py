from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.scanner import StockScanner
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
def bullish_ohlcv() -> pd.DataFrame:
    """Generates a DataFrame set with fully bullish properties (consistently rising close & volume)."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    
    # Consistently rising trend to trigger EMA crossovers and ADX trend
    close = 100.0 + (2.0 * np.arange(100))  # 100 to 298
    high = close + 1.0
    low = close - 1.0
    open_p = close - 0.5
    # Force a massive volume spike on the final day (20000 compared to average 1000)
    volume = [1000] * 99 + [20000]

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def bearish_ohlcv() -> pd.DataFrame:
    """Generates a DataFrame set with fully bearish properties (consistently falling close & flat volume)."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    
    # Consistently falling trend
    close = 300.0 - (2.0 * np.arange(100))  # 300 to 102
    high = close + 1.0
    low = close - 1.0
    open_p = close + 0.5
    volume = [1000] * 100

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def test_score_stock_bullish(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    bullish_ohlcv: pd.DataFrame,
) -> None:
    """Verify that a stock with bullish alignment gets a very high score (e.g. 100)."""
    scanner = StockScanner(
        market_client=mock_market_client,
        indicator_engine=indicator_engine,
    )
    
    res = scanner.score_stock(bullish_ohlcv)
    
    assert res["Score"] == 100.0  # Should trigger all 7 conditions
    assert res["Above_EMA20"] is True
    assert res["Above_EMA50"] is True
    assert res["EMA_Crossover"] is True
    assert res["MACD_Bullish"] is True
    assert res["Volume_Spike"] is True
    assert res["ADX"] > 25.0


def test_score_stock_bearish(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    bearish_ohlcv: pd.DataFrame,
) -> None:
    """Verify that a stock with bearish alignment gets a very low score (e.g. 0)."""
    scanner = StockScanner(
        market_client=mock_market_client,
        indicator_engine=indicator_engine,
    )
    
    res = scanner.score_stock(bearish_ohlcv)
    
    # Should get 0.0 since close is below EMAs, EMA20 < EMA50, MACD is bearish, Volume is flat, and trend is downwards (RSI will be low)
    assert res["Score"] == 0.0
    assert res["Above_EMA20"] is False
    assert res["Above_EMA50"] is False
    assert res["EMA_Crossover"] is False
    assert res["MACD_Bullish"] is False
    assert res["Volume_Spike"] is False


def test_scan_orchestration_and_sorting(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    bullish_ohlcv: pd.DataFrame,
    bearish_ohlcv: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Ensure scanner pulls data, scores, sorts descending, and saves CSV/JSON."""
    scanner = StockScanner(
        market_client=mock_market_client,
        indicator_engine=indicator_engine,
    )
    
    # Mock market client responses
    mock_market_client.fetch_ohlcv.side_effect = lambda ticker, *args, **kwargs: (
        bullish_ohlcv if ticker == "BULL.NS" else bearish_ohlcv
    )
    
    save_dir = tmp_path / "data_out"
    
    tickers = ["BEAR.NS", "BULL.NS"]
    
    res_df = scanner.scan(tickers=tickers, save_dir=save_dir)
    
    # Assert return types and sorting
    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) == 2
    
    # BULL.NS should have a higher score and appear first (index 0)
    assert res_df.loc[0, "Ticker"] == "BULL.NS"
    assert res_df.loc[0, "Score"] == 100.0
    
    assert res_df.loc[1, "Ticker"] == "BEAR.NS"
    assert res_df.loc[1, "Score"] == 0.0

    # Check files were written
    csv_file = save_dir / "scan_results.csv"
    json_file = save_dir / "scan_results.json"
    
    assert csv_file.exists()
    assert json_file.exists()
