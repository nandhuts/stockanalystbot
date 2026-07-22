from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.backtester import BacktestingEngine
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient


@pytest.fixture
def mock_market_client() -> MagicMock:
    """Mock MarketDataClient."""
    return MagicMock(spec=MarketDataClient)


@pytest.fixture
def indicator_engine() -> TechnicalIndicatorEngine:
    """TechnicalIndicatorEngine instance."""
    return TechnicalIndicatorEngine()


@pytest.fixture
def mock_backtest_data() -> pd.DataFrame:
    """Generates 60 rows of daily price history simulating entering and exiting a trade."""
    dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
    data = {
        "Open": [100.0] * 60,
        "High": [102.0] * 60,
        "Low": [98.0] * 60,
        "Close": [100.0] * 60,
        "Volume": [1000] * 60,
        "EMA_20": [98.0] * 60,
        "EMA_50": [97.0] * 60,
        "RSI_14": [40.0] * 60,
        "MACD": [1.5] * 60,
        "MACD_Signal": [1.0] * 60,
        "ATR_14": [2.0] * 60
    }
    df = pd.DataFrame(data, index=dates)
    
    # Induce an entry trigger on index 10 (Close increases, RSI surges, MACD increases)
    df.loc[df.index[10]:, "Open"] = 105.0
    df.loc[df.index[10]:, "High"] = 106.0
    df.loc[df.index[10]:, "Low"] = 104.0
    df.loc[df.index[10]:, "Close"] = 105.0
    df.loc[df.index[10]:, "EMA_20"] = 100.0
    df.loc[df.index[10]:, "EMA_50"] = 99.0
    df.loc[df.index[10]:, "RSI_14"] = 60.0
    df.loc[df.index[10]:, "MACD"] = 2.0
    df.loc[df.index[10]:, "MACD_Signal"] = 1.0
    
    # Induce a Target Hit on index 20 (High surges to 110.0, target is entry 105.0 + 1.5 * 2.0 = 108.0)
    df.loc[df.index[20], "High"] = 110.0
    
    return df


def test_run_backtest_logic_and_metrics(
    mock_market_client: MagicMock,
    indicator_engine: TechnicalIndicatorEngine,
    mock_backtest_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Ensure backtester runs sequential ticks, registers buy/sell events, and exports correct metrics."""
    backtester = BacktestingEngine(mock_market_client, indicator_engine)
    
    # Mock data download returns
    mock_market_client.fetch_ohlcv.return_value = mock_backtest_data
    indicator_engine.compute_all_indicators = MagicMock(return_value=mock_backtest_data)

    results = backtester.run_backtest("TEST.NS", initial_capital=100000.0, save_dir=tmp_path)
    
    # Verify outputs
    assert results["Ticker"] == "TEST.NS"
    assert results["Initial_Capital"] == 100000.0
    assert results["Total_Trades"] >= 1
    assert "Win_Rate_Pct" in results
    assert "Profit_Factor" in results
    assert "Max_Drawdown_Pct" in results
    assert "Average_Return_Pct" in results
    
    # Check that individual trades are logged
    assert len(results["Trades"]) >= 1
    trade = results["Trades"][0]
    assert trade["Exit_Reason"] == "Target"
    assert trade["Return_Pct"] > 0.0
    
    # Check saved output files
    assert (tmp_path / "backtest_trades.csv").exists()
    assert (tmp_path / "backtest_metrics.json").exists()
