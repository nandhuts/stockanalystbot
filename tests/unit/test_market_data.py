import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.exceptions import (
    InvalidTickerError,
    MarketDataServiceError,
)
from ai_stock_advisor.services.market_data.client import MarketDataClient


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Fixture returning a clean temporary path for cache files."""
    return tmp_path / "cache"


@pytest.fixture
def mock_ohlcv_data() -> pd.DataFrame:
    """Mock OHLCV DataFrame."""
    dates = pd.date_range(start="2026-07-01", periods=5, freq="D")
    data = {
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    }
    return pd.DataFrame(data, index=dates)


def test_fetch_ohlcv_cache_miss_and_hit(
    temp_cache_dir: Path,
    mock_ohlcv_data: pd.DataFrame,
) -> None:
    """Validate that cache miss triggers API call and cache hit reads local file."""
    client = MarketDataClient(cache_dir=temp_cache_dir)
    ticker = "RELIANCE.NS"

    with patch.object(client, "_download_history", return_value=mock_ohlcv_data) as mock_download:
        # First call: Cache miss. Direct API call should occur.
        df1 = client.fetch_ohlcv(ticker, period="1mo", interval="1d")
        
        mock_download.assert_called_once_with(ticker, "1mo", "1d")
        assert not df1.empty
        assert df1.loc[df1.index[0], "Close"] == 102.0
        
        # Verify cache file was written
        cache_file = temp_cache_dir / "reliance_ns_1mo_1d.csv"
        assert cache_file.exists()

        # Second call: Cache hit. No API call should occur.
        mock_download.reset_mock()
        df2 = client.fetch_ohlcv(ticker, period="1mo", interval="1d")
        
        mock_download.assert_not_called()
        assert len(df2) == len(mock_ohlcv_data)
        assert df2.loc[df2.index[0], "Close"] == 102.0


def test_fetch_ohlcv_cache_expired(
    temp_cache_dir: Path,
    mock_ohlcv_data: pd.DataFrame,
) -> None:
    """Validate cache TTL check correctly forces refresh when expired."""
    # Initialize client with 0-second cache TTL (forces immediate expiration)
    client = MarketDataClient(cache_dir=temp_cache_dir, cache_ttl_seconds=0)
    ticker = "INFY.NS"

    with patch.object(client, "_download_history", return_value=mock_ohlcv_data) as mock_download:
        # First call creates cache
        client.fetch_ohlcv(ticker, period="1mo", interval="1d")
        mock_download.assert_called_once()
        
        # Second call: expired cache should trigger another API download
        mock_download.reset_mock()
        time.sleep(0.01)  # small delta to ensure modification time is older
        client.fetch_ohlcv(ticker, period="1mo", interval="1d")
        mock_download.assert_called_once()


def test_fetch_ohlcv_invalid_ticker(temp_cache_dir: Path) -> None:
    """Ensure querying invalid stock returns custom InvalidTickerError."""
    client = MarketDataClient(cache_dir=temp_cache_dir)
    ticker = "INVALID_SYMBOL"

    # yfinance returns an empty DataFrame when ticker is invalid
    with patch.object(client, "_download_history", return_value=pd.DataFrame()) as mock_download:
        with pytest.raises(InvalidTickerError) as exc_info:
            client.fetch_ohlcv(ticker, period="1mo", interval="1d")
            
        assert exc_info.value.ticker == "INVALID_SYMBOL"
        assert "No stock data found" in exc_info.value.message
        mock_download.assert_called_once()


def test_fetch_ohlcv_network_failure_and_retries(temp_cache_dir: Path) -> None:
    """Verify client exhausts retries on API failure and raises MarketDataServiceError."""
    client = MarketDataClient(cache_dir=temp_cache_dir)
    ticker = "TCS.NS"

    # Mock Ticker.history to raise an error
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = Exception("Connection Refused")

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(MarketDataServiceError) as exc_info:
            # We mock time.sleep inside the client logic to speed up test execution
            with patch("time.sleep", return_value=None):
                client.fetch_ohlcv(ticker, period="1mo", interval="1d", force_refresh=True)

        assert "Failed executing" in exc_info.value.message
        assert mock_ticker.history.call_count == 3  # Max attempts is 3
