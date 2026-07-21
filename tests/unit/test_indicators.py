import numpy as np
import pandas as pd
import pytest

from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generates synthetic stock data for consistent mathematical testing."""
    dates = pd.date_range(start="2026-01-01", periods=250, freq="D")
    
    # Base price of 100 with small trending rise and sine wave cycle
    t = np.arange(250)
    trend = 0.05 * t
    cycle = 5 * np.sin(2 * np.pi * t / 20)
    close = 100.0 + trend + cycle
    
    high = close + 1.5
    low = close - 1.5
    open_p = close + np.random.uniform(-0.5, 0.5, 250)
    volume = np.random.randint(5000, 15000, 250)

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


def test_column_validation() -> None:
    """Verify engine raises ValueError if any required column is missing."""
    malformed_df = pd.DataFrame({"Close": [10, 20, 30]})
    with pytest.raises(ValueError) as exc_info:
        TechnicalIndicatorEngine.compute_all_indicators(malformed_df)
    
    assert "missing required columns" in str(exc_info.value)


def test_compute_all_returns_new_dataframe_with_correct_columns(
    synthetic_ohlcv: pd.DataFrame
) -> None:
    """Ensure compute_all_indicators appends all columns and doesn't mutate original."""
    raw_df_copy = synthetic_ohlcv.copy()
    res_df = TechnicalIndicatorEngine.compute_all_indicators(synthetic_ohlcv)

    # Check original remains unmutated
    pd.testing.assert_frame_equal(synthetic_ohlcv, raw_df_copy)

    # Verify column count and specific names
    expected_new_columns = {
        "EMA_20", "EMA_50", "EMA_200",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower",
        "ATR_14",
        "VWAP",
        "ADX_14"
    }
    
    assert expected_new_columns.issubset(res_df.columns)
    assert len(res_df) == len(synthetic_ohlcv)


def test_ema_calculations(synthetic_ohlcv: pd.DataFrame) -> None:
    """Validate basic properties of the Exponential Moving Average calculations."""
    ema_20 = TechnicalIndicatorEngine.calculate_ema(synthetic_ohlcv, "Close", 20)
    
    assert len(ema_20) == len(synthetic_ohlcv)
    # EMA should smoothen the curve compared to Close
    assert ema_20.std() < synthetic_ohlcv["Close"].std()


def test_rsi_bounds(synthetic_ohlcv: pd.DataFrame) -> None:
    """Ensure RSI values are mathematically constrained within [0, 100]."""
    rsi = TechnicalIndicatorEngine.calculate_rsi(synthetic_ohlcv, 14)
    
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0


def test_bollinger_bands_relations(synthetic_ohlcv: pd.DataFrame) -> None:
    """Verify mathematical relations of Bollinger Bands: Upper > Middle > Lower."""
    bb_df = TechnicalIndicatorEngine.calculate_bollinger_bands(synthetic_ohlcv, 20, 2.0)
    
    # Bollinger Bands have NaN for initial 19 items
    valid_bb = bb_df.dropna()
    assert len(valid_bb) > 0
    assert (valid_bb["BB_Upper"] > valid_bb["BB_Middle"]).all()
    assert (valid_bb["BB_Middle"] > valid_bb["BB_Lower"]).all()


def test_atr_positive(synthetic_ohlcv: pd.DataFrame) -> None:
    """Ensure Average True Range calculations are strictly positive."""
    atr = TechnicalIndicatorEngine.calculate_atr(synthetic_ohlcv, 14)
    
    # High-Low is always 3.0 in synthetic_ohlcv, so TR should be >= 3.0
    assert (atr.dropna() >= 3.0).all()


def test_vwap_calculations(synthetic_ohlcv: pd.DataFrame) -> None:
    """Validate basic properties of cumulative VWAP calculations."""
    vwap = TechnicalIndicatorEngine.calculate_vwap(synthetic_ohlcv)
    
    assert len(vwap) == len(synthetic_ohlcv)
    # Cumulative VWAP should stay in range of Close prices
    assert vwap.min() >= synthetic_ohlcv["Close"].min() - 5
    assert vwap.max() <= synthetic_ohlcv["Close"].max() + 5


def test_adx_bounds(synthetic_ohlcv: pd.DataFrame) -> None:
    """Verify ADX values are bound within [0, 100]."""
    adx = TechnicalIndicatorEngine.calculate_adx(synthetic_ohlcv, 14)
    
    assert adx.min() >= 0.0
    assert adx.max() <= 100.0
