from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ai_stock_advisor.core.ranker import StockRanker
from ai_stock_advisor.core.scanner import StockScanner


@pytest.fixture
def mock_scanner() -> MagicMock:
    """Mock StockScanner instance."""
    return MagicMock(spec=StockScanner)


@pytest.fixture
def bullish_row() -> pd.Series:
    """Mock row containing fully bullish indicators."""
    return pd.Series(
        {
            "Ticker": "BULL.NS",
            "Above_EMA20": True,
            "Above_EMA50": True,
            "EMA_Crossover": True,
            "ADX": 30.0,
            "MACD_Bullish": True,
            "RSI": 55.0,  # inside 45-65 momentum sweet spot
            "Volume_Spike": True,
        }
    )


@pytest.fixture
def bearish_row() -> pd.Series:
    """Mock row containing fully bearish indicators."""
    return pd.Series(
        {
            "Ticker": "BEAR.NS",
            "Above_EMA20": False,
            "Above_EMA50": False,
            "EMA_Crossover": False,
            "ADX": 10.0,
            "MACD_Bullish": False,
            "RSI": 80.0,  # overbought/outside sweet spot
            "Volume_Spike": False,
        }
    )


def test_calculate_probability_score_bullish(
    mock_scanner: MagicMock,
    bullish_row: pd.Series,
) -> None:
    """Verify fully bullish stock triggers 100% probability rating."""
    ranker = StockRanker(scanner=mock_scanner)
    prob = ranker.calculate_probability_score(bullish_row)
    
    assert prob == 100.0


def test_calculate_probability_score_bearish(
    mock_scanner: MagicMock,
    bearish_row: pd.Series,
) -> None:
    """Verify fully bearish stock triggers 0% probability rating."""
    ranker = StockRanker(scanner=mock_scanner)
    prob = ranker.calculate_probability_score(bearish_row)
    
    assert prob == 0.0


def test_rank_stocks_limits_to_twenty_and_sorts(
    mock_scanner: MagicMock,
    tmp_path: Path,
) -> None:
    """Ensure ranker loads scan data, calculates scores, limits to Top 20, and saves output."""
    ranker = StockRanker(scanner=mock_scanner)
    
    # Generate 30 mock stock rows
    records = []
    for idx in range(30):
        records.append(
            {
                "Ticker": f"TICK{idx}.NS",
                "Above_EMA20": idx % 2 == 0,
                "Above_EMA50": idx % 3 == 0,
                "EMA_Crossover": idx % 4 == 0,
                "ADX": 10.0 + idx,
                "MACD_Bullish": idx % 2 == 0,
                "RSI": 40.0 + (idx % 30),
                "Volume_Spike": idx % 5 == 0,
                "Close": 100.0 + idx
            }
        )
    scan_df = pd.DataFrame(records)

    # Patch the scanner load function to return our custom dataframe
    with patch("pandas.read_csv", return_value=scan_df):
        with patch("pathlib.Path.exists", return_value=True):
            save_dir = tmp_path / "out"
            res_df = ranker.rank_stocks(save_dir=save_dir)

            # Assert return parameters
            assert isinstance(res_df, pd.DataFrame)
            assert len(res_df) == 20  # Limit is Top 20
            
            # Assert descending sorting
            scores = res_df["Probability_Score"].tolist()
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            
            # Assert file creations
            assert (save_dir / "rankings_results.csv").exists()
            assert (save_dir / "rankings_results.json").exists()
