from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest

from ai_stock_advisor.ml.pipeline import StockMLPipeline


@pytest.fixture
def mock_pipeline_data() -> pd.DataFrame:
    """Generates 60 rows of synthetic OHLCV metrics."""
    dates = pd.date_range(start="2026-06-01", periods=60, freq="D")
    data = {
        "Open": [100.0] * 60,
        "High": [105.0] * 60,
        "Low": [95.0] * 60,
        "Close": [100.0 + i * 0.2 for i in range(60)],  # steadily increasing close prices
        "Volume": [10000] * 60,
        "EMA_20": [99.0] * 60,
        "EMA_50": [98.0] * 60,
        "EMA_200": [95.0] * 60,
        "RSI_14": [55.0] * 60,
        "MACD": [1.5] * 60,
        "MACD_Signal": [1.0] * 60,
        "MACD_Hist": [0.5] * 60,
        "ADX_14": [28.0] * 60,
        "Vol_MA20": [9000.0] * 60,
        "ATR_14": [2.5] * 60
    }
    return pd.DataFrame(data, index=dates)


def test_prepare_data(
    mock_pipeline_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Ensure prepare_data computes correct feature shapes and target classifications."""
    pipeline = StockMLPipeline(tickers=["TICK.NS"], horizon=5)
    pipeline.model_dir = tmp_path
    
    # Mock download return
    pipeline.market_client.fetch_ohlcv = MagicMock(return_value=mock_pipeline_data)
    pipeline.indicator_engine.compute_all_indicators = MagicMock(return_value=mock_pipeline_data)

    X, y = pipeline.prepare_data(save_scaler=True)
    
    # Features count: 8 features. Target horizon is 5, so we shift by 5 and drop.
    # Total rows: 40. Minus rolling window requirements / NaNs.
    # Scaler file should be created.
    assert (tmp_path / "scaler.joblib").exists()
    assert X.shape[1] == 8
    assert len(X) == len(y)
    assert set(np.unique(y)).issubset({0, 1})


def test_train_and_evaluate_scores(
    mock_pipeline_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify that cross-validation trains RF, XGB, and LGBM, saving model binaries."""
    pipeline = StockMLPipeline(tickers=["TICK.NS"], horizon=5)
    pipeline.model_dir = tmp_path
    
    # Mock queries
    pipeline.market_client.fetch_ohlcv = MagicMock(return_value=mock_pipeline_data)
    pipeline.indicator_engine.compute_all_indicators = MagicMock(return_value=mock_pipeline_data)

    # Trigger training
    report = pipeline.train_and_evaluate()
    
    # Check returned keys
    for model_name in ["Random_Forest", "XGBoost", "LightGBM"]:
        assert model_name in report
        metrics = report[model_name]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        
    # Check saved binaries
    assert (tmp_path / "random_forest_model.joblib").exists()
    assert (tmp_path / "xgboost_model.joblib").exists()
    assert (tmp_path / "lightgbm_model.joblib").exists()


def test_predict_probability(
    mock_pipeline_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Ensure predict_probability processes new daily feature rows and outputs percentage predictions."""
    pipeline = StockMLPipeline(tickers=["TICK.NS"], horizon=5)
    pipeline.model_dir = tmp_path
    
    # Mock data references
    pipeline.market_client.fetch_ohlcv = MagicMock(return_value=mock_pipeline_data)
    pipeline.indicator_engine.compute_all_indicators = MagicMock(return_value=mock_pipeline_data)

    # First run training to establish scaler and models
    pipeline.train_and_evaluate()

    # Predict
    prob = pipeline.predict_probability("TICK.NS")
    
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 100.0
