"""
Machine Learning Pipeline.
Prepares datasets, performs 5-fold cross-validation on Random Forest, XGBoost,
and LightGBM, and serializes trained models for prediction.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient

logger = logging.getLogger("ai_stock_advisor.ml.pipeline")

# Selection of Nifty 50 tickers to train on
DEFAULT_TRAIN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "SBIN.NS"
]


class StockMLPipeline:
    """
    ML training and prediction pipeline.
    Calculates features, targets, performs Stratified K-Fold validation,
    and handles joblib saves/loads.
    """

    def __init__(
        self,
        tickers: List[str] | None = None,
        horizon: int = 5,
    ) -> None:
        """Initializes pipeline parameters."""
        self.tickers = tickers or DEFAULT_TRAIN_TICKERS
        self.horizon = horizon
        self.market_client = MarketDataClient()
        self.indicator_engine = TechnicalIndicatorEngine()
        
        self.model_dir = Path(settings.BASE_DIR) / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_cols = [
            "RSI_14",
            "EMA_Ratio_20",
            "EMA_Ratio_50",
            "EMA_Crossover_Ratio",
            "MACD_Hist",
            "ADX_14",
            "ATR_Pct",
            "Volume_Ratio"
        ]

    def _extract_features(self, df_ind: pd.DataFrame) -> pd.DataFrame:
        """Helper to compute features relative ratios for model ingestion."""
        df = df_ind.copy()
        
        df["EMA_Ratio_20"] = df["Close"] / df["EMA_20"] - 1.0
        df["EMA_Ratio_50"] = df["Close"] / df["EMA_50"] - 1.0
        df["EMA_Crossover_Ratio"] = df["EMA_20"] / df["EMA_50"] - 1.0
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        df["ATR_Pct"] = df["ATR_14"] / df["Close"]
        
        # Vol_MA20 check safety
        vol_ma = df.get("Vol_MA20", df["Volume"].rolling(20).mean())
        df["Volume_Ratio"] = df["Volume"] / vol_ma
        
        # Select target features only
        return df[self.feature_cols]

    def prepare_data(self, save_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downloads daily history for configured training tickers.
        Calculates features, shifts target (1 if Close rises in horizon days, 0 otherwise).
        Standardizes feature matrices and fits/saves the StandardScaler.
        """
        all_X = []
        all_y = []

        for ticker in self.tickers:
            logger.info("ML Pipeline: Preparing training data for ticker '%s'...", ticker)
            try:
                # Fetch 2 years of daily data
                hist_df = self.market_client.fetch_ohlcv(ticker, period="2y", interval="1d")
                if hist_df.empty or len(hist_df) < 50:
                    continue
                    
                df_ind = self.indicator_engine.compute_all_indicators(hist_df)
                
                # Feature calculation
                features = self._extract_features(df_ind)
                
                # Target: Upward movement 5 days in the future
                target = (df_ind["Close"].shift(-self.horizon) > df_ind["Close"]).astype(int)
                
                # Combine
                combined = pd.concat([features, target], axis=1)
                combined.columns = self.feature_cols + ["Target"]
                combined = combined.dropna()
                
                if not combined.empty:
                    all_X.append(combined[self.feature_cols].values)
                    all_y.append(combined["Target"].values)
            except Exception as exc:
                logger.warning("ML Pipeline: Failed loading data for %s: %s", ticker, exc)

        if not all_X:
            raise ValueError("No valid training samples could be prepared. Aborting.")

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if save_scaler:
            scaler_path = self.model_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info("ML Pipeline: Saved StandardScaler to %s", scaler_path)

        return X_scaled, y

    def train_and_evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Trains models and evaluates them using 5-fold cross-validation.
        Fits final models on the entire dataset and serializes them.
        Returns the evaluation report.
        """
        X, y = self.prepare_data(save_scaler=True)
        
        # Instantiate classifiers
        models = {
            "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbosity=-1)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        evaluation_results: Dict[str, Dict[str, List[float]]] = {
            name: {"precision": [], "recall": [], "f1": [], "auc": []} for name in models
        }

        # 5-fold cross validation split loop
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, model in models.items():
                # Clone model for training
                from sklearn.base import clone
                trained_fold_model = clone(model)
                trained_fold_model.fit(X_train, y_train)
                
                # Predictions
                preds = trained_fold_model.predict(X_test)
                probs = trained_fold_model.predict_proba(X_test)[:, 1]
                
                # Metrics
                evaluation_results[name]["precision"].append(precision_score(y_test, preds, zero_division=0))
                evaluation_results[name]["recall"].append(recall_score(y_test, preds, zero_division=0))
                evaluation_results[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
                evaluation_results[name]["auc"].append(roc_auc_score(y_test, probs))

        # Compile consolidated averages report
        report: Dict[str, Dict[str, float]] = {}
        for name in models:
            report[name] = {
                "precision": float(np.mean(evaluation_results[name]["precision"])),
                "recall": float(np.mean(evaluation_results[name]["recall"])),
                "f1": float(np.mean(evaluation_results[name]["f1"])),
                "auc": float(np.mean(evaluation_results[name]["auc"]))
            }
            logger.info(
                "ML Pipeline: %s CV Metrics -> Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f",
                name, report[name]["precision"], report[name]["recall"], report[name]["f1"], report[name]["auc"]
            )

        # Fit final models on the entire dataset and save
        for name, model in models.items():
            logger.info("ML Pipeline: Fitting and saving final model for %s...", name)
            model.fit(X, y)
            model_path = self.model_dir / f"{name.lower()}_model.joblib"
            joblib.dump(model, model_path)
            logger.info("ML Pipeline: Saved final model to %s", model_path)

        # Save CV metrics report for dashboard rendering reference
        try:
            report_df = pd.DataFrame(report).T
            report_df.to_csv(self.model_dir / "cv_metrics.csv")
        except Exception as exc:
            logger.error("Failed saving cv_metrics report: %s", exc)

        return report

    def predict_probability(self, ticker: str) -> float:
        """
        Predicts the probability of upward movement over the horizon (0% to 100%)
        using the ensemble average of the trained Random Forest, XGBoost, and LightGBM models.
        """
        scaler_path = self.model_dir / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError("StandardScaler missing. Run pipeline training first.")

        # Load scaler and models
        scaler = joblib.load(scaler_path)
        ensemble = []
        for name in ["random_forest", "xgboost", "lightgbm"]:
            m_path = self.model_dir / f"{name}_model.joblib"
            if m_path.exists():
                ensemble.append(joblib.load(m_path))

        if not ensemble:
            raise FileNotFoundError("Trained models are missing. Run pipeline training first.")

        # Fetch latest daily ticker data
        hist_df = self.market_client.fetch_ohlcv(ticker, period="1mo", interval="1d")
        if hist_df.empty or len(hist_df) < 20:
            raise ValueError(f"Insufficient history to predict for ticker: {ticker}")

        df_ind = self.indicator_engine.compute_all_indicators(hist_df)
        features = self._extract_features(df_ind)
        
        # Latest row feature vector
        latest_vector = features.iloc[[-1]].values
        latest_scaled = scaler.transform(latest_vector)

        # Average probabilities across the models
        probs = []
        for model in ensemble:
            prob = model.predict_proba(latest_scaled)[0, 1]
            probs.append(prob)

        mean_prob = float(np.mean(probs)) * 100.0
        return float(round(mean_prob, 1))
