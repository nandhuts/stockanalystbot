"""
AI Stock Ranking Engine.
Applies quantitative weights over technical indicators to calculate probability scores.
Selects and ranks the top 20 stock profiles.
"""
import logging
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.scanner import StockScanner

logger = logging.getLogger("ai_stock_advisor.core.ranker")


class StockRanker:
    """
    Ranks stock scanner results using a weighted multi-factor scoring model.
    Computes a bullish trend probability score (0% to 100%) and extracts top performers.
    """

    def __init__(self, scanner: StockScanner) -> None:
        """Initializes the ranker with an existing scanner instance."""
        self.scanner = scanner

    def calculate_probability_score(self, row: pd.Series) -> float:
        """
        Calculates the probability of trend continuation or breakout on a 0-100% scale.
        
        Factors:
          - Trend Sub-Score (35% weight):
            - Close > EMA20 (+30 pts)
            - Close > EMA50 (+30 pts)
            - EMA20 > EMA50 (+20 pts)
            - ADX > 25 (+20 pts)
          - Momentum Sub-Score (35% weight):
            - MACD > MACD_Signal (+50 pts)
            - 45 <= RSI <= 65 (+50 pts)  (Bullish trend momentum zone)
          - Volume Sub-Score (30% weight):
            - Volume Spike Active (+100 pts)
        """
        # 1. Trend Sub-Score (Max 100)
        trend = 0.0
        if bool(row.get("Above_EMA20", False)):
            trend += 30.0
        if bool(row.get("Above_EMA50", False)):
            trend += 30.0
        if bool(row.get("EMA_Crossover", False)):
            trend += 20.0
        if float(row.get("ADX", 0.0)) > 25.0:
            trend += 20.0

        # 2. Momentum Sub-Score (Max 100)
        momentum = 0.0
        if bool(row.get("MACD_Bullish", False)):
            momentum += 50.0
        
        rsi_val = float(row.get("RSI", 50.0))
        if 45.0 <= rsi_val <= 65.0:
            momentum += 50.0

        # 3. Volume Sub-Score (Max 100)
        volume = 0.0
        if bool(row.get("Volume_Spike", False)):
            volume += 100.0

        # Weighted calculation
        probability = (trend * 0.35) + (momentum * 0.35) + (volume * 0.30)
        return float(round(probability, 1))

    def rank_stocks(
        self,
        save_dir: Path | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Ranks stocks based on the calculated probability score.
        Pulls scanner data (generating new scan if missing) and selects the Top 20 stocks.
        Saves output in CSV and JSON formats.
        """
        out_dir = save_dir or Path(settings.BASE_DIR) / "data"
        scan_file = out_dir / "scan_results.csv"

        # Check for scanner file or run fresh
        if force_refresh or not scan_file.exists():
            logger.info("Scanner results missing or refresh requested. Executing scanner run...")
            scan_df = self.scanner.scan(force_refresh=force_refresh, save_dir=out_dir)
        else:
            try:
                scan_df = pd.read_csv(scan_file)
            except Exception as exc:
                logger.error("Error reading scan file: %s. Re-running scanner.", exc)
                scan_df = self.scanner.scan(force_refresh=True, save_dir=out_dir)

        if scan_df.empty:
            logger.warning("No data retrieved from scanner. Aborting rank process.")
            return pd.DataFrame()

        # Apply probability scoring
        logger.info("Computing probability scores over %d stocks...", len(scan_df))
        scan_df["Probability_Score"] = scan_df.apply(self.calculate_probability_score, axis=1)

        # Sort and filter Top 20
        ranked_df = scan_df.sort_values(
            by=["Probability_Score", "Ticker"], 
            ascending=[False, True]
        ).head(20).reset_index(drop=True)

        # File saves
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "rankings_results.csv"
        json_path = out_dir / "rankings_results.json"

        try:
            ranked_df.to_csv(csv_path, index=False)
            logger.info("Saved stock rankings CSV to %s", csv_path)
            
            ranked_df.to_json(json_path, orient="records", indent=2)
            logger.info("Saved stock rankings JSON to %s", json_path)
        except Exception as exc:
            logger.error("Error writing ranking files: %s", exc)

        return ranked_df
