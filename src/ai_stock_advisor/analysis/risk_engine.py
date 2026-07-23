"""
Risk Evaluation Engine.
Evaluates volatility, gap risk, news sentiment, and option IV to assign dynamic risk profiles.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("ai_stock_advisor.analysis.risk_engine")


class RiskEngine:
    """
    Computes numerical and categorical risk indicators for stock trading setups.
    Classifies assets from 'Very Low' to 'Very High' risk.
    """

    @staticmethod
    def calculate_atr_risk(atr: float, close: float) -> float:
        """Calculates volatility relative to price (ATR / Close ratio)."""
        if close <= 0:
            return 0.0
        return float(atr / close)

    @staticmethod
    def calculate_gap_risk(df: pd.DataFrame) -> float:
        """Calculates historical standard deviation of overnight open-to-prev-close gaps."""
        if len(df) < 5:
            return 0.0
        close_prev = df["Close"].shift(1)
        gaps = (df["Open"] - close_prev) / close_prev.replace(0, np.nan)
        return float(gaps.std())

    def assess_risk(
        self,
        df: pd.DataFrame,
        atr: float,
        iv: float,
        news_sentiment: str,
        news_score: int,
    ) -> Dict[str, Any]:
        """
        Synthesizes indicators into a consolidated Risk Profile.
        Returns numerical component scores and a categorical Risk Level classification.
        """
        close = float(df["Close"].iloc[-1]) if not df.empty else 1.0
        
        # 1. ATR Risk (relative daily trading range)
        atr_ratio = self.calculate_atr_risk(atr, close)
        # Normalized score: average ATR ratio is around 2-3%, so scale it:
        atr_score = min(1.0, atr_ratio / 0.05)  # Max out at 5% daily range

        # 2. Gap Risk (overnight gaps)
        gap_std = self.calculate_gap_risk(df)
        gap_score = min(1.0, gap_std / 0.02)  # Max out at 2% standard deviation

        # 3. News Risk (catalyst volatility)
        if news_sentiment == "NEGATIVE":
            news_risk_val = 0.8
        elif news_sentiment == "POSITIVE":
            news_risk_val = 0.3
        else:
            news_risk_val = 0.5
        # Refine based on exact news score
        news_risk_val += (50 - news_score) / 250.0
        news_risk = min(1.0, max(0.0, news_risk_val))

        # 4. Option Volatility Risk (Implied Volatility)
        # Standard average Nifty IV is ~15-20%. Let's scale:
        option_risk = min(1.0, iv / 0.50)  # Max out at 50% IV

        # 5. Volatility Risk (historical standard deviation of returns)
        if len(df) >= 10:
            returns = df["Close"].pct_change()
            hist_vol = float(returns.std())
            vol_score = min(1.0, hist_vol / 0.03)  # Max out at 3% daily volatility
        else:
            vol_score = 0.5

        # Weighted cumulative risk score (0.0 to 1.0)
        # Weights: ATR (25%), Gap (20%), News (15%), Options (20%), Volatility (20%)
        total_risk_score = float(
            (atr_score * 0.25)
            + (gap_score * 0.20)
            + (news_risk * 0.15)
            + (option_risk * 0.20)
            + (vol_score * 0.20)
        )

        # Categorize
        if total_risk_score < 0.25:
            level = "Very Low"
        elif total_risk_score < 0.40:
            level = "Low"
        elif total_risk_score < 0.60:
            level = "Medium"
        elif total_risk_score < 0.75:
            level = "High"
        else:
            level = "Very High"

        return {
            "Total_Risk_Score": round(total_risk_score, 4),
            "Risk_Level": level,
            "ATR_Risk": round(atr_score, 2),
            "Gap_Risk": round(gap_score, 2),
            "News_Risk": round(news_risk, 2),
            "Option_Risk": round(option_risk, 2),
            "Volatility_Risk": round(vol_score, 2),
        }
