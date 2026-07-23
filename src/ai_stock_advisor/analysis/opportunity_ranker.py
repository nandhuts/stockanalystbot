"""
Opportunity Ranker Module.
Synthesizes technical, volume, options, news, and risk indicators to rank top opportunities.
"""
import logging
from typing import Any, Dict, List
import pandas as pd

logger = logging.getLogger("ai_stock_advisor.analysis.opportunity_ranker")


class OpportunityRanker:
    """
    Ranks derivative-grade securities using a multi-factor weighted scoring model.
    Generates exact entry, stop loss, target, position size, and option structure suggestions.
    """

    def __init__(self) -> None:
        """Initializes the ranker."""
        pass

    def calculate_scores(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        options_data: Dict[str, Any],
        news_data: Dict[str, Any],
        risk_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes sub-component scores and returns final opportunity rating.
        Formula weights: Trend (20%), VWAP (10%), EMA (10%), Momentum (15%),
        Volume (15%), News (15%), Options (15%), Risk (10%).
        """
        # Determine baseline direction (Bullish or Bearish)
        # We classify setup direction primarily by current Price vs EMA50, Supertrend, or PCR.
        close = indicators.get("Close", 0.0)
        ema50 = indicators.get("EMA_50", 0.0)
        pcr = options_data.get("PCR_OI", 1.0)
        supertrend_dir = indicators.get("Supertrend_Direction", 1)

        is_bullish = True
        if pcr <= 0.80 or (close < ema50 and supertrend_dir == -1):
            is_bullish = False

        # 1. Trend Score (20% weight)
        trend_score = 0.0
        if is_bullish:
            if supertrend_dir == 1:
                trend_score += 50
            if indicators.get("EMA_20", 0.0) > indicators.get("EMA_50", 0.0):
                trend_score += 30
            if close > indicators.get("EMA_200", 0.0):
                trend_score += 20
        else:
            if supertrend_dir == -1:
                trend_score += 50
            if indicators.get("EMA_20", 0.0) < indicators.get("EMA_50", 0.0):
                trend_score += 30
            if close < indicators.get("EMA_200", 0.0):
                trend_score += 20

        # 2. VWAP Score (10% weight)
        vwap = indicators.get("VWAP", close)
        vwap_score = 100.0 if (is_bullish and close > vwap) or (not is_bullish and close < vwap) else 40.0

        # 3. EMA Score (10% weight)
        ema20 = indicators.get("EMA_20", close)
        ema_score = 100.0 if (is_bullish and close > ema20) or (not is_bullish and close < ema20) else 40.0

        # 4. Momentum Score (15% weight)
        rsi = indicators.get("RSI_14", 50.0)
        macd_hist = indicators.get("MACD_Hist", 0.0)
        adx = indicators.get("ADX_14", 20.0)

        momentum_score = 0.0
        # RSI component (max 40)
        if is_bullish:
            if rsi > 50:
                momentum_score += 25
            if 55 <= rsi <= 70:
                momentum_score += 15
        else:
            if rsi < 50:
                momentum_score += 25
            if 30 <= rsi <= 45:
                momentum_score += 15

        # MACD component (max 30)
        if (is_bullish and macd_hist > 0) or (not is_bullish and macd_hist < 0):
            momentum_score += 30

        # ADX trend strength component (max 30)
        if adx > 25:
            momentum_score += 30
        elif adx > 15:
            momentum_score += 15

        # 5. Volume Score (15% weight)
        volume_ratio = indicators.get("Volume_Ratio", 1.0)
        if volume_ratio >= 2.0:
            volume_score = 100.0
        elif volume_ratio >= 1.5:
            volume_score = 80.0
        elif volume_ratio >= 1.0:
            volume_score = 60.0
        else:
            volume_score = 40.0

        # 6. News Sentiment Score (15% weight)
        news_score = news_data.get("Score", 50)

        # 7. Option Chain Score (15% weight)
        option_sentiment = options_data.get("Sentiment", "NEUTRAL")
        if (is_bullish and option_sentiment == "BULLISH") or (not is_bullish and option_sentiment == "BEARISH"):
            option_score = 100.0
        elif option_sentiment == "NEUTRAL":
            option_score = 60.0
        else:
            option_score = 30.0

        # 8. Risk Score (10% weight) - inverse of risk total
        risk_score = (1.0 - risk_profile.get("Total_Risk_Score", 0.5)) * 100.0

        # Final Weighted Score Calculation
        final_score = float(
            (trend_score * 0.20)
            + (vwap_score * 0.10)
            + (ema_score * 0.10)
            + (momentum_score * 0.15)
            + (volume_score * 0.15)
            + (news_score * 0.15)
            + (option_score * 0.15)
            + (risk_score * 0.10)
        )

        return {
            "Final_Score": round(final_score, 2),
            "Direction": "BULLISH" if is_bullish else "BEARISH",
            "SubScores": {
                "Trend": round(trend_score, 2),
                "VWAP": round(vwap_score, 2),
                "EMA": round(ema_score, 2),
                "Momentum": round(momentum_score, 2),
                "Volume": round(volume_score, 2),
                "News": round(news_score, 2),
                "Options": round(option_score, 2),
                "Risk": round(risk_score, 2),
            }
        }

    def compile_recommendation(
        self,
        ticker: str,
        close: float,
        direction: str,
        atr: float,
        final_score: float,
        risk_profile: Dict[str, Any],
        options_data: Dict[str, Any],
        news_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Creates detailed trade setup structures (stops, targets, sizing, options contracts).
        """
        # Calculate dynamic ATR targets/stops
        if direction == "BULLISH":
            entry = close
            stop_loss = close - (1.5 * atr)
            target1 = close + (1.0 * atr)
            target2 = close + (1.5 * atr)
            target3 = close + (2.0 * atr)
            suggested_opt_type = "CALL"
            premium = options_data.get("Call_Premium", 0.0)
        else:
            entry = close
            stop_loss = close + (1.5 * atr)
            target1 = close - (1.0 * atr)
            target2 = close - (1.5 * atr)
            target3 = close - (2.0 * atr)
            suggested_opt_type = "PUT"
            premium = options_data.get("Put_Premium", 0.0)

        # Risk reward based on Target 1
        rr_ratio = abs(target1 - entry) / abs(entry - stop_loss) if abs(entry - stop_loss) > 0 else 1.0

        # Suggested position size based on Risk Level
        risk_level = risk_profile.get("Risk_Level", "Medium")
        sizing_map = {
            "Very Low": "5% of capital",
            "Low": "4% of capital",
            "Medium": "3% of capital",
            "High": "2% of capital",
            "Very High": "1% of capital",
        }
        pos_size = sizing_map.get(risk_level, "3% of capital")

        # Expiry and Strike details
        strike = options_data.get("ATM_Strike", close)
        expiry = "Nearest Month Expiry"
        premium_range = f"₹{premium * 0.9:.2f} - ₹{premium * 1.1:.2f}"

        # Confidence Score calculation (scale score and news alignment)
        confidence = float(final_score * 0.8 + (100 - risk_profile.get("Total_Risk_Score", 0.5)*100)*0.2)

        # AI Explanation generator
        catalysts = []
        if direction == "BULLISH":
            catalysts.append("Strong technical trend with positive price crossovers")
            if options_data.get("PCR_OI", 1.0) >= 1.2:
                catalysts.append("Put writing concentration provides solid underlying option support")
        else:
            catalysts.append("Weakening technical structure below major moving averages")
            if options_data.get("PCR_OI", 1.0) <= 0.8:
                catalysts.append("Call writing concentration acts as immediate overhead resistance")

        if news_data.get("Sentiment") == direction:
            catalysts.append("Overnight news sentiment aligns with current breakout setup direction")
        
        explanation = (
            f"Secured an Opportunity Score of {final_score:.1f}% backed by a {risk_level} risk outlook. "
            f"Primary catalysts: {', '.join(catalysts)}. "
            f"Recommend buying the {strike} {suggested_opt_type} option contract."
        )

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Trend": direction,
            "Entry": round(entry, 2),
            "Stop_Loss": round(stop_loss, 2),
            "Target_1": round(target1, 2),
            "Target_2": round(target2, 2),
            "Target_3": round(target3, 2),
            "Risk_Reward": round(rr_ratio, 2),
            "Probability": round(final_score, 1),
            "Confidence": round(confidence, 1),
            "Risk_Level": risk_level,
            "Position_Size": pos_size,
            "Option_Type": suggested_opt_type,
            "Strike": strike,
            "Expiry": expiry,
            "Premium_Range": premium_range,
            "Explanation": explanation,
        }

    def rank_opportunities(self, candidate_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts candidates by final opportunity score and returns the Top 10 setups.
        """
        sorted_list = sorted(candidate_list, key=lambda x: x["Probability"], reverse=True)
        return sorted_list[:10]
