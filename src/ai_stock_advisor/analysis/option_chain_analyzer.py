"""
Option Chain Analysis Module.
Processes option chain metrics (OI, PCR, IV, Max Pain) for pre-market opportunity scans.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from ai_stock_advisor.core.options import OptionAnalyzer

logger = logging.getLogger("ai_stock_advisor.analysis.option_chain_analyzer")


class MorningOptionChainAnalyzer:
    """
    Computes derivative market indicators for stock underlying assets,
    such as PCR, Max Pain, ATM Implied Volatilities, and option sentiment.
    """

    def __init__(self, option_analyzer: OptionAnalyzer) -> None:
        """Initializes the morning option chain analyzer with standard OptionAnalyzer."""
        self.option_analyzer = option_analyzer

    def analyze_option_chain(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches option chain contracts and extracts quantitative metrics.
        Returns a dictionary representing option sentiment (Bullish/Bearish/Neutral) and indicators.
        """
        logger.info("Running option chain analysis for ticker '%s'...", ticker)
        try:
            # Resolve Yahoo symbol and fetch contracts
            calls, puts, spot = self.option_analyzer.fetch_option_chain(ticker)
            
            if calls.empty and puts.empty:
                raise ValueError("No option contracts returned.")

            # Calculate ATM strike
            atm_strike = float(min(calls["strike"], key=lambda x: abs(x - spot)))

            # Calculate Open Interest metrics
            total_call_oi = int(calls["openInterest"].sum())
            total_put_oi = int(puts["openInterest"].sum())
            pcr_oi = float(total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

            # Calculate Volume metrics
            total_call_vol = int(calls["volume"].fillna(0).sum())
            total_put_vol = int(puts["volume"].fillna(0).sum())
            pcr_vol = float(total_put_vol / total_call_vol) if total_call_vol > 0 else 1.0

            # Calculate Max Pain
            max_pain = self.option_analyzer.calculate_max_pain(calls, puts)

            # Implied Volatility (IV) computation for ATM contracts
            atm_calls = calls[calls["strike"] == atm_strike]
            atm_puts = puts[puts["strike"] == atm_strike]
            
            call_iv = float(atm_calls["impliedVolatility"].iloc[0]) if not atm_calls.empty else 0.0
            put_iv = float(atm_puts["impliedVolatility"].iloc[0]) if not atm_puts.empty else 0.0
            avg_iv = (call_iv + put_iv) / 2.0 if (call_iv > 0 and put_iv > 0) else (call_iv or put_iv or 0.2)

            # Categorize strikes
            itm_calls_count = len(calls[calls["strike"] < spot])
            otm_calls_count = len(calls[calls["strike"] > spot])
            itm_puts_count = len(puts[puts["strike"] > spot])
            otm_puts_count = len(puts[puts["strike"] < spot])

            # Determine Option Sentiment
            # Standard guidelines:
            # PCR >= 1.25 is Bullish
            # PCR <= 0.75 is Bearish
            # Otherwise Neutral
            if pcr_oi >= 1.20:
                sentiment = "BULLISH"
                reason = "Heavy writing of puts indicates strong support levels."
            elif pcr_oi <= 0.80:
                sentiment = "BEARISH"
                reason = "Heavy writing of calls indicates strong overhead resistance."
            else:
                sentiment = "NEUTRAL"
                reason = "Balanced call/put writing activity."

            # Estimate ATM premium range
            call_premium = float(atm_calls["lastPrice"].iloc[0]) if not atm_calls.empty else (spot * 0.02)
            put_premium = float(atm_puts["lastPrice"].iloc[0]) if not atm_puts.empty else (spot * 0.02)

            return {
                "Spot_Price": spot,
                "ATM_Strike": atm_strike,
                "PCR_OI": round(pcr_oi, 2),
                "PCR_Vol": round(pcr_vol, 2),
                "Max_Pain": max_pain,
                "IV": round(avg_iv, 4),
                "Total_Call_OI": total_call_oi,
                "Total_Put_OI": total_put_oi,
                "Total_Call_Vol": total_call_vol,
                "Total_Put_Vol": total_put_vol,
                "Sentiment": sentiment,
                "Explanation": reason,
                "ITM_Calls": itm_calls_count,
                "OTM_Calls": otm_calls_count,
                "ITM_Puts": itm_puts_count,
                "OTM_Puts": otm_puts_count,
                "Call_Premium": round(call_premium, 2),
                "Put_Premium": round(put_premium, 2),
            }

        except Exception as exc:
            logger.warning("Options chain analysis failed for '%s': %s. Returning neutral default profile.", ticker, str(exc))
            # Fallback values
            return {
                "Spot_Price": 0.0,
                "ATM_Strike": 0.0,
                "PCR_OI": 1.0,
                "PCR_Vol": 1.0,
                "Max_Pain": 0.0,
                "IV": 0.20,
                "Total_Call_OI": 0,
                "Total_Put_OI": 0,
                "Total_Call_Vol": 0,
                "Total_Put_Vol": 0,
                "Sentiment": "NEUTRAL",
                "Explanation": "Options chain data was unavailable or index contracts are skipped.",
                "ITM_Calls": 0,
                "OTM_Calls": 0,
                "ITM_Puts": 0,
                "OTM_Puts": 0,
                "Call_Premium": 0.0,
                "Put_Premium": 0.0,
            }
