"""
Option Chain Analyzer Module.
Extracts NSE option chain details (via yfinance) and performs max pain, PCR,
and volatility-adjusted target calculations.
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from ai_stock_advisor.core.exceptions import MarketDataServiceError
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient

logger = logging.getLogger("ai_stock_advisor.core.options")


class OptionAnalyzer:
    """
    Option Chain Analyzer.
    Fetches option chains for active expiries, calculates Put-Call Ratio (PCR),
    evaluates Max Pain point, and structures Stop Loss & Target recommendations.
    """

    def __init__(
        self,
        market_client: MarketDataClient,
        indicator_engine: TechnicalIndicatorEngine,
    ) -> None:
        """Initializes OptionAnalyzer with market client and indicators engine."""
        self.market_client = market_client
        self.indicator_engine = indicator_engine

    def resolve_ticker(self, ticker: str) -> str:
        """Maps common NSE index search inputs to Yahoo Finance symbols."""
        symbol = ticker.strip().upper()
        if symbol in ("NIFTY", "NIFTY50", "NIFTY 50"):
            return "^NSEI"
        if symbol in ("BANKNIFTY", "NIFTYBANK", "BANK NIFTY"):
            return "^NSEBANK"
        return symbol

    def fetch_option_chain(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Queries Yahoo Finance for the nearest expiry option chain data.
        Returns Calls DataFrame, Puts DataFrame, and Current Spot Price.
        """
        resolved_symbol = self.resolve_ticker(ticker)
        logger.info("Retrieving option chain records for symbol '%s'...", resolved_symbol)
        
        try:
            # Get latest spot price
            price_df = self.market_client.fetch_ohlcv(resolved_symbol, period="5d", interval="1d")
            if price_df.empty:
                raise ValueError(f"Failed loading price history for spot price reference: {resolved_symbol}")
            spot_price = float(price_df["Close"].iloc[-1])

            # Query yfinance Options
            yf_ticker = yf.Ticker(resolved_symbol)
            expiries = yf_ticker.options
            
            if not expiries:
                raise ValueError(f"No option chains or expiry dates found for symbol: {resolved_symbol}")
                
            # Query nearest expiry date option chain
            nearest_expiry = expiries[0]
            logger.info("Loading options for nearest expiry date: %s", nearest_expiry)
            
            chain = yf_ticker.option_chain(nearest_expiry)
            calls = chain.calls.dropna(subset=["strike", "openInterest"])
            puts = chain.puts.dropna(subset=["strike", "openInterest"])
            
            return calls, puts, spot_price
            
        except Exception as exc:
            raise MarketDataServiceError(
                f"Option chain retrieval failed for ticker '{ticker}'",
                details={"ticker": ticker, "resolved": resolved_symbol, "error": str(exc)}
            ) from exc

    def calculate_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame) -> float:
        """
        Calculates the Max Pain strike price.
        The Max Pain point is the strike price where option buyers experience maximum loss,
        representing the point of minimum cumulative payout for option writers.
        """
        # Combine unique strike values
        strikes = sorted(list(set(calls["strike"]).union(set(puts["strike"]))))
        if not strikes:
            return 0.0

        min_loss = float("inf")
        max_pain_strike = strikes[0]

        # Convert to numpy arrays for high-performance vectorized operations
        call_strikes = calls["strike"].values
        call_oi = calls["openInterest"].values
        
        put_strikes = puts["strike"].values
        put_oi = puts["openInterest"].values

        for K in strikes:
            # Loss for call writers: max(0, spot - strike) * OI
            call_loss = np.maximum(0.0, K - call_strikes) * call_oi
            
            # Loss for put writers: max(0, strike - spot) * OI
            put_loss = np.maximum(0.0, put_strikes - K) * put_oi
            
            total_loss = float(np.sum(call_loss) + np.sum(put_loss))
            
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = K

        return float(max_pain_strike)

    def analyze_options(self, ticker: str) -> Dict[str, Any]:
        """
        Runs the full options analysis.
        Calculates ATM strike, PCR, Max Pain, and volatility-adjusted Stop Loss & Target.
        """
        resolved_symbol = self.resolve_ticker(ticker)
        calls, puts, spot = self.fetch_option_chain(ticker)
        
        # Calculate ATM Strike (closest strike to spot)
        atm_strike = float(min(calls["strike"], key=lambda x: abs(x - spot)))

        # 1. ITM / OTM Categorization
        itm_calls = calls[calls["strike"] < spot]["strike"].tolist()
        otm_calls = calls[calls["strike"] > spot]["strike"].tolist()
        
        itm_puts = puts[puts["strike"] > spot]["strike"].tolist()
        otm_puts = puts[puts["strike"] < spot]["strike"].tolist()

        # 2. Highest Open Interest (OI)
        highest_oi_call_idx = calls["openInterest"].idxmax()
        highest_oi_put_idx = puts["openInterest"].idxmax()
        
        highest_oi_call_strike = float(calls.loc[highest_oi_call_idx, "strike"])
        highest_oi_call_val = int(calls.loc[highest_oi_call_idx, "openInterest"])
        
        highest_oi_put_strike = float(puts.loc[highest_oi_put_idx, "strike"])
        highest_oi_put_val = int(puts.loc[highest_oi_put_idx, "openInterest"])

        # 3. Put-Call Ratio (PCR)
        total_call_oi = int(calls["openInterest"].sum())
        total_put_oi = int(puts["openInterest"].sum())
        pcr = float(total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0

        # 4. Max Pain
        max_pain = self.calculate_max_pain(calls, puts)

        # 5. Determine Sentiment
        # High PCR indicates puts written > calls written (Bullish)
        # Low PCR indicates calls written > puts written (Bearish)
        if pcr >= 1.25:
            sentiment = "BULLISH"
            suggestion = "BUY CALL"
        elif pcr <= 0.75:
            sentiment = "BEARISH"
            suggestion = "BUY PUT"
        else:
            sentiment = "NEUTRAL"
            suggestion = "HOLD / NO TRADE"

        # 6. Stop Loss & Target generation using ATR (Volatility-Adjusted)
        # Fetch 1y history for ATR calculation reference
        hist_df = self.market_client.fetch_ohlcv(resolved_symbol, period="1y", interval="1d")
        ind_df = self.indicator_engine.compute_all_indicators(hist_df)
        atr = float(ind_df["ATR_14"].iloc[-1])

        # Adjust SL and Target relative to current spot price
        if sentiment == "BULLISH":
            target = spot + (1.5 * atr)
            stop_loss = spot - (1.0 * atr)
        elif sentiment == "BEARISH":
            target = spot - (1.5 * atr)
            stop_loss = spot + (1.0 * atr)
        else:
            target = spot
            stop_loss = spot

        return {
            "Ticker": resolved_symbol,
            "Spot_Price": spot,
            "ATM_Strike": atm_strike,
            "PCR": round(pcr, 2),
            "Max_Pain": max_pain,
            "Sentiment": sentiment,
            "Suggestion": suggestion,
            "Suggested_Strike": atm_strike,
            "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "ATR": round(atr, 2),
            "Highest_OI_Call": {"strike": highest_oi_call_strike, "oi": highest_oi_call_val},
            "Highest_OI_Put": {"strike": highest_oi_put_strike, "oi": highest_oi_put_val},
            "ITM_Calls_Count": len(itm_calls),
            "OTM_Calls_Count": len(otm_calls),
            "ITM_Puts_Count": len(itm_puts),
            "OTM_Puts_Count": len(otm_puts),
        }
