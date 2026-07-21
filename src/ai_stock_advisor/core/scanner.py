"""
Stock Scanner Module.
Orchestrates downloading, indicator computation, scoring (0-100), and saving.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.market_data.constants import NIFTY_50_TICKERS

logger = logging.getLogger("ai_stock_advisor.core.scanner")


class StockScanner:
    """
    Scanner that pulls data for tickers, calculates technical indicators,
    and rates bullish strength on a 0-100 scale.
    """

    def __init__(
        self,
        market_client: MarketDataClient,
        indicator_engine: TechnicalIndicatorEngine,
    ) -> None:
        """Initializes the scanner with market client and indicator engine."""
        self.market_client = market_client
        self.indicator_engine = indicator_engine

    def score_stock(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates bullish trend indicators and scores the stock from 0 to 100
        based on the latest row in the technical indicator series.
        
        Weighting:
          - Close > EMA20: 15 points
          - Close > EMA50: 15 points
          - EMA20 > EMA50 (crossover alignment): 15 points
          - RSI > 50 (bullish side): 15 points
          - MACD > MACD_Signal: 15 points
          - ADX > 25 (strong trend): 15 points
          - Volume > 1.5 * 20-day Average Volume (volume spike): 10 points
        """
        # Calculate all indicators
        indicators_df = self.indicator_engine.compute_all_indicators(df)
        
        # Calculate 20-day volume average
        indicators_df["Vol_MA20"] = indicators_df["Volume"].rolling(window=20).mean()
        
        latest = indicators_df.iloc[-1]
        
        close = float(latest["Close"])
        volume = float(latest["Volume"])
        
        # 1. Close > EMA20 (15 pts)
        above_ema20 = bool(close > latest["EMA_20"])
        
        # 2. Close > EMA50 (15 pts)
        above_ema50 = bool(close > latest["EMA_50"])
        
        # 3. EMA Crossover (EMA20 > EMA50) (15 pts)
        ema_crossover = bool(latest["EMA_20"] > latest["EMA_50"])
        
        # 4. RSI > 50 (15 pts)
        rsi_val = float(latest["RSI_14"])
        rsi_bullish = bool(rsi_val > 50.0)
        
        # 5. MACD > MACD_Signal (15 pts)
        macd_bullish = bool(latest["MACD"] > latest["MACD_Signal"])
        
        # 6. ADX > 25 (15 pts)
        adx_val = float(latest["ADX_14"])
        adx_strong = bool(adx_val > 25.0)
        
        # 7. Volume Spike (10 pts)
        vol_ma20 = latest["Vol_MA20"]
        if pd.isna(vol_ma20) or vol_ma20 == 0:
            volume_spike = False
        else:
            volume_spike = bool(volume > 1.5 * vol_ma20)

        # Compute sum score
        score = 0.0
        if above_ema20:
            score += 15.0
        if above_ema50:
            score += 15.0
        if ema_crossover:
            score += 15.0
        if rsi_bullish:
            score += 15.0
        if macd_bullish:
            score += 15.0
        if adx_strong:
            score += 15.0
        if volume_spike:
            score += 10.0

        return {
            "Close": close,
            "Score": score,
            "Above_EMA20": above_ema20,
            "Above_EMA50": above_ema50,
            "EMA_Crossover": ema_crossover,
            "RSI": rsi_val,
            "MACD_Bullish": macd_bullish,
            "Volume_Spike": volume_spike,
            "ADX": adx_val,
        }

    def scan(
        self,
        tickers: List[str] | None = None,
        save_dir: Path | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Scans a list of stock tickers (defaults to Nifty 50).
        Downloads price data, calculates indicator ratings, and sorts by score.
        Saves output records in both CSV and JSON formats.
        """
        # Determine target list
        is_default_nifty = False
        if tickers is None:
            tickers = list(NIFTY_50_TICKERS)
            is_default_nifty = True
            
        logger.info("Starting scan for %d stock tickers...", len(tickers))
        
        # Retrieve stock historical data
        ticker_data: Dict[str, pd.DataFrame] = {}
        
        if is_default_nifty:
            try:
                # Optimized bulk download for default Nifty 50
                ticker_data = self.market_client.fetch_nifty_50(
                    period="1y", interval="1d", force_refresh=force_refresh
                )
            except Exception as exc:
                logger.error("Failed downloading Nifty 50 bulk index records: %s. Falling back to individual fetch.", exc)
                is_default_nifty = False

        if not is_default_nifty:
            # Fallback/Custom: Download individual tickers one by one safely
            for ticker in tickers:
                try:
                    df = self.market_client.fetch_ohlcv(
                        ticker, period="1y", interval="1d", force_refresh=force_refresh
                    )
                    ticker_data[ticker] = df
                except Exception as exc:
                    logger.warning("Skipping ticker '%s' due to query failure: %s", ticker, str(exc))

        # Perform scoring calculations
        scan_records: List[Dict[str, Any]] = []
        for ticker, df in ticker_data.items():
            try:
                # Minimum rows check (needs at least 200 rows for stable EMA 200)
                if len(df) < 50:
                    logger.warning("Ticker '%s' has insufficient history (%d rows). Skipping score.", ticker, len(df))
                    continue
                    
                score_dict = self.score_stock(df)
                record = {"Ticker": ticker}
                record.update(score_dict)
                scan_records.append(record)
            except Exception as exc:
                logger.error("Failed calculating scoring profile for '%s': %s", ticker, str(exc), exc_info=True)

        if not scan_records:
            logger.warning("No records were successfully scanned.")
            return pd.DataFrame()

        # Create sorted DataFrame
        scan_df = pd.DataFrame(scan_records)
        scan_df = scan_df.sort_values(by=["Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)

        # Ensure storage directory exists
        out_dir = save_dir or Path(settings.BASE_DIR) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = out_dir / "scan_results.csv"
        json_path = out_dir / "scan_results.json"

        # Save to CSV and JSON formats
        try:
            scan_df.to_csv(csv_path, index=False)
            logger.info("Saved scanner CSV results to %s", csv_path)
            
            scan_df.to_json(json_path, orient="records", indent=2)
            logger.info("Saved scanner JSON results to %s", json_path)
        except Exception as exc:
            logger.error("Error writing scan result files: %s", str(exc))

        return scan_df
