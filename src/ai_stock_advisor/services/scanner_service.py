"""
Scanner Service Module.
Downloads active F&O ticker listings and collects historical daily and intraday candles.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf

from ai_stock_advisor.services.market_data.client import MarketDataClient

logger = logging.getLogger("ai_stock_advisor.services.scanner_service")


class MorningScannerService:
    """
    Manages collection of market data (daily, 15-minute, 5-minute candles)
    for all active equity derivatives (F&O segment) on the National Stock Exchange (NSE).
    """

    def __init__(self, market_client: MarketDataClient | None = None) -> None:
        """Initializes the service with standard market data client."""
        self.market_client = market_client or MarketDataClient()

    def fetch_fo_tickers(self) -> List[str]:
        """
        Dynamically downloads and filters F&O underlying stock tickers from Zerodha Kite.
        Filters out index derivatives (e.g. NIFTY, BANKNIFTY) and formats with '.NS'.
        """
        logger.info("Fetching F&O underlyings list from Zerodha API...")
        try:
            url = "https://api.kite.trade/instruments"
            df = pd.read_csv(url)
            
            # Filter for NSE Derivatives futures segment
            nfo_futs = df[(df["exchange"] == "NFO") & (df["segment"] == "NFO-FUT")]
            
            # Known indices to exclude
            indices = {
                "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50",
                "SENSEX", "BANKEX", "INDIAVIX"
            }
            
            raw_names = nfo_futs["name"].unique()
            tickers = [
                f"{name}.NS" for name in raw_names 
                if isinstance(name, str) and name not in indices
            ]
            
            tickers = sorted(list(set(tickers)))
            logger.info("Successfully fetched %d active F&O equity symbols.", len(tickers))
            return tickers

        except Exception as exc:
            logger.error("Failed fetching dynamic F&O tickers: %s. Using fallback list.", str(exc))
            # Safe curated fallback list of top liquid F&O stocks
            return [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LTIM.NS", "LT.NS",
                "AXISBANK.NS", "KOTAKBANK.NS", "M&M.NS", "MARUTI.NS", "TATAMOTORS.NS",
                "TATASTEEL.NS", "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS"
            ]

    def _fetch_stock_timeframes(self, ticker: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Helper method executed inside thread pool to retrieve daily, 15m, and 5m candles.
        """
        try:
            # 1. Fetch Daily Candles (1 Month history)
            df_daily = self.market_client.fetch_ohlcv(ticker, period="1mo", interval="1d", force_refresh=True)
            
            # 2. Fetch 15-Minute Candles (5 Days history)
            df_15m = self.market_client.fetch_ohlcv(ticker, period="5d", interval="15m", force_refresh=True)
            
            # 3. Fetch 5-Minute Candles (5 Days history)
            df_5m = self.market_client.fetch_ohlcv(ticker, period="5d", interval="5m", force_refresh=True)
            
            return ticker, df_daily, df_15m, df_5m
        except Exception as exc:
            logger.warning("Failed loading candle timeframes for symbol '%s': %s", ticker, str(exc))
            return ticker, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def fetch_all_candles(self, tickers: List[str], max_workers: int = 15) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Orchestrates multi-threaded download of candle histories for a list of tickers.
        """
        logger.info("Initializing parallel fetch of candles for %d symbols...", len(tickers))
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_stock_timeframes, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    sym, daily, m15, m5 = future.result()
                    if not daily.empty:
                        results[sym] = {
                            "daily": daily,
                            "15m": m15,
                            "5m": m5
                        }
                except Exception as exc:
                    logger.error("Thread execution failed for '%s': %s", ticker, str(exc))
                    
        logger.info("Parallel candle fetch complete. Loaded datasets for %d/%d securities.", len(results), len(tickers))
        return results
