import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List
import pandas as pd
import yfinance as yf

from config.settings import settings
from ai_stock_advisor.core.exceptions import (
    InvalidTickerError,
    MarketDataServiceError,
)
from ai_stock_advisor.services.market_data.constants import NIFTY_50_TICKERS

logger = logging.getLogger("ai_stock_advisor.services.market_data")


def retry_on_failure(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[..., Any]:
    """
    Decorator that retries the wrapped function upon encountering general exceptions,
    implementing exponential backoff.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    logger.warning(
                        "Attempt %d/%d failed for '%s'. Error: %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(exc),
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
                        delay *= backoff_factor
            
            # If all attempts are exhausted, raise the final exception
            raise MarketDataServiceError(
                f"Failed executing '{func.__name__}' after {max_attempts} attempts.",
                details={"original_error": str(last_exception)}
            ) from last_exception
        return wrapper
    return decorator


class MarketDataClient:
    """
    Client for retrieving historical stock market OHLCV data using yfinance.
    Implements robust error translation, local file-based caching, and exponential retries.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int = 43200,  # 12 hours default
    ) -> None:
        """
        Initializes the market data client.
        Creates caching directory structures.
        """
        self.cache_dir = cache_dir or Path(settings.BASE_DIR) / "data" / "cache"
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("MarketDataClient initialized with cache dir: %s", self.cache_dir)

    def _get_cache_path(self, ticker: str, period: str, interval: str) -> Path:
        """Helper to generate local cache file path."""
        # Sanitize ticker string for file names (remove dots, special chars)
        safe_ticker = ticker.replace(".", "_").replace("-", "_").lower()
        filename = f"{safe_ticker}_{period}_{interval}.csv"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Helper to check if cache exists and has not expired."""
        if not cache_path.exists():
            return False
        
        mtime = os.path.getmtime(cache_path)
        age = time.time() - mtime
        return age < self.cache_ttl_seconds

    @retry_on_failure(max_attempts=3, initial_delay=1.0, backoff_factor=2.0)
    def _download_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Performs the direct yfinance network fetch.
        Decorated to support automatic exponential retries.
        """
        logger.info("Fetching network market data for %s (period=%s, interval=%s)", ticker, period, interval)
        try:
            # Fetch using Ticker object
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, interval=interval)
            return df
        except Exception as exc:
            raise MarketDataServiceError(
                f"yfinance failed querying ticker '{ticker}'",
                details={"ticker": ticker, "period": period, "interval": interval, "error": str(exc)}
            ) from exc

    def fetch_ohlcv(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves historical stock data for a given ticker.
        Applies local cache checking first, falling back to network fetch.
        
        Raises InvalidTickerError if ticker returns empty data.
        Raises MarketDataServiceError if network request fails after retries.
        """
        cache_path = self._get_cache_path(ticker, period, interval)
        
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info("Cache hit for %s (%s, %s). Loading file.", ticker, period, interval)
            try:
                # Load cache index as parsed datetime
                cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not cached_df.empty:
                    return cached_df
            except Exception as exc:
                logger.warning("Failed reading cache file %s: %s. Reloading from source.", cache_path, str(exc))

        # Cache miss or expired: download from API
        df = self._download_history(ticker, period, interval)
        
        # Verify ticker validity (yfinance returns empty DataFrame for invalid ticker symbols)
        if df.empty:
            raise InvalidTickerError(
                ticker=ticker,
                message=f"No stock data found for ticker '{ticker}' (period={period}, interval={interval})"
            )
            
        # Write to cache
        try:
            df.to_csv(cache_path)
            logger.debug("Successfully saved cache file: %s", cache_path)
        except Exception as exc:
            logger.warning("Failed writing cache to %s: %s", cache_path, str(exc))
            
        return df

    def fetch_nifty_50(
        self,
        period: str = "1mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Downloads data for all 50 stocks in the Nifty 50 index.
        Uses yfinance bulk download helper for efficiency, and caches files individually.
        Returns a dictionary mapping ticker symbol to individual DataFrame.
        """
        logger.info("Initializing download of Nifty 50 index stocks...")
        
        tickers = list(NIFTY_50_TICKERS)
        results: Dict[str, pd.DataFrame] = {}
        missing_tickers: List[str] = []
        
        # Check cache first for all tickers to avoid large downloads
        if not force_refresh:
            for ticker in tickers:
                cache_path = self._get_cache_path(ticker, period, interval)
                if self._is_cache_valid(cache_path):
                    try:
                        results[ticker] = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    except Exception:
                        missing_tickers.append(ticker)
                else:
                    missing_tickers.append(ticker)
        else:
            missing_tickers = tickers
            
        if not missing_tickers:
            logger.info("Loaded all Nifty 50 tickers successfully from local cache.")
            return results

        logger.info(
            "Cache miss for %d/%d Nifty 50 tickers. Downloading batch...",
            len(missing_tickers),
            len(tickers),
        )
        
        try:
            # Perform bulk download
            # group_by='ticker' ensures multi-indexed column or dictionary-like structures
            data = yf.download(
                tickers=missing_tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            
            # Process bulk results
            # yfinance return structure depends on number of tickers requested
            for ticker in missing_tickers:
                ticker_df = pd.DataFrame()
                
                # If only one ticker was missing
                if len(missing_tickers) == 1:
                    ticker_df = data
                else:
                    # If multiple tickers, columns are multi-indexed: (Ticker, PriceType)
                    if ticker in data.columns.levels[0]:
                        ticker_df = data[ticker].dropna(how="all")
                
                if ticker_df.empty:
                    logger.warning("No data returned for Nifty 50 component ticker: %s", ticker)
                    continue
                    
                # Cache results individually
                cache_path = self._get_cache_path(ticker, period, interval)
                try:
                    ticker_df.to_csv(cache_path)
                except Exception as exc:
                    logger.warning("Failed writing bulk ticker cache for %s: %s", ticker, str(exc))
                    
                results[ticker] = ticker_df
                
        except Exception as exc:
            raise MarketDataServiceError(
                "Bulk fetch of Nifty 50 tickers encountered an unexpected API failure.",
                details={"missing_tickers": missing_tickers, "error": str(exc)}
            ) from exc
            
        return results
