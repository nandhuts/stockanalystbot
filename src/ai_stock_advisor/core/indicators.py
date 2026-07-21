"""
Technical Indicator Engine.
Provides clean, high-performance mathematical indicators calculated over OHLCV DataFrames.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("ai_stock_advisor.core.indicators")


class TechnicalIndicatorEngine:
    """
    Calculates technical analysis indicators (EMAs, RSI, MACD, Bollinger Bands, ATR, VWAP, ADX)
    from a standard stock market DataFrame.
    
    Operates on copies of data to maintain immutability of original structures.
    """

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Validates that input DataFrame has standard OHLCV columns."""
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns for technical analysis: {missing}")

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = "Close", span: int = 20) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        return df[column].ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI) using Wilder's Smoothing Method.
        """
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Exponential moving average with alpha = 1 / period (Wilder's Smoothing)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Replace NaN with 50 (neutral level) for safety
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculates Moving Average Convergence Divergence (MACD).
        Returns a DataFrame with MACD, MACD_Signal, and MACD_Hist columns.
        """
        fast_ema = df["Close"].ewm(span=fast, adjust=False).mean()
        slow_ema = df["Close"].ewm(span=slow, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "MACD": macd_line,
                "MACD_Signal": signal_line,
                "MACD_Hist": histogram,
            },
            index=df.index,
        )

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculates Bollinger Bands (Middle, Upper, Lower).
        Returns a DataFrame containing BB_Middle, BB_Upper, and BB_Lower.
        """
        middle = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return pd.DataFrame(
            {
                "BB_Middle": middle,
                "BB_Upper": upper,
                "BB_Lower": lower,
            },
            index=df.index,
        )

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates Average True Range (ATR) using Wilder's Smoothing.
        """
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's Smoothing (alpha = 1 / period)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculates cumulative Volume Weighted Average Price (VWAP).
        Formula: sum(Typical Price * Volume) / sum(Volume)
        """
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        pv = typical_price * df["Volume"]
        
        cum_pv = pv.cumsum()
        cum_vol = df["Volume"].cumsum()
        
        # Handle zero cumulative volume cases
        vwap = cum_pv / cum_vol.replace(0, np.nan)
        return vwap.ffill().fillna(df["Close"])

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates Average Directional Index (ADX) using Wilder's DM smoothing.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # High and Low differences
        up_move = high.diff()
        down_move = -low.diff()

        # Directional Movement (+DM, -DM)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # True Range
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's Smoothing (alpha = 1 / period)
        smoothed_tr = tr.ewm(alpha=1 / period, adjust=False).mean()
        smoothed_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        smoothed_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

        # Directional Indicators (+DI, -DI)
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, np.nan))
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, np.nan))

        # Directional Index (DX)
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / di_sum.replace(0, np.nan))

        # Average Directional Index (ADX)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx.fillna(0.0)

    @classmethod
    def compute_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates columns and calculates EMA (20, 50, 200), RSI, MACD, Bollinger Bands, ATR, VWAP, and ADX.
        Returns a new DataFrame combining the raw data with all calculated indicators.
        """
        cls._validate_columns(df)
        
        # Work on a copy of input DataFrame to ensure immutability
        res = df.copy()

        # Exponential Moving Averages
        res["EMA_20"] = cls.calculate_ema(df, column="Close", span=20)
        res["EMA_50"] = cls.calculate_ema(df, column="Close", span=50)
        res["EMA_200"] = cls.calculate_ema(df, column="Close", span=200)

        # RSI (14)
        res["RSI_14"] = cls.calculate_rsi(df, period=14)

        # MACD (12, 26, 9)
        macd_df = cls.calculate_macd(df, fast=12, slow=26, signal=9)
        res["MACD"] = macd_df["MACD"]
        res["MACD_Signal"] = macd_df["MACD_Signal"]
        res["MACD_Hist"] = macd_df["MACD_Hist"]

        # Bollinger Bands (20, 2.0)
        bb_df = cls.calculate_bollinger_bands(df, period=20, num_std=2.0)
        res["BB_Middle"] = bb_df["BB_Middle"]
        res["BB_Upper"] = bb_df["BB_Upper"]
        res["BB_Lower"] = bb_df["BB_Lower"]

        # ATR (14)
        res["ATR_14"] = cls.calculate_atr(df, period=14)

        # VWAP
        res["VWAP"] = cls.calculate_vwap(df)

        # ADX (14)
        res["ADX_14"] = cls.calculate_adx(df, period=14)

        logger.info("Successfully calculated all technical indicators.")
        return res
