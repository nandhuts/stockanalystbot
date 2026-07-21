"""
LLM Service client.
Interfaces with OpenAI API using Pydantic Structured Outputs to guarantee type safety.
"""
from enum import Enum
import logging
from typing import Any, Dict
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from ai_stock_advisor.core.exceptions import (
    ConfigurationError,
    LLMServiceError,
)

logger = logging.getLogger("ai_stock_advisor.services.llm")


class RecommendationEnum(str, Enum):
    """Enumeration of possible recommendation actions."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class TechnicalAnalysisReport(BaseModel):
    """
    Pydantic schema representing the structured response required from the LLM.
    Guarantees strict formatting matching the professional investment report.
    """
    trend_analysis: str = Field(description="Professional analysis of the security's current price trend (e.g. support, resistance, direction).")
    rsi_analysis: str = Field(description="Explanation of the RSI oscillator signal and current overbought/oversold levels.")
    macd_analysis: str = Field(description="Detailed explanation of the MACD convergence/divergence lines and momentum histogram.")
    ema_analysis: str = Field(description="Analysis of the EMA (20, 50, 200) alignments and crossover signals.")
    recommendation: RecommendationEnum = Field(description="Action recommendation: BUY, HOLD, or SELL.")
    confidence_percentage: int = Field(description="Confidence level as an integer percentage between 0 and 100.", ge=0, le=100)
    investment_thesis: str = Field(description="A professional, investment-grade summary presenting the overall thesis for the rating.")


class LLMAnalysisClient:
    """
    Client for interacting with OpenAI to generate structured stock research reviews.
    Translates library exceptions to domain errors.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        """
        Initializes the LLM analysis client.
        Uses specified API key or defaults to environment-configured settings.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        
        # Verify API configuration
        if not self.api_key or self.api_key == "mock_openai_api_key_for_testing":
            logger.warning("No valid OPENAI_API_KEY detected. Calls to OpenAI will fail unless mocked.")
            
        try:
            # We initialize client. It will fail on actual calls if key is bad, which is handled at call time.
            self.client = OpenAI(api_key=self.api_key or "dummy_key")
        except Exception as exc:
            raise ConfigurationError(
                "Failed to instantiate OpenAI client wrapper.",
                details={"error": str(exc)}
            ) from exc

    def _prepare_prompt_payload(self, ticker: str, latest_row: pd.Series, score: float) -> Dict[str, Any]:
        """Formats indicators data row into a structured prompt dictionary payload."""
        # Safely extract values with default fallbacks if missing
        close = latest_row.get("Close", 0.0)
        volume = latest_row.get("Volume", 0.0)
        
        ema_20 = latest_row.get("EMA_20", None)
        ema_50 = latest_row.get("EMA_50", None)
        ema_200 = latest_row.get("EMA_200", None)
        
        rsi = latest_row.get("RSI_14", None)
        
        macd = latest_row.get("MACD", None)
        macd_sig = latest_row.get("MACD_Signal", None)
        macd_hist = latest_row.get("MACD_Hist", None)
        
        adx = latest_row.get("ADX_14", None)
        
        vol_ma = latest_row.get("Vol_MA20", None)

        return {
            "ticker": ticker,
            "close": close,
            "score": score,
            "volume": volume,
            "volume_avg_20d": vol_ma,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_sig,
            "macd_histogram": macd_hist,
            "adx": adx,
        }

    def generate_analysis(
        self,
        ticker: str,
        indicators_df: pd.DataFrame,
        score: float,
    ) -> TechnicalAnalysisReport:
        """
        Queries OpenAI with the latest technical indicators to construct
        a structured, professional TechnicalAnalysisReport.
        
        Raises LLMServiceError if OpenAI communication or parsing fails.
        """
        if indicators_df.empty:
            raise ValueError(f"Cannot generate LLM analysis for empty DataFrame on ticker: {ticker}")

        latest_row = indicators_df.iloc[-1]
        payload = self._prepare_prompt_payload(ticker, latest_row, score)
        
        system_instructions = (
            "You are a professional chartered financial analyst (CFA) and senior portfolio advisor. "
            "Your objective is to write structured, professional, and investment-grade technical analysis reports. "
            "Ensure that you strictly explain each technical parameter (Trend, RSI, MACD, and EMAs) in detail, "
            "provide a clear recommendation (BUY, HOLD, or SELL), assign a confidence percentage (0-100), "
            "and write a cohesive, comprehensive investment thesis."
        )

        user_content = (
            f"Please generate a technical analysis review for the stock ticker: {payload['ticker']}.\n\n"
            f"Latest Market Price Data & Technical Indicators:\n"
            f"- Close Price: ₹{payload['close']:,.2f}\n"
            f"- Algorithmic Bullish Trend Score: {payload['score']} / 100\n"
            f"- Exponential Moving Averages: EMA20={payload['ema_20']}, EMA50={payload['ema_50']}, EMA200={payload['ema_200']}\n"
            f"- RSI (14-day): {payload['rsi']}\n"
            f"- MACD: Line={payload['macd']}, Signal={payload['macd_signal']}, Histogram={payload['macd_histogram']}\n"
            f"- ADX (14-day Trend Strength): {payload['adx']}\n"
            f"- Volume: Current={payload['volume']}, 20-Day Average={payload['volume_avg_20d']}\n\n"
            "Format your response as a professional investment report according to the requested JSON structure."
        )

        logger.info("Sending analysis query to OpenAI model %s for ticker %s...", self.model, ticker)

        try:
            # Query OpenAI Chat Completion with structured outputs parsing
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content},
                ],
                response_format=TechnicalAnalysisReport,
            )
            
            report = completion.choices[0].message.parsed
            
            if report is None:
                raise LLMServiceError(
                    f"OpenAI returned empty parsed result for ticker '{ticker}'",
                    details={"ticker": ticker}
                )
                
            return report

        except Exception as exc:
            logger.error("Failed communicating with OpenAI API: %s", str(exc), exc_info=True)
            raise LLMServiceError(
                f"Failed generating AI analysis report for symbol '{ticker}' via OpenAI API.",
                details={"ticker": ticker, "model": self.model, "error": str(exc)}
            ) from exc
