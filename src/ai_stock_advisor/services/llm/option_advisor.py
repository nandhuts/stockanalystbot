"""
AI Option Advisor Module.
Combines technical indicators and option chain metrics to generate AI trade recommendations.
"""
from enum import Enum
import logging
from typing import Any, Dict
from openai import OpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from ai_stock_advisor.core.exceptions import (
    ConfigurationError,
    LLMServiceError,
)

logger = logging.getLogger("ai_stock_advisor.services.llm.option_advisor")


class OptionSentimentEnum(str, Enum):
    """Option chain market sentiment categories."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class OptionTradeSuggestion(BaseModel):
    """Pydantic schema representing the AI options trade recommendation."""
    suggested_strategy: str = Field(description="The recommended options strategy, e.g., 'Long Call (ATM)', 'Bull Call Spread', 'Long Put (ATM)'.")
    strike_price: float = Field(description="The recommended option strike price to trade.")
    target_price: float = Field(description="Volatility-adjusted target price for the underlying security.")
    stop_loss: float = Field(description="Volatility-adjusted protective stop loss price for the underlying security.")
    probability_score: float = Field(description="AI probability score of the trade succeeding, as a percentage from 0% to 100%.", ge=0.0, le=100.0)
    sentiment: OptionSentimentEnum = Field(description="Market sentiment direction: BULLISH, BEARISH, or NEUTRAL.")
    thesis_reasoning: str = Field(description="Professional investment-style explanation justifying the trade setup relative to technicals, volume spikes, OI boundaries, and PCR.")


class AIOptionAdvisorClient:
    """
    Client for generating AI option trade recommendations by combining
    technical indicator inputs and options chain analysis.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        """Initializes the options advisor and OpenAI client."""
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model

        if not self.api_key or self.api_key == "mock_openai_api_key_for_testing":
            logger.warning("No valid OPENAI_API_KEY detected. Calls to OpenAI will fail unless mocked.")

        try:
            self.client = OpenAI(api_key=self.api_key or "dummy_key")
        except Exception as exc:
            raise ConfigurationError(
                "Failed to instantiate OpenAI client wrapper for option advisor.",
                details={"error": str(exc)}
            ) from exc

    def generate_option_trade(
        self,
        ticker: str,
        technicals: Dict[str, Any],
        options_data: Dict[str, Any],
    ) -> OptionTradeSuggestion:
        """
        Sends technical indicators and options metrics to OpenAI.
        Returns a structured OptionTradeSuggestion model.
        """
        system_instructions = (
            "You are a professional options trader, quantitative market analyst, and derivatives expert. "
            "Given a security's recent price actions, technical indicator values, option chain open interest (OI) profile, "
            "Put-Call Ratio (PCR), and Max Pain point, your task is to:\n"
            "1. Synthesize all trend, volume, and open interest metrics.\n"
            "2. Recommend the best options trading strategy (e.g. Call Buy, Put Buy, Spreads) matching the directional bias.\n"
            "3. Specify the optimal strike price, underlying target price, and stop loss coordinates.\n"
            "4. Assign a probability score (0% to 100%) indicating your confidence in the trade success.\n"
            "5. Generate a professional investment-style explanation explaining your thesis, referencing key indicators."
        )

        user_content = (
            f"Analyze the options trading opportunity for symbol: {ticker}.\n\n"
            f"💰 Spot Price: ₹{options_data.get('Spot_Price', 0.0):,.2f}\n"
            f"⚡ Scanner Trend Score: {technicals.get('Score', 0.0):.0f} / 100\n\n"
            f"📊 Technical Indicators:\n"
            f"- Close Price: ₹{technicals.get('Close', 0.0):,.2f}\n"
            f"- Above EMA 20: {technicals.get('Above_EMA20', False)}\n"
            f"- Above EMA 50: {technicals.get('Above_EMA50', False)}\n"
            f"- EMA Crossover (20 > 50): {technicals.get('EMA_Crossover', False)}\n"
            f"- RSI (14): {technicals.get('RSI', 50.0):.1f}\n"
            f"- MACD Bullish: {technicals.get('MACD_Bullish', False)}\n"
            f"- Volume Spike Active: {technicals.get('Volume_Spike', False)}\n"
            f"- ADX (14) Trend Strength: {technicals.get('ADX', 0.0):.1f}\n"
            f"- Volatility ATR (14): {options_data.get('ATR', 0.0):.2f}\n\n"
            f"🔑 Option Chain Data (Nearest Expiry):\n"
            f"- At-The-Money (ATM) Strike: ₹{options_data.get('ATM_Strike', 0.0):,.2f}\n"
            f"- Put-Call Ratio (PCR): {options_data.get('PCR', 0.0):.2f}\n"
            f"- Max Pain Strike: ₹{options_data.get('Max_Pain', 0.0):,.2f}\n"
            f"- Highest Call OI strike: ₹{options_data.get('Highest_OI_Call', {}).get('strike', 0.0):,.2f} "
            f"({options_data.get('Highest_OI_Call', {}).get('oi', 0)} contracts)\n"
            f"- Highest Put OI strike: ₹{options_data.get('Highest_OI_Put', {}).get('strike', 0.0):,.2f} "
            f"({options_data.get('Highest_OI_Put', {}).get('oi', 0)} contracts)\n"
            f"- Calls Count (ITM/OTM): {options_data.get('ITM_Calls_Count', 0)} / {options_data.get('OTM_Calls_Count', 0)}\n"
            f"- Puts Count (ITM/OTM): {options_data.get('ITM_Puts_Count', 0)} / {options_data.get('OTM_Puts_Count', 0)}\n\n"
            "Formulate the response using the requested JSON structure."
        )

        logger.info("Requesting AI Option Trade Suggestion from OpenAI for ticker %s...", ticker)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content},
                ],
                response_format=OptionTradeSuggestion,
            )

            report = completion.choices[0].message.parsed
            
            if report is None:
                raise LLMServiceError(
                    f"OpenAI returned empty parsed options advisor trade for ticker '{ticker}'",
                    details={"ticker": ticker}
                )

            return report

        except Exception as exc:
            logger.error("Failed communicating with OpenAI API during option advising: %s", str(exc), exc_info=True)
            raise LLMServiceError(
                f"Failed generating AI options trade recommendation for ticker '{ticker}' via OpenAI API.",
                details={"ticker": ticker, "model": self.model, "error": str(exc)}
            ) from exc
