"""
AI Chat Assistant Module.
Parses user financial questions, extracts ticker symbols, builds technical and news contexts,
and generates structured investment advisory reports via OpenAI.
"""
import logging
from typing import Any, Dict, Optional
import pandas as pd
from openai import OpenAI

from config.settings import settings
from ai_stock_advisor.core.exceptions import (
    ConfigurationError,
    LLMServiceError,
)
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.options import OptionAnalyzer
from ai_stock_advisor.core.scanner import StockScanner
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer

logger = logging.getLogger("ai_stock_advisor.services.llm.chat_assistant")


class AIChatAssistantClient:
    """
    RAG-based conversational AI Stock Assistant.
    Extracts tickers, fetches live statistics and news summaries,
    and returns investment advice reports.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        """Initializes OpenAI client connection."""
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model

        if not self.api_key or self.api_key == "mock_openai_api_key_for_testing":
            logger.warning("No valid OPENAI_API_KEY detected. Calls to OpenAI will fail unless mocked.")

        try:
            self.client = OpenAI(api_key=self.api_key or "dummy_key")
        except Exception as exc:
            raise ConfigurationError(
                "Failed to instantiate OpenAI client wrapper for chat assistant.",
                details={"error": str(exc)}
            ) from exc

    def extract_ticker(self, query: str) -> Optional[str]:
        """
        Queries OpenAI with the user message to extract a clean ticker symbol.
        Returns Yahoo Finance ticker (e.g. INFY.NS, TCS.NS, ^NSEI, AAPL) or None.
        """
        system_instructions = (
            "You are a helpful stock market data parser. "
            "Identify if the user is asking about a specific stock, index, or cryptocurrency. "
            "If they are, output ONLY the corresponding Yahoo Finance ticker symbol (e.g., RELIANCE.NS, TCS.NS, ^NSEI, AAPL, BTC-USD). "
            "If they are asking a general question, talking about multiple stocks, or if no single ticker fits, reply with 'NONE'."
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            response = completion.choices[0].message.content.strip().upper()
            if response == "NONE" or not response:
                return None
            return response
        except Exception as exc:
            logger.error("Failed to extract ticker from user query: %s", exc)
            return None

    def compile_context(self, ticker: str) -> Dict[str, Any]:
        """
        Compiles real-time metrics for a target ticker:
        Technicals, Support, Resistance, Option Chains, and live News Sentiment.
        """
        logger.info("Chat Assistant: Compiling context dataset for symbol '%s'...", ticker)
        market_client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        
        # 1. Historical Prices and Technical Indicators
        try:
            hist_df = market_client.fetch_ohlcv(ticker, period="1y", interval="1d")
            if hist_df.empty:
                raise ValueError("Empty price history loaded.")
            df_ind = engine.compute_all_indicators(hist_df)
            latest = df_ind.iloc[-1]
            
            # Support (20-day low) and Resistance (20-day high)
            support = float(df_ind["Low"].tail(20).min())
            resistance = float(df_ind["High"].tail(20).max())
            spot_price = float(latest["Close"])
        except Exception as exc:
            logger.error("Failed compiling core technicals context: %s", exc)
            raise ValueError(f"Could not load technical indicators history for symbol: {ticker}")

        # 2. Score Dict from StockScanner
        try:
            scanner = StockScanner(market_client, engine)
            score_dict = scanner.score_stock(hist_df)
        except Exception:
            score_dict = {}

        # 3. Option Chain Metrics
        try:
            opt_analyzer = OptionAnalyzer(market_client, engine)
            opt_data = opt_analyzer.analyze_options(ticker)
        except Exception:
            opt_data = {}

        # 4. News Sentiment
        try:
            news_analyzer = NewsAnalyzer()
            news_report = news_analyzer.analyze_news(ticker, max_articles=3)
            news_summary = {
                "overall_sentiment": news_report.overall_sentiment.value,
                "average_score": news_report.average_sentiment_score,
                "articles": [
                    {"headline": art.headline, "summary": art.summary, "sentiment": art.sentiment.value}
                    for art in news_report.articles
                ]
            }
        except Exception:
            news_summary = {}

        return {
            "Ticker": ticker,
            "Spot_Price": spot_price,
            "Support": support,
            "Resistance": resistance,
            "Technicals": score_dict,
            "Option_Chain": opt_data,
            "News_Sentiment": news_summary
        }

    def ask_advisor(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Sends the query and compiled context to OpenAI.
        Generates a structured investment advisor markdown response.
        """
        system_instructions = (
            "You are the AI Stock Advisor Chat Assistant, a premier derivatives researcher and quantitative technical analyst. "
            "If the user is asking about a specific stock, we will provide you with a structured context block containing live technical indicators, support and resistance boundaries, Put-Call Ratio (PCR), Max Pain strikes, and consolidated news sentiment.\n\n"
            "If context is provided, you MUST structure your reply using this exact markdown template:\n"
            "### 📊 [Ticker] AI Investment Advice\n"
            "* **Trend**: [Brief summary of EMA crossover direction and ADX strength]\n"
            "* **Support**: [Specific Support Price Level, e.g. ₹1,450.00]\n"
            "* **Resistance**: [Specific Resistance Price Level, e.g. ₹1,550.00]\n"
            "* **Entry**: [Recommended Entry Price Level or range]\n"
            "* **Stop Loss**: [Protective Stop Loss price level]\n"
            "* **Target**: [Target take profit price level]\n"
            "* **Risk**: [Evaluate risk-to-reward ratio and volatility setup]\n"
            "* **Probability**: [Estimated success probability percentage, e.g. 78%]\n\n"
            "### 🔬 Thesis & Technical Analysis\n"
            "[Detailed paragraph explaining EMA, RSI, MACD alignments, and volume profiles.]\n\n"
            "### 📰 Recent News Sentiment\n"
            "[Detailed paragraph summarizing recent news headlines and their consolidated sentiment impact.]\n\n"
            "If no context is provided (e.g. the user asks a general financial question like 'What is RSI?'), "
            "answer the user's question directly using your general financial expertise in a polite, professional manner "
            "without needing the stock templates."
        )

        user_content = f"User Question: {query}\n\n"
        if context:
            user_content += f"Compiled Stock Context:\n{context}\n"

        logger.info("Requesting chat advice from OpenAI model %s...", self.model)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content}
                ]
            )
            return completion.choices[0].message.content
        except Exception as exc:
            logger.error("Failed generating chat advisor advice: %s", exc)
            raise LLMServiceError(
                "Failed generating AI chat advisor advice via OpenAI.",
                details={"model": self.model, "error": str(exc)}
            ) from exc
