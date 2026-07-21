"""
News Analyzer Module.
Retrieves ticker news from yfinance and performs structured sentiment analysis via OpenAI.
"""
from enum import Enum
import logging
from typing import Any, Dict, List
import yfinance as yf
from openai import OpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from ai_stock_advisor.core.exceptions import (
    ConfigurationError,
    LLMServiceError,
)

logger = logging.getLogger("ai_stock_advisor.services.llm.news_analyzer")


class SentimentEnum(str, Enum):
    """Sentiment classification categories."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class NewsSentimentAnalysis(BaseModel):
    """Pydantic schema representing individual news sentiment rating."""
    headline: str = Field(description="Original headline/title of the news story.")
    summary: str = Field(description="A concise one-sentence AI summary outlining the financial impact of this story.")
    sentiment: SentimentEnum = Field(description="Classification: POSITIVE (bullish), NEGATIVE (bearish), or NEUTRAL.")
    sentiment_score: float = Field(description="Sentiment polarity score from -1.0 (very bearish) to 1.0 (very bullish).", ge=-1.0, le=1.0)


class StockNewsReport(BaseModel):
    """Pydantic schema representing the complete news analysis report payload."""
    articles: List[NewsSentimentAnalysis] = Field(description="List of individually analyzed news articles.")
    overall_sentiment: SentimentEnum = Field(description="Consolidated overall market sentiment across all news articles.")
    average_sentiment_score: float = Field(description="The calculated average polarity score (-1.0 to 1.0) across all articles.", ge=-1.0, le=1.0)


class NewsAnalyzer:
    """
    Service for collecting stock market news and applying OpenAI models to
    summarize, classify, and calculate sentiment polarity scores.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        """Initializes the news analyzer client and OpenAI wrapper."""
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model

        if not self.api_key or self.api_key == "mock_openai_api_key_for_testing":
            logger.warning("No valid OPENAI_API_KEY detected. Calls to OpenAI will fail unless mocked.")

        try:
            self.client = OpenAI(api_key=self.api_key or "dummy_key")
        except Exception as exc:
            raise ConfigurationError(
                "Failed to instantiate OpenAI client wrapper for news analyzer.",
                details={"error": str(exc)}
            ) from exc

    def fetch_news_headlines(self, ticker: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves news feed records for a given stock using yfinance.
        Extracts title, publisher, and timestamp details.
        """
        logger.info("Fetching raw news feed for ticker '%s' from yfinance...", ticker)
        try:
            yf_ticker = yf.Ticker(ticker)
            raw_news = yf_ticker.news
            
            if not raw_news:
                logger.warning("No news stories returned for ticker '%s'", ticker)
                return []
                
            parsed_stories = []
            for story in raw_news:
                # Standard story attributes
                title = story.get("title", "").strip()
                publisher = story.get("publisher", "").strip()
                link = story.get("link", "").strip()
                
                if title:
                    parsed_stories.append({
                        "title": title,
                        "publisher": publisher,
                        "link": link
                    })
                    
                if len(parsed_stories) >= max_articles:
                    break
                    
            return parsed_stories
        except Exception as exc:
            logger.error("Failed querying news items via yfinance: %s", str(exc))
            # Fall back to empty rather than crashing, as news is non-blocking for indicators
            return []

    def analyze_news(self, ticker: str, max_articles: int = 5) -> StockNewsReport:
        """
        Pulls stock news and invokes OpenAI structured outputs to generate
        a consolidated StockNewsReport.
        """
        stories = self.fetch_news_headlines(ticker, max_articles=max_articles)
        
        if not stories:
            logger.info("No news stories available for '%s'. Returning neutral default report.", ticker)
            return StockNewsReport(
                articles=[],
                overall_sentiment=SentimentEnum.NEUTRAL,
                average_sentiment_score=0.0
            )

        # Formulate headlines prompt block
        stories_block = ""
        for idx, story in enumerate(stories, 1):
            stories_block += f"Article #{idx}:\n"
            stories_block += f"- Headline: {story['title']}\n"
            if story.get('publisher'):
                stories_block += f"- Publisher: {story['publisher']}\n"
            stories_block += "\n"

        system_instructions = (
            "You are a professional financial news analyst and market sentiment researcher. "
            "Given a list of recent financial news articles for a stock ticker, your job is to: "
            "1. Read and analyze the headline of each story.\n"
            "2. Generate a clear one-sentence summary explaining its potential impact on the stock price.\n"
            "3. Classify the sentiment of each article as POSITIVE, NEGATIVE, or NEUTRAL.\n"
            "4. Assign a polarity sentiment score between -1.0 (highly negative/bearish) and 1.0 (highly positive/bullish).\n"
            "5. Consolidate these findings into a final report containing all articles, an overall sentiment class, and an average score."
        )

        user_content = (
            f"Analyze the following recent news articles for the stock ticker: {ticker}.\n\n"
            f"{stories_block}"
            "Format the response using the requested JSON structure."
        )

        logger.info("Sending news sentiment analysis request to OpenAI model %s for ticker %s...", self.model, ticker)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content},
                ],
                response_format=StockNewsReport,
            )
            
            report = completion.choices[0].message.parsed
            
            if report is None:
                raise LLMServiceError(
                    f"OpenAI returned empty parsed news report for ticker '{ticker}'",
                    details={"ticker": ticker}
                )
                
            return report

        except Exception as exc:
            logger.error("Failed communicating with OpenAI API during news analysis: %s", str(exc), exc_info=True)
            raise LLMServiceError(
                f"Failed generating AI news sentiment analysis report for symbol '{ticker}' via OpenAI API.",
                details={"ticker": ticker, "model": self.model, "error": str(exc)}
            ) from exc
