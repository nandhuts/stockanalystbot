"""
News Sentiment Analysis Module.
Extracts financial news from yfinance and processes sentiment indicators.
"""
import logging
from typing import Any, Dict, List

from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer, SentimentEnum

logger = logging.getLogger("ai_stock_advisor.analysis.news_sentiment")


class MorningNewsSentimentAnalyzer:
    """
    Analyzes overnight market news for individual securities,
    returning normalized sentiment ratings and impact summaries.
    """

    def __init__(self, news_analyzer: NewsAnalyzer | None = None) -> None:
        """Initializes the morning news analyzer with a shared NewsAnalyzer client."""
        self.news_analyzer = news_analyzer or NewsAnalyzer()

    def analyze_overnight_news(self, ticker: str) -> Dict[str, Any]:
        """
        Gathers news stories and compiles a sentiment profile.
        Maps polarity scores (-1.0 to +1.0) to a standard 0 to 100 range.
        """
        logger.info("Running morning overnight news evaluation for '%s'...", ticker)
        try:
            # Fetch and score headlines via standard news service
            report = self.news_analyzer.analyze_news(ticker, max_articles=4)
            
            # Map polarity score (-1.0 to 1.0) to (0 to 100) range
            polarity = report.average_sentiment_score
            score = int((polarity + 1.0) * 50.0)
            
            # Extract key reasons from individual summaries or construct high-level points
            reasons = []
            for art in report.articles:
                if art.sentiment == SentimentEnum.POSITIVE and len(reasons) < 3:
                    reasons.append(f"Positive News: {art.headline} - {art.summary}")
                elif art.sentiment == SentimentEnum.NEGATIVE and len(reasons) < 3:
                    reasons.append(f"Negative News: {art.headline} - {art.summary}")
            
            if not reasons:
                reasons = ["No major corporate or news catalysts detected overnight."]

            # Formulate overall summary
            sentiment_class = report.overall_sentiment.value
            ai_summary = f"Overall overnight sentiment is {sentiment_class} with an average polarity score of {polarity:+.2f}."
            
            return {
                "Score": score,
                "Sentiment": sentiment_class,
                "Summary": ai_summary,
                "Reasons": reasons,
                "Raw_Articles_Count": len(report.articles)
            }

        except Exception as exc:
            logger.warning("Overnight news analysis failed for '%s': %s. Using neutral default profile.", ticker, str(exc))
            return {
                "Score": 50,
                "Sentiment": "NEUTRAL",
                "Summary": "Overnight news was unavailable or skipped during scheduling checks.",
                "Reasons": ["No active news indicators available."],
                "Raw_Articles_Count": 0
            }
