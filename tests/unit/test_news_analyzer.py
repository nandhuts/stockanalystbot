from unittest.mock import MagicMock, patch
import pytest

from ai_stock_advisor.core.exceptions import LLMServiceError
from ai_stock_advisor.services.llm.news_analyzer import (
    NewsAnalyzer,
    NewsSentimentAnalysis,
    SentimentEnum,
    StockNewsReport,
)


@pytest.fixture
def mock_yf_news() -> list:
    """Mock yfinance raw news feed response."""
    return [
        {
            "title": "Reliance Industries posts strong Q1 profit growth",
            "publisher": "Reuters",
            "link": "http://reuters.com/rel1",
            "providerPublishTime": 1782000000,
            "type": "STORY"
        },
        {
            "title": "Market consolidation ahead, warns analyst",
            "publisher": "CNBC",
            "link": "http://cnbc.com/rel2",
            "providerPublishTime": 1782005000,
            "type": "STORY"
        }
    ]


@pytest.fixture
def expected_news_report() -> StockNewsReport:
    """Mock report output matching target schema."""
    return StockNewsReport(
        articles=[
            NewsSentimentAnalysis(
                headline="Reliance Industries posts strong Q1 profit growth",
                summary="Reliance posted positive profit metrics, boosting investor interest.",
                sentiment=SentimentEnum.POSITIVE,
                sentiment_score=0.8
            ),
            NewsSentimentAnalysis(
                headline="Market consolidation ahead, warns analyst",
                summary="Analysts expect short-term stock trading range consolidation.",
                sentiment=SentimentEnum.NEUTRAL,
                sentiment_score=0.0
            )
        ],
        overall_sentiment=SentimentEnum.POSITIVE,
        average_sentiment_score=0.4
    )


def test_fetch_news_headlines_parsing(mock_yf_news: list) -> None:
    """Ensure fetch_news_headlines parses raw yfinance list correctly."""
    analyzer = NewsAnalyzer(api_key="test_key")
    
    mock_ticker = MagicMock()
    mock_ticker.news = mock_yf_news
    
    with patch("yfinance.Ticker", return_value=mock_ticker):
        headlines = analyzer.fetch_news_headlines("RELIANCE.NS", max_articles=2)
        
        assert len(headlines) == 2
        assert headlines[0]["title"] == "Reliance Industries posts strong Q1 profit growth"
        assert headlines[0]["publisher"] == "Reuters"
        assert headlines[0]["link"] == "http://reuters.com/rel1"


def test_analyze_news_success(
    mock_yf_news: list,
    expected_news_report: StockNewsReport,
) -> None:
    """Verify analyze_news runs OpenAI query and parses structured output report."""
    analyzer = NewsAnalyzer(api_key="test_key")
    
    # Mock yfinance news fetch
    mock_ticker = MagicMock()
    mock_ticker.news = mock_yf_news
    
    # Mock OpenAI completions.parse response
    mock_parsed_choice = MagicMock()
    mock_parsed_choice.message.parsed = expected_news_report
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_parsed_choice]

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with patch.object(analyzer.client.beta.chat.completions, "parse", return_value=mock_completion) as mock_parse:
            report = analyzer.analyze_news("RELIANCE.NS", max_articles=2)
            
            assert isinstance(report, StockNewsReport)
            assert report.overall_sentiment == SentimentEnum.POSITIVE
            assert report.average_sentiment_score == 0.4
            assert len(report.articles) == 2
            assert report.articles[0].sentiment_score == 0.8
            assert "Q1 profit growth" in report.articles[0].headline
            
            mock_parse.assert_called_once()


def test_analyze_news_api_failure_raises_custom_exception(mock_yf_news: list) -> None:
    """Verify analyze_news translates OpenAI exceptions to LLMServiceError."""
    analyzer = NewsAnalyzer(api_key="test_key")
    
    # Mock yfinance news fetch
    mock_ticker = MagicMock()
    mock_ticker.news = mock_yf_news
    
    with patch("yfinance.Ticker", return_value=mock_ticker):
        with patch.object(analyzer.client.beta.chat.completions, "parse", side_effect=Exception("API connection timeout")):
            with pytest.raises(LLMServiceError) as exc_info:
                analyzer.analyze_news("RELIANCE.NS", max_articles=2)
                
            assert "Failed generating AI news sentiment analysis" in exc_info.value.message
            assert exc_info.value.details["ticker"] == "RELIANCE.NS"
