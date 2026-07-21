import pytest
from ai_stock_advisor.core.exceptions import (
    StockAdvisorError,
    InvalidTickerError,
    LLMServiceError,
)


def test_base_exception_stores_attributes() -> None:
    """Ensure base domain exception captures messages and detail payloads."""
    payload = {"reason": "rate limit reached", "retry_after": 60}
    with pytest.raises(StockAdvisorError) as exc_info:
        raise StockAdvisorError("Test base error message", details=payload)
        
    assert exc_info.value.message == "Test base error message"
    assert exc_info.value.details == payload
    assert str(exc_info.value) == "Test base error message"


def test_invalid_ticker_exception_captures_symbol() -> None:
    """Ensure InvalidTickerError stores ticker and builds matching payload."""
    with pytest.raises(InvalidTickerError) as exc_info:
        raise InvalidTickerError("XYZ", "Ticker XYZ not found")
        
    assert exc_info.value.ticker == "XYZ"
    assert exc_info.value.details == {"ticker": "XYZ"}


def test_subclass_relationships() -> None:
    """Verify standard exception hierarchy relationships."""
    assert issubclass(LLMServiceError, StockAdvisorError)
    assert issubclass(InvalidTickerError, StockAdvisorError)
