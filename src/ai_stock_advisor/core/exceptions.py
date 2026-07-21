class StockAdvisorError(Exception):
    """Base exception class for all AI Stock Advisor custom exceptions."""
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(StockAdvisorError):
    """Raised when environment variables or dynamic configurations are invalid."""
    pass


class ServiceError(StockAdvisorError):
    """Base exception for issues communicating with external services (LLM, Market Data, etc.)."""
    pass


class LLMServiceError(ServiceError):
    """Raised when interaction with LLM providers fails."""
    pass


class MarketDataServiceError(ServiceError):
    """Raised when interaction with financial data APIs fails."""
    pass


class InvalidTickerError(StockAdvisorError):
    """Raised when an operation is attempted with an unrecognized stock ticker symbol."""
    def __init__(self, ticker: str, message: str = "Invalid stock ticker symbol") -> None:
        super().__init__(message, details={"ticker": ticker})
        self.ticker = ticker
