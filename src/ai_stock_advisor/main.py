import logging
import sys
from config.settings import settings
from config.logging_config import setup_logging
from ai_stock_advisor.core.exceptions import StockAdvisorError

# Initialize global logger
setup_logging()
logger = logging.getLogger("ai_stock_advisor.main")


def run_advisor(ticker: str) -> None:
    """
    Main orchestration step to execute advisor reasoning over a ticker.
    Stubs out flow since business logic is not yet implemented.
    """
    logger.info("Initializing analysis process for stock ticker: %s", ticker)
    
    # Configuration checks
    if not settings.GEMINI_API_KEY and settings.LLM_PROVIDER == "gemini":
        logger.warning("GEMINI_API_KEY is not defined. LLM calls will fail.")
        
    logger.debug("Active environment config: %s", settings.APP_ENV)
    
    # Business logic is stubbed out for architectural setup
    logger.info("Modular components are configured. Stock analysis logic is pending implementation.")


def main() -> None:
    """
    Application CLI Entrypoint.
    """
    logger.info("Starting AI Stock Advisor Application...")
    
    # Check arguments or default to a dummy analysis
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOG"
    
    try:
        run_advisor(ticker)
    except StockAdvisorError as err:
        logger.error("Core domain error encountered during execution: %s", err.message, exc_info=True)
        sys.exit(1)
    except Exception as err:
        logger.critical("Unexpected system failure: %s", str(err), exc_info=True)
        sys.exit(2)
        
    logger.info("AI Stock Advisor completed execution successfully.")


if __name__ == "__main__":
    main()
