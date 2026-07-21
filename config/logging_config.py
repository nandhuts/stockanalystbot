import logging
import logging.config
import os
from pathlib import Path
from config.settings import settings


def setup_logging() -> None:
    """
    Sets up the central logger hierarchy for the application.
    Supports structured formats, standard outputs, and log file rotation.
    """
    # Ensure the logs directory exists
    log_dir = Path(settings.BASE_DIR) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file_path = log_dir / "app.log"

    # Define standard formatters
    standard_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": standard_format,
                "datefmt": date_format,
            },
            "json_stub": {
                # JSON formatter stub - easily integrated in production with python-json-logger
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "file": "%(filename)s:%(lineno)d", "message": "%(message)s"}',
                "datefmt": date_format,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": settings.LOG_LEVEL,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "level": settings.LOG_LEVEL,
                "filename": str(log_file_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            }
        },
        "root": {
            "handlers": ["console", "file"],
            "level": settings.LOG_LEVEL,
        },
        "loggers": {
            "ai_stock_advisor": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": False,
            }
        }
    }

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized. Level: %s", settings.LOG_LEVEL)
