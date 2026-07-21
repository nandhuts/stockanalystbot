"""
Configuration Package.
Contains application setting declarations, environment bindings, and logging configs.
"""
from config.settings import settings
from config.logging_config import setup_logging

__all__ = ["settings", "setup_logging"]
