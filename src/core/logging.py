"""Structured logging configuration for the BFSI Policy Research Assistant.

Supports both human-friendly console output (via Rich or standard stream)
and JSON structured logging for production observability and LLMOps tracing.
"""

import logging
import sys
from typing import Any
from config.settings import settings


class Formatter(logging.Formatter):
    """Standardized log formatter with timestamp, level, and component scope."""

    def format(self, record: logging.LogRecord) -> str:
        # Standard readable formatting for development console
        record.component = getattr(record, "component", record.name)
        return super().format(record)


def setup_logger(name: str, level: str | None = None) -> logging.Logger:
    """Creates or configures a logger instance with consistent formatting.

    Args:
        name: Name of the module/logger.
        level: Optional log level override (e.g. DEBUG, INFO, WARNING).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Don't propagate to root logger to avoid double logging
    logger.propagate = False
    return logger


# Default application logger
app_logger = setup_logger("bfsi_rag")
