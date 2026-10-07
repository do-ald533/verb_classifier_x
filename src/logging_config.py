from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

LOG_DIR = Path("logs")


def get_logger(name: str) -> logging.Logger:
    """Pipeline mechanics only (trial starts, pruning events, retries).
    Results (metrics/params/artifacts) belong in MLflow, not here — kept
    deliberately separate (plan §9)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console_handler: logging.Handler
    if _HAS_RICH:
        console_handler = RichHandler(rich_tracebacks=True)
    else:
        console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(LOG_DIR / "pipeline.log", maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
