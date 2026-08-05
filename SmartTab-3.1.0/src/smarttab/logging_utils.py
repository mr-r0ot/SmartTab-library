"""Central logger + verbose-level mapping for SmartTab."""

from __future__ import annotations

import logging

_LOGGER_NAME = "smarttab"

# fit(verbose=...) -> logging level
VERBOSE_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[smarttab] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def set_verbosity(verbose: int) -> None:
    level = VERBOSE_LEVELS.get(verbose, logging.INFO)
    get_logger().setLevel(level)
