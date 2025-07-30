"""
Logging utilities.

Creates namespaced, timestamped loggers for each pipeline component with safe
file-handling and optional dependency-injection for tests.

Usage
-----
from utils.logger import instantiate_logger, LOG_TOPIC

log = instantiate_logger(LOG_TOPIC.SCHEDULE_GENERATION)
log.info("Hello, Oracle-Bets!")
"""

from __future__ import annotations

import datetime as dt
import logging
from enum import Enum, unique
from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv

from utils.paths import LOGS_DIR

if TYPE_CHECKING:
    from pathlib import Path

load_dotenv()  # ensures BASE_DIR is resolved before LOGS_DIR is imported


# --------------------------------------------------------------------------- #
# Constants / Enums
# --------------------------------------------------------------------------- #
ISO_TIME_FMT: Final = "%Y-%m-%dT%H:%M:%S"
DEFAULT_FORMAT: Final = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"

# Timestamp used in all file names for the current interpreter session
_TS: Final = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


@unique
class LOG_TOPIC(Enum):
    """Pre-defined logger namespaces (extend as needed)."""

    DATA_PIPELINE = "DataPipelineLogger"
    MODEL_TRAINING = "ModelTrainingLogger"
    ORACLE_BOT = "OracleBotLogger"
    SCHEDULE_GENERATION = "ScheduleGenerationLogger"
    GENERAL = "GeneralLogger"

    def __str__(self) -> str:  # for nice `str(LOG_TOPIC.SCHEDULE_GENERATION)`
        return self.value


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class LogConfigurationError(RuntimeError):
    """Raised when a logger cannot be configured as requested."""


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def _build_handler(*, log_file: Path | None, fmt: str, mode: str) -> logging.Handler:
    """Return a `FileHandler` or `StreamHandler` with the given formatter."""
    formatter = logging.Formatter(fmt, datefmt=ISO_TIME_FMT)
    if log_file:
        try:
            handler: logging.Handler = logging.FileHandler(
                log_file, mode=mode, encoding="utf-8"
            )
        except OSError as exc:  # e.g. permission denied
            msg = f"Cannot open log file '{log_file}': {exc}"
            raise LogConfigurationError(msg) from exc
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler


def create_logger(
    name: str | LOG_TOPIC,
    *,
    log_file: Path | None = None,
    level: int | str = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    mode: str = "a",
) -> logging.Logger:
    """
    Build (or fetch) a configured logger.

    Parameters
    ----------
    name:
        Logger name or `LOG_TOPIC` enum value.
    log_file:
        Full path to log file.  If omitted, logs stream to stdout/stderr.
    level:
        Logging level (int or `"DEBUG"`, `"INFO"`, …).
    fmt:
        Format string for the log records.
    mode:
        File mode when using a file handler.

    Returns
    -------
    logging.Logger

    """
    logger_name = str(name) if not isinstance(name, str) else name
    logger = logging.getLogger(logger_name)

    # Avoid duplicated handlers in interactive contexts / tests
    if not logger.handlers:
        logger.setLevel(
            level if isinstance(level, int) else logging.getLevelName(level)
        )
        logger.propagate = False  # don’t bubble to root

        handler = _build_handler(log_file=log_file, fmt=fmt, mode=mode)
        logger.addHandler(handler)

    return logger


def instantiate_logger(
    topic: LOG_TOPIC, *, level: int | str = logging.INFO
) -> logging.Logger:
    """
    Convenience helper – always writes to a timestamped file in ``LOGS_DIR``.
    """
    file_path = LOGS_DIR / f"{_TS}_{topic.name.lower()}.log"
    return create_logger(topic, log_file=file_path, level=level)


# --------------------------------------------------------------------------- #
# Eagerly create a general-purpose logger so modules can ``from utils.logger import logger``.
# --------------------------------------------------------------------------- #
logger: logging.Logger = instantiate_logger(LOG_TOPIC.GENERAL)
logger.debug("Logging subsystem initialised.")
