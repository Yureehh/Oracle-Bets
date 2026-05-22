"""
Logging utilities.

Creates namespaced loggers for each pipeline component with bounded rotating
file-handling and optional dependency-injection for tests.

Usage
-----
from oracle_bets_core.logger import LOG_TOPIC, instantiate_logger

log = instantiate_logger(LOG_TOPIC.SCHEDULE_GENERATION)
log.info("Hello, Oracle-Bets!")
"""

from __future__ import annotations

import logging
import os
from enum import Enum, unique
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv

from oracle_bets_core.paths import LOGS_DIR

if TYPE_CHECKING:
    from pathlib import Path

load_dotenv()

# --------------------------------------------------------------------------- #
# Constants / Enums
# --------------------------------------------------------------------------- #
ISO_TIME_FMT: Final = "%Y-%m-%dT%H:%M:%S"
DEFAULT_FORMAT: Final = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
DEFAULT_LOG_MAX_BYTES: Final = 5_000_000
DEFAULT_LOG_BACKUPS: Final = 3


# --------------------------------------------------------------------------- #
# Enum of topics
# --------------------------------------------------------------------------- #
@unique
class LOG_TOPIC(Enum):  # noqa: N801
    """Pre-defined logger namespaces (extend as needed)."""

    DATA_PIPELINE = "DataPipelineLogger"
    MODEL_TRAINING = "ModelTrainingLogger"
    ORACLE_BOT = "OracleBotLogger"
    SCHEDULE_GENERATION = "ScheduleGenerationLogger"
    GENERAL = "GeneralLogger"

    def __str__(self) -> str:
        return self.value


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class LogConfigurationError(RuntimeError):
    """Raised when the logger can’t be configured as requested."""


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_handler(
    *,
    log_file: Path | None,
    fmt: str,
    mode: str,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUPS,
) -> logging.Handler:
    formatter = logging.Formatter(fmt, datefmt=ISO_TIME_FMT)
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = RotatingFileHandler(
                log_file,
                mode=mode,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        except OSError as exc:  # permission denied, etc.
            msg = f"Cannot open log file '{log_file}': {exc}"
            raise LogConfigurationError(msg) from exc
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler


# --------------------------------------------------------------------------- #
# Public factory helpers
# --------------------------------------------------------------------------- #
def create_logger(
    name: str | LOG_TOPIC,
    *,
    log_file: Path | None = None,
    level: int | str = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    mode: str = "a",
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUPS,
) -> logging.Logger:
    """
    Build (or fetch) a configured logger.
    """
    logger_name = name if isinstance(name, str) else str(name)
    lg = logging.getLogger(logger_name)

    if not lg.handlers:  # avoid duplicate handlers
        if isinstance(level, str):
            level_val = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        else:
            level_val = level
        lg.setLevel(level_val)
        lg.propagate = False
        lg.addHandler(
            _build_handler(
                log_file=log_file,
                fmt=fmt,
                mode=mode,
                max_bytes=max_bytes,
                backup_count=backup_count,
            )
        )

    return lg


def instantiate_logger(
    topic: LOG_TOPIC, level: int | str = logging.INFO
) -> logging.Logger:
    """
    Convenience helper – writes to one stable rotating file per topic.
    """
    if isinstance(level, str):
        effective_level: int | str = level
    else:
        effective_level = os.getenv(
            "ORACLE_BETS_LOG_LEVEL", logging.getLevelName(level)
        )

    log_to_file = _env_flag("ORACLE_BETS_LOG_TO_FILE", default=True)
    log_file = LOGS_DIR / f"{str(topic).lower()}.log" if log_to_file else None
    return create_logger(
        topic,
        log_file=log_file,
        level=effective_level,
        max_bytes=_env_int("ORACLE_BETS_LOG_MAX_BYTES", default=DEFAULT_LOG_MAX_BYTES),
        backup_count=_env_int("ORACLE_BETS_LOG_BACKUPS", default=DEFAULT_LOG_BACKUPS),
    )


# --------------------------------------------------------------------------- #
# Console-only general logger (no file output)
# --------------------------------------------------------------------------- #
logger: logging.Logger = create_logger(LOG_TOPIC.GENERAL)
logger.debug("Logging subsystem initialised.")
