"""
Logging utilities for the project.

This module provides a `create_logger` function to create and manage custom loggers
for different components of the pipeline, with timestamped log files.
"""

import datetime as dt
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

from src.utils.paths import LOGS_DIR

# Load environment variables from .env file
load_dotenv()

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

# Predefined logger names for various components
PREDEFINED_LOGGERS = {
    "data_pipeline": "DataPipelineLogger",
    "model_training": "ModelTrainingLogger",
    "oracle_bot": "OracleBotLogger",
    "schedule_generation": "ScheduleGenerationLogger",
}

# Generate a timestamp to uniquely identify log files
TIMESTAMP_STR = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
    mode: str = "a",
) -> logging.Logger:
    """
    Create a custom logger with either a file handler or console handler.

    Args:
        name (str): The name of the logger.
        log_file (Optional[str]): The file path for the log file. If None, logs to console.
        level (int): The logging level.
        format_string (Optional[str]): The logging format string.
        mode (str): The file mode (default is 'a' for append).

    Returns:
        logging.Logger: A configured logger.

    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers to the same logger
    if not logger.hasHandlers():
        logger.setLevel(level)
        logger.propagate = False  # Prevent logs from propagating to the root logger

        # Use the provided format or default format
        format_string = format_string or DEFAULT_LOG_FORMAT
        formatter = logging.Formatter(format_string)

        # Add appropriate handler
        if log_file:
            try:
                handler = logging.FileHandler(log_file, mode=mode)
            except OSError as e:
                msg = f"Failed to create or access log file at '{log_file}': {e}"
                raise ValueError(msg) from e
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@dataclass
class ConfigurableLogger:
    """Dataclass for creating and managing a configurable logger."""

    name: str
    log_file: str | None = None
    level: int = logging.INFO
    format_string: str = DEFAULT_LOG_FORMAT
    mode: str = "a"
    logger: logging.Logger | None = None

    def __post_init__(self):
        """Initialize the logger instance."""
        self.logger = create_logger(
            self.name, self.log_file, self.level, self.format_string, self.mode
        )

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger.

        Returns:
            logging.Logger: The configured logger instance.

        """
        return self.logger


def instantiate_conf_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Instantiate a logger with the given parameters.

    Args:
        name (str): The name of the logger.
        log_file (Optional[str]): The file path for the log file. If None, logs to console.
        level (int): The logging level.

    Returns:
        logging.Logger: The configured logger instance.

    """
    return ConfigurableLogger(
        PREDEFINED_LOGGERS.get(name, name),
        log_file=LOGS_DIR / f"{TIMESTAMP_STR}_{name}.log",
        level=level,
    ).get_logger()


# Instantiate loggers for specific pipeline components, adding timestamp to each log file name
logger = ConfigurableLogger("GeneralLogger").get_logger()
logger.info("Loggers initialized.")
