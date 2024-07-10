"""
Logging utilities for the project.

This module provides a create_logger function that creates a custom logger with either a file handler or console handler.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from src.utils.paths import MODELS_LOGS

# Load environment variables from .env file
load_dotenv()


def create_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    mode: str = "a",
) -> logging.Logger:
    """
    Create a custom logger with either a file handler or console handler.

    Parameters:
        name (str): The name of the logger.
        log_file (Optional[str]): The file path for the log file. If None, logs to console.
        level (int): The logging level.
        format_string (Optional[str]): The logging format string.
        mode (str): The file mode (default is 'a' for append).

    Returns:
        logging.Logger: A configured logger.
    """
    format_string = format_string or "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    formatter = logging.Formatter(format_string)

    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        logger.propagate = False

        if log_file:
            handler = logging.FileHandler(log_file, mode=mode)
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@dataclass
class ConfigurableLogger:
    """
    Dataclass for initializing loggers with different configurations.
    """

    name: str
    log_file: Optional[str] = None
    level: int = logging.INFO
    format_string: str = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    mode: str = "a"

    def __post_init__(self):
        self.logger = create_logger(self.name, self.log_file, self.level, self.format_string, self.mode)

    def get_logger(self) -> logging.Logger:
        return self.logger


# Initialize loggers with various configurations
logger = ConfigurableLogger("GeneralLogger").get_logger()
models_logger = ConfigurableLogger(
    "ModelsLogger", MODELS_LOGS, format_string="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
).get_logger()

logger.info("Loggers initialized.")
