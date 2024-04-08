"""
Logger

This script defines a dataclass for initializing a basic logger with global settings.
This also includes dataclasses for initializing loggers for early game metrics imputer and models.
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.paths import EARLY_GAME_INPUTING_LOGS, MODELS_LOGS

load_dotenv()


def initialize_basic_logger(name: str, date_format: str = "%m/%d/%Y %I:%M:%S %p") -> logging.Logger:
    """
    Initialize a basic logger with global settings.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt=date_format,
        )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def add_file_handler(logger: logging.Logger, log_file: str, formatter: logging.Formatter, level: int = logging.DEBUG):
    """
    Add a file handler to the logger.
    """
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    logger.propagate = False


@dataclass
class Logger:
    """
    Dataclass for initializing the basic logger.
    """

    name: str = os.getenv("LOGGER_NAME", __name__)
    date_format: str = "%m/%d/%Y %I:%M:%S %p"

    def __post_init__(self):
        self.logger = initialize_basic_logger(self.name, self.date_format)

    def get_logger(self) -> logging.Logger:
        return self.logger


@dataclass
class EarlyGameMetricsImputerLogger(Logger):
    """
    Dataclass for initializing the logger for early game metrics imputer.
    """

    name: str = os.getenv("EARLY_GAME_METRICS_IMPUTER_LOGGER_NAME", "EarlyGameMetricsImputerLogger")

    def __post_init__(self):
        super().__post_init__()
        formatter = logging.Formatter("%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", "%H:%M:%S")
        add_file_handler(self.logger, EARLY_GAME_INPUTING_LOGS, formatter)


@dataclass
class ModelsLogger(Logger):
    """ "
    Dataclass for initializing the logger for models.
    """

    name: str = os.getenv("MODELS_LOGGER_NAME", "ModelsLogger")

    def __post_init__(self):
        super().__post_init__()
        formatter = logging.Formatter("%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", "%H:%M:%S")
        add_file_handler(self.logger, MODELS_LOGS, formatter)


# Usage example
logger = Logger().get_logger()
early_game_metrics_imputer_logger = EarlyGameMetricsImputerLogger().get_logger()
models_logger = ModelsLogger().get_logger()

logger.info("Loggers initialized.\n")
