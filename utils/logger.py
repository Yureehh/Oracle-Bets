import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Logger:
    name: str = os.getenv("LOGGER_NAME", __name__)
    date_format: str = "%m/%d/%Y %I:%M:%S %p"

    def __post_init__(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", datefmt=self.date_format
        )
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Logger initialized.")

    def get_logger(self):
        return self.logger


# Usage
logger = Logger().get_logger()
