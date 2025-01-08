"""
PandaScore Schedule Fetcher.

This script fetches and stores the schedule data from the PandaScore API.
It retrieves the schedule for a specified range of days and exports it to a Parquet file.
"""

import datetime as dt
import os
import sys
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from ingestion.schedule import PandaScoreSchedule, schedule_generation_logger
from src.utils.logger import logger
from src.utils.paths import SCHEDULE
from src.utils.utils import safe_store_df_as_parquet

# Constants
TIME_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"
MAX_DAYS_RANGE: int = 7
PANDASCORE_API_KEY_ENV: str = "PANDASCORE_API_KEY"


def get_api_key_from_env(env_key: str = PANDASCORE_API_KEY_ENV) -> str:
    """
    Retrieves the PandaScore API key from environment variables.

    Args:
        env_key (str): Name of the environment variable that holds the PandaScore API key.

    Returns:
        str: The PandaScore API key.

    Raises:
        ValueError: If the environment variable is not set or empty.
    """
    api_key = os.getenv(env_key, "")
    if not api_key:
        error_message = f"PandaScore API key is not set in environment variables ({env_key})."
        schedule_generation_logger.error(error_message)
        raise ValueError(error_message)
    return api_key.strip()


def fetch_schedule_data(
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
    api_key: str,
    max_day_range: int = MAX_DAYS_RANGE,
    time_format: str = TIME_FORMAT,
) -> pd.DataFrame:
    """
    Fetch the schedule data from the PandaScore API for the specified time range.

    Args:
        start_datetime (dt.datetime): Start of the time window (UTC).
        end_datetime (dt.datetime): End of the time window (UTC).
        api_key (str): Valid PandaScore API key.
        max_day_range (int): Maximum range (in days) for a single API call.
        time_format (str): Desired datetime format string.

    Returns:
        pd.DataFrame: DataFrame containing schedule data.
    """
    logger.info("Initializing PandaScoreSchedule...")
    schedule_generation_logger.info("Initializing PandaScoreSchedule...")
    panda_schedule = PandaScoreSchedule(api_key=api_key)
    return panda_schedule.get_schedule(
        start_datetime=start_datetime, end_datetime=end_datetime, max_day_range=max_day_range, time_format=time_format
    )


def fetch_and_store_schedule(
    start_datetime: dt.datetime, end_datetime: dt.datetime, api_key: str, output_file_path: str = SCHEDULE
) -> None:
    """
    Fetches the schedule from the PandaScore API and stores it in a Parquet file.

    Args:
        start_datetime (dt.datetime): Start datetime (UTC).
        end_datetime (dt.datetime): End datetime (UTC).
        api_key (str): PandaScore API key.
        output_file_path (str): Path where fetched data will be stored as Parquet.
    """
    schedule_df = fetch_schedule_data(start_datetime, end_datetime, api_key)

    if schedule_df.empty:
        warning_message = "No schedule data retrieved."
        logger.warning(warning_message)
        schedule_generation_logger.warning(warning_message)
        return

    logger.info(f"Fetched {len(schedule_df)} matches from PandaScore.")
    schedule_generation_logger.info(f"Fetched {len(schedule_df)} matches from PandaScore.")
    schedule_generation_logger.info(f"Sample of the schedule data:\n{schedule_df.head()}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Export the schedule to a Parquet file
    safe_store_df_as_parquet(schedule_df, output_file_path, [logger, schedule_generation_logger])
    logger.info(f"Schedule successfully stored at {output_file_path}")
    schedule_generation_logger.info(f"Schedule successfully stored at {output_file_path}")


def main(start_datetime: Optional[dt.datetime] = None, end_datetime: Optional[dt.datetime] = None):
    """
    Main entry point for fetching and storing PandaScore schedule data.

    Steps:
    1. Load environment variables.
    2. Retrieve API key from environment.
    3. Set default date range if not provided.
    4. Fetch and store schedule data.

    Args:
        start_datetime (Optional[dt.datetime]): Start datetime (UTC). Defaults to current time.
        end_datetime (Optional[dt.datetime]): End datetime (UTC). Defaults to start + MAX_DAYS_RANGE days.
    """
    logger.info("Schedule Generation process initialized.")
    schedule_generation_logger.info("Schedule Generation process initialized.")
    load_dotenv()  # Load environment variables

    # Determine the date range if not provided
    if start_datetime is None:
        start_datetime = dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    if end_datetime is None:
        end_datetime = start_datetime + dt.timedelta(days=MAX_DAYS_RANGE)

    formatted_start = start_datetime.strftime(TIME_FORMAT)
    formatted_end = end_datetime.strftime(TIME_FORMAT)

    logger.info(f"Fetching schedule from {formatted_start} to {formatted_end}")
    schedule_generation_logger.info(f"Fetching schedule from {formatted_start} to {formatted_end}")

    # Retrieve API key
    try:
        api_key = get_api_key_from_env()
    except ValueError as ex:
        logger.error(str(ex))
        schedule_generation_logger.error(str(ex))
        sys.exit(1)

    # Fetch and store schedule
    try:
        fetch_and_store_schedule(start_datetime, end_datetime, api_key)
    except Exception as e:
        logger.exception(f"Error during schedule fetching: {e}")
        schedule_generation_logger.exception(f"Error during schedule fetching: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
