import datetime as dt
from os import getenv

from dotenv import load_dotenv

from ingestion.schedule import PandaScoreSchedule
from utils.logger import logger
from utils.paths import SCHEDULE

# Load environment variables from .env file
load_dotenv()

# Constants
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MAX_DAYS_RANGE = 7
PANDASCORE_API_KEY_ENV = "PANDASCORE_API_KEY"


def main():
    """Main function to fetch and store the schedule data."""
    try:
        # Get the current date and time, and calculate the end date
        start = dt.datetime.now().strftime(TIME_FORMAT)
        end = (dt.datetime.now() + dt.timedelta(days=MAX_DAYS_RANGE)).strftime(TIME_FORMAT)
        logger.info(f"Fetching schedule from {start} to {end}")

        # Fetch the schedule from PandaScore API
        api_key = getenv(PANDASCORE_API_KEY_ENV)
        if not api_key:
            raise ValueError("PandaScore API key is not set in environment variables.")

        panda_schedule = PandaScoreSchedule(api_key=api_key)
        schedule = panda_schedule.get_schedule(
            start_datetime=start, end_datetime=end, max_day_range=MAX_DAYS_RANGE, time_format=TIME_FORMAT
        )

        # Export the schedule to a parquet file
        schedule.to_parquet(SCHEDULE, index=False)
        logger.info(f"Schedule stored to {SCHEDULE}")

    except Exception as e:
        error_message = f"An error occurred during the schedule fetching process: {e}"
        logger.error(error_message)
        raise


if __name__ == "__main__":
    main()
