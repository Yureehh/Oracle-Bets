"""
Schedule Data Ingestion.

This script fetches the schedule of upcoming matches from the PandaScore API
using a Pandas-like interface from `fireducks.pandas`.

Please visit and support www.pandascore.com
"""

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union

# Replace polars with fireducks.pandas (identical to pandas syntax)
import fireducks.pandas as pd
import requests
from dateutil import parser
from dotenv import load_dotenv

from src.utils.logger import instantiate_conf_logger, logger

# Load environment variables from .env file
load_dotenv()

# Constants
PANDASCORE_BASE_URL = "https://api.pandascore.co/lol/matches/upcoming"
ACCEPT_JSON_HEADER = {"Accept": "application/json"}
PER_PAGE = 100
START_DATETIME_COLUMN = "Start (UTC)"
schedule_generation_logger = instantiate_conf_logger("schedule_generation")


@dataclass
class PandaScoreSchedule:
    """Schedule Data Ingestion from the PandaScore API."""

    api_key: str
    headers: dict = field(default_factory=lambda: ACCEPT_JSON_HEADER)
    base_url: str = field(default_factory=lambda: PANDASCORE_BASE_URL)

    def _fetch_matches(self, page: int) -> List[dict]:
        """
        Fetch matches from the PandaScore API for a specific page.

        Args:
            page (int): The page number to fetch.

        Returns:
            List[dict]: List of match data dictionaries.
        """
        params = {
            "sort": "",
            "page": page,
            "per_page": PER_PAGE,
            "token": self.api_key,
        }
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            logger.info("Successful PandaScore API request for page: %s", page)
            schedule_generation_logger.info("Successful PandaScore API request for page: %s", page)
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTPError during API request: {e}")
            schedule_generation_logger.error(f"HTTPError during API request: {e}")
            return []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            schedule_generation_logger.error(f"Failed to fetch data: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Exception during API request: {e}")
            schedule_generation_logger.error(f"Exception during API request: {e}")
            return []

    def _fetch_all_matches(self) -> Iterator[List[dict]]:
        """
        Fetches all matches across pages.

        Yields:
            Iterator[List[dict]]: Iterator of match data lists.
        """
        page = 1
        while True:
            matches = self._fetch_matches(page)
            if not matches:
                if page == 1:
                    logger.error("No matches found or failed to fetch matches.")
                    schedule_generation_logger.error("No matches found or failed to fetch matches.")
                break

            yield matches
            logger.info(f"Fetched {len(matches)} matches from page {page}.")
            schedule_generation_logger.info(f"Fetched {len(matches)} matches from page {page}.")
            page += 1

            # Adjust the sleep duration based on API guidelines
            time.sleep(1)

    @staticmethod
    def _parse_matches_response(matches: List[dict]) -> pd.DataFrame:
        """
        Parse and structure matches data from the API response into a DataFrame.

        Args:
            matches (List[dict]): List of match data dictionaries.

        Returns:
            pd.DataFrame: DataFrame containing parsed match data.
        """
        data = []
        for match in matches:
            try:
                scheduled_at = match.get("scheduled_at")
                if not scheduled_at:
                    continue

                opponents = match.get("opponents", [])
                blue_team = opponents[0].get("opponent", {}).get("name", "N/A") if len(opponents) > 0 else "N/A"
                red_team = opponents[1].get("opponent", {}).get("name", "N/A") if len(opponents) > 1 else "N/A"

                data.append(
                    {
                        "match_id": match.get("id", "N/A"),
                        "league": match.get("league", {}).get("name", "N/A"),
                        "Blue": blue_team,
                        "Red": red_team,
                        START_DATETIME_COLUMN: scheduled_at or "N/A",
                        "Best Of": match.get("number_of_games", "N/A"),
                    }
                )
            except KeyError as e:
                logger.warning(f"Missing expected data in match: {e}")
                schedule_generation_logger.warning(f"Missing expected data in match: {e}")
            except Exception as e:
                logger.error(f"Failed to parse match: {e}")
                schedule_generation_logger.error(f"Failed to parse match: {e}")

        if not data:
            # Return empty DataFrame with the expected columns
            return pd.DataFrame(
                columns=[
                    "match_id",
                    "league",
                    "Blue",
                    "Red",
                    START_DATETIME_COLUMN,
                    "Best Of",
                ]
            )

        return pd.DataFrame(data)

    @staticmethod
    def filter_by_league(schedule: pd.DataFrame, leagues: str) -> pd.DataFrame:
        """
        Filter the schedule by specified leagues.

        Args:
            schedule (pd.DataFrame): DataFrame containing match schedules.
            leagues (str): Comma-separated string of league names to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only specified leagues.
        """
        if "league" not in schedule.columns:
            logger.error("The schedule DataFrame does not contain a 'league' column.")
            schedule_generation_logger.error("The schedule DataFrame does not contain a 'league' column.")
            return pd.DataFrame()

        league_list = [lg.strip().lower() for lg in leagues.split(",")]
        # Convert league column to lowercase
        schedule["league"] = schedule["league"].astype(str).str.strip().str.lower()

        filtered_schedule = schedule[schedule["league"].isin(league_list)]
        logger.info("Filtered schedule by leagues: %s", league_list)
        schedule_generation_logger.info("Filtered schedule by leagues: %s", league_list)
        return filtered_schedule

    @staticmethod
    def load_schedule(schedule_path: str, leagues: Optional[str] = None) -> pd.DataFrame:
        """
        Load the schedule from a Parquet file and optionally filter by leagues.

        Args:
            schedule_path (str): Path to the parquet file containing the schedule.
            leagues (Optional[str]): Comma-separated leagues to filter. Defaults to None.

        Returns:
            pd.DataFrame: Loaded and optionally filtered schedule DataFrame.
        """
        try:
            # Using the same syntax as pandas
            schedule_df = pd.read_parquet(schedule_path)
            logger.info("Loaded schedule from %s", schedule_path)
            schedule_generation_logger.info("Loaded schedule from %s", schedule_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {schedule_path} - {e}")
            schedule_generation_logger.error(f"File not found: {schedule_path} - {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading schedule: {e}")
            schedule_generation_logger.error(f"Error loading schedule: {e}")
            return pd.DataFrame()

        if leagues:
            schedule_df = PandaScoreSchedule.filter_by_league(schedule_df, leagues)

        return schedule_df

    def get_schedule(
        self,
        start_datetime: Union[str, dt.datetime],
        end_datetime: Union[str, dt.datetime],
        max_day_range: int,
        time_format: str,
        leagues: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieves schedule of upcoming matches within a specified datetime range.

        Args:
            start_datetime (Union[str, dt.datetime]): Start datetime.
            end_datetime (Union[str, dt.datetime]): End datetime.
            max_day_range (int): Maximum allowed range in days between start and end datetime.
            time_format (str): Format of the datetime strings for parsing.
            leagues (Optional[str]): Comma-separated leagues to filter. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the schedule of upcoming matches.
        """
        start_dt, end_dt = self._validate_and_parse_dates(start_datetime, end_datetime, max_day_range)
        schedule_df = pd.DataFrame()

        for matches in self._fetch_all_matches():
            parsed_matches = self._parse_and_filter_matches(matches, start_dt, end_dt, time_format)
            if parsed_matches.empty:
                continue

            schedule_df = self._append_to_schedule(schedule_df, parsed_matches)

            # If the earliest match is after the end date, stop further API calls
            if self._should_stop_fetching(parsed_matches, end_dt):
                break

        # Filter by leagues if specified
        if leagues:
            schedule_df = self.filter_by_league(schedule_df, leagues)

        logger.info(f"Completed fetching and processing the schedule. Total matches collected: {schedule_df.shape[0]}")
        schedule_generation_logger.info(
            f"Completed fetching and processing the schedule. Total matches collected: {schedule_df.shape[0]}"
        )
        return schedule_df

    def _validate_and_parse_dates(
        self,
        start_datetime: Union[str, dt.datetime],
        end_datetime: Union[str, dt.datetime],
        max_day_range: int,
    ) -> tuple[dt.datetime, dt.datetime]:
        """
        Validates and parses start and end datetime inputs.

        Args:
            start_datetime (Union[str, dt.datetime]): Start datetime.
            end_datetime (Union[str, dt.datetime]): End datetime.
            max_day_range (int): Maximum allowed range in days.

        Returns:
            tuple[dt.datetime, dt.datetime]: Parsed and validated start and end datetimes.
        """
        start_dt = self._parse_datetime(start_datetime)
        end_dt = self._parse_datetime(end_datetime)

        if start_dt > end_dt:
            raise ValueError("Start datetime must be before end datetime.")

        if (end_dt - start_dt).days > max_day_range:
            raise ValueError(f"Time range exceeds the maximum allowed {max_day_range} days.")

        return start_dt, end_dt

    def _parse_datetime(self, datetime_input: Union[str, dt.datetime]) -> dt.datetime:
        """
        Parses a datetime input, ensuring it is in UTC.

        Args:
            datetime_input (Union[str, dt.datetime]): Input datetime.

        Returns:
            dt.datetime: UTC datetime object.
        """
        if isinstance(datetime_input, dt.datetime):
            parsed_dt = datetime_input
        else:
            parsed_dt = parser.isoparse(datetime_input)

        return parsed_dt.astimezone(dt.timezone.utc)

    def _parse_and_filter_matches(
        self,
        matches: List[dict],
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        time_format: str,
    ) -> pd.DataFrame:
        """
        Parses and filters matches within the datetime range.

        Args:
            matches (List[dict]): List of match data.
            start_dt (dt.datetime): Start datetime.
            end_dt (dt.datetime): End datetime.
            time_format (str): Time format string for parsing.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        parsed_matches = self._parse_matches_response(matches)
        if parsed_matches.empty:
            return parsed_matches

        # Convert scheduled column to datetime, drop invalid rows
        parsed_matches[START_DATETIME_COLUMN] = pd.to_datetime(
            parsed_matches[START_DATETIME_COLUMN], format=time_format, errors="coerce"  # any invalid parse becomes NaT
        )
        parsed_matches.dropna(subset=[START_DATETIME_COLUMN], inplace=True)

        # Filter rows by time range
        parsed_matches = parsed_matches[
            (parsed_matches[START_DATETIME_COLUMN] >= start_dt.replace(tzinfo=None))
            & (parsed_matches[START_DATETIME_COLUMN] <= end_dt.replace(tzinfo=None))
        ]

        return parsed_matches

    def _append_to_schedule(self, schedule_df: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Appends new data to the schedule, ensuring no duplicates by 'match_id'.

        Args:
            schedule_df (pd.DataFrame): Existing schedule DataFrame.
            new_data (pd.DataFrame): New data to append.

        Returns:
            pd.DataFrame: Updated schedule DataFrame.
        """
        if schedule_df.empty:
            # Just remove duplicates from new_data (if any) before returning
            return new_data.drop_duplicates(subset=["match_id"])

        combined_df = pd.concat([schedule_df, new_data], axis=0)
        combined_df = combined_df.drop_duplicates(subset=["match_id"])
        return combined_df

    def _should_stop_fetching(self, parsed_matches: pd.DataFrame, end_dt: dt.datetime) -> bool:
        """
        Determines whether to stop fetching more data.

        Args:
            parsed_matches (pd.DataFrame): Parsed matches DataFrame.
            end_dt (dt.datetime): End datetime.

        Returns:
            bool: True if fetching should stop, False otherwise.
        """
        if parsed_matches.empty:
            return False
        earliest_match = parsed_matches[START_DATETIME_COLUMN].min()
        return earliest_match and earliest_match > end_dt.replace(tzinfo=None)
