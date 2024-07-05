"""
Schedule Data Ingestion

This script fetches the schedule of upcoming matches from the PandaScore API.

Please visit and support www.pandascore.com
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()

# Constants
PANDASCORE_BASE_URL = "https://api.pandascore.co/lol/matches/upcoming"
ACCEPT_JSON_HEADER = {"Accept": "application/json"}
PER_PAGE = 100
START_STRING = "Start (UTC)"


@dataclass
class PandaScoreSchedule:
    api_key: str
    headers: dict = field(default_factory=lambda: ACCEPT_JSON_HEADER)
    base_url: str = field(default_factory=lambda: PANDASCORE_BASE_URL)

    def _fetch_matches(self, page: int) -> Optional[List[dict]]:
        """Fetch matches from the PandaScore API for a specific page."""
        params = {"sort": "", "page": page, "per_page": PER_PAGE, "token": self.api_key}
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            logger.info("Successful PandaScore API request for page: %s", page)
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTPError during API request: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Exception during API request: {e}")
            return None

    @staticmethod
    def _parse_matches_response(matches: List[dict]) -> pd.DataFrame:
        """Parse and structure matches data from the API response."""
        try:
            data = [
                {
                    "league": match.get("league", {}).get("name", "N/A"),
                    "Blue": match["opponents"][0]["opponent"]["name"] if match.get("opponents") else "N/A",
                    "Red": match["opponents"][1]["opponent"]["name"] if len(match.get("opponents", [])) > 1 else "N/A",
                    START_STRING: match["scheduled_at"],
                    "Best Of": match["number_of_games"],
                }
                for match in matches
                if match.get("scheduled_at")
            ]
            return pd.DataFrame(data)
        except IndexError:
            logger.warning(f"Invalid match format: {matches}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to parse matches: {e}")
            return pd.DataFrame()

    @staticmethod
    def filter_by_league(schedule: pd.DataFrame, leagues: str) -> pd.DataFrame:
        """Filter the schedule by leagues."""
        return schedule[schedule["league"].isin(leagues.split(","))].reset_index(drop=True)

    @staticmethod
    def load_schedule(schedule_path: str, leagues: Optional[str] = None) -> pd.DataFrame:
        """Load the schedule from a parquet file and optionally filter by leagues."""
        try:
            schedule_df = pd.read_parquet(schedule_path)
            if leagues:
                schedule_df = schedule_df[schedule_df["league"].isin(leagues.split(","))]
            return schedule_df
        except FileNotFoundError as e:
            logger.error(f"File not found: {schedule_path} - {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading schedule: {e}")
            return pd.DataFrame()

    def get_schedule(
        self,
        start_datetime: str,
        end_datetime: str,
        max_day_range: int,
        time_format: str,
        leagues: Optional[str] = None,
    ) -> pd.DataFrame:
        """Gets the schedule of upcoming matches."""
        start_datetime = pd.to_datetime(start_datetime).tz_convert("UTC")
        end_datetime = pd.to_datetime(end_datetime).tz_convert("UTC")
        if (end_datetime - start_datetime).days > max_day_range:
            raise ValueError(f"Time range exceeds the maximum allowed {max_day_range} days.")

        schedule_df = pd.DataFrame()
        page = 1

        while True:
            matches = self._fetch_matches(page)
            if not matches:
                if page == 1:
                    logger.error("No matches found or failed to fetch matches.")
                break

            parsed_matches = self._parse_matches_response(matches)
            if parsed_matches.empty:
                break

            parsed_matches[START_STRING] = pd.to_datetime(parsed_matches[START_STRING], format=time_format)
            if parsed_matches[START_STRING].dt.tz is None:
                parsed_matches[START_STRING] = parsed_matches[START_STRING].dt.tz_localize("UTC")

            within_range_df = parsed_matches[
                (parsed_matches[START_STRING] >= start_datetime) & (parsed_matches[START_STRING] <= end_datetime)
            ]

            if within_range_df.empty and parsed_matches[START_STRING].min() > end_datetime:
                break

            schedule_df = pd.concat([schedule_df, within_range_df], ignore_index=True)
            page += 1

        if leagues:
            schedule_df = self.filter_by_league(schedule_df, leagues)

        return schedule_df
