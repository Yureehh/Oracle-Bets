import datetime as dt
import json
from dataclasses import dataclass, field
from os import getenv
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.logger import logger
from utils.paths import PROCESSED_DIR, RAW_DIR

load_dotenv()

# Constants
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
PANDASCORE_BASE_URL = "https://api.pandascore.co/lol/matches/upcoming"
ACCEPT_JSON_HEADER = {"Accept": "application/json"}
MAX_DAYS_RANGE = 7


@dataclass
class PandaScoreSchedule:
    api_key: str
    headers: dict = field(default_factory=lambda: ACCEPT_JSON_HEADER)
    base_url: str = field(default_factory=lambda: PANDASCORE_BASE_URL)

    def _fetch_matches(self, page: int) -> Optional[List[dict]]:
        """Fetch matches from the PandaScore API for a specific page."""
        params = {"sort": "", "page": page, "per_page": 100, "token": self.api_key}
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()  # Will raise HTTPError for bad requests
            logger.info("Successful PandaScore API request.")
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTPError during API request: {e}")
            return None
        except Exception as e:
            logger.error(f"Exception during API request: {e}")
            return None

    @staticmethod
    def _parse_matches_response(matches: List[dict]) -> List[dict]:
        """Parse and structure matches data from the API response."""
        parsed_matches = []
        for match in matches:
            try:
                match_data = {
                    "league": match["league"]["name"],
                    "Blue": match["opponents"][0]["opponent"]["name"],
                    "Red": match["opponents"][1]["opponent"]["name"],
                    "Start (UTC)": match["scheduled_at"],
                    "Best Of": match["number_of_games"],
                }
                parsed_matches.append(match_data)
            except IndexError:
                logger.warning(f"Invalid match format: {match}")
        return parsed_matches

    @staticmethod
    def filter_by_league(schedule: pd.DataFrame, leagues: str) -> pd.DataFrame:
        """Filter the schedule by leagues."""
        return schedule[schedule["league"].isin(leagues)].reset_index(drop=True)

    def get_schedule(
        self, start_datetime: str, end_datetime: str, leagues: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Gets the schedule of upcoming matches.

        Parameters
        ----------
        start_datetime : str
            The start datetime for the matches in 'YYYY-MM-DDTHH:MM:SSZ' format.
        end_datetime : str
            The end datetime for the matches in 'YYYY-MM-DDTHH:MM:SSZ' format.
        leagues : str, optional
            An optional string containing leagues of interest.
            Multiple leagues can be specified as a comma-separated string.
            ex: LCK, LPL, LEC

        Returns
        -------
        upcoming : pd.DataFrame
            A pandas DataFrame containing the upcoming matches.
        """
        # Establish Time Range
        start_datetime = pd.to_datetime(start_datetime).tz_convert("UTC")
        end_datetime = pd.to_datetime(end_datetime).tz_convert("UTC")
        if (end_datetime - start_datetime).days > MAX_DAYS_RANGE:
            raise ValueError(
                f"Time range exceeds the maximum allowed {MAX_DAYS_RANGE} days."
            )

        schedule_df = pd.DataFrame()
        page = 1

        # Paginate Data Within Time Range
        while True:
            matches = self._fetch_matches(page)
            if not matches:
                break  # No more data to fetch

            # Store matches in RAW_DIR
            with open(RAW_DIR / f"raw_schedule.json", "w") as file:
                json.dump(matches, file)

            parsed_matches = self._parse_matches_response(matches)
            page_matches_df = pd.DataFrame(parsed_matches)
            page_matches_df["Start (UTC)"] = pd.to_datetime(
                page_matches_df["Start (UTC)"], format=TIME_FORMAT
            )

            # Check if 'Start (UTC)' is timezone-naive and localize if necessary
            if page_matches_df["Start (UTC)"].dt.tz is None:
                page_matches_df["Start (UTC)"] = page_matches_df[
                    "Start (UTC)"
                ].dt.tz_localize("UTC")

            # Filter matches by datetime range
            within_range_df = page_matches_df[
                (page_matches_df["Start (UTC)"] >= start_datetime)
                & (page_matches_df["Start (UTC)"] <= end_datetime)
            ]
            if (
                within_range_df.empty
                and page_matches_df["Start (UTC)"].min() > end_datetime
            ):
                break  # All future data will be out of range
            schedule_df = pd.concat([schedule_df, within_range_df], ignore_index=True)
            page += 1

        # Filter by leagues, if specified
        if leagues:
            schedule_df = self.filter_by_league(schedule_df, leagues)

        return schedule_df


if __name__ == "__main__":
    start = dt.datetime.now().strftime(TIME_FORMAT)
    end = (dt.datetime.now() + dt.timedelta(days=3)).strftime(TIME_FORMAT)

    # Fetch
    panda_schedule = PandaScoreSchedule(api_key=getenv("PANDASCORE_API_KEY"))
    schedule = panda_schedule.get_schedule(start_datetime=start, end_datetime=end)

    # Export
    schedule.to_csv(PROCESSED_DIR / "schedule.csv", index=False)
