import datetime as dt
import json
from typing import List, Optional

import pandas as pd
import requests


class PandascoreSchedule:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Accept": "application/json"}

    def fetch_data(self, page: int) -> dict:
        """
        Fetches data from the Pandascore API.

        Parameters
        ----------
        page : int
            The page number to fetch from the API.

        Returns
        -------
        pandascore_response : dict
            The response from the API as a dictionary.
        """
        url = f"https://api.pandascore.co/lol/matches/upcoming?" \
              f"sort=&page={page}" \
              f"&per_page=100" \
              f"&token={self.api_key}"

        try:
            res = requests.get(url, headers=self.headers)
            if res.status_code == 200:
                pandascore_response = json.loads(res.text)
            else:
                raise ConnectionError(f"Error: {res.status_code}")
        except ConnectionError as e:
            raise e

        return pandascore_response

    @staticmethod
    def process_response(pandascore_response: dict) -> List[dict]:
        """
        Processes the response from the Pandascore API.

        Parameters
        ----------
        pandascore_response : dict
            The response from the API as a dictionary.

        Returns
        -------
        schedule : list of dict
            A list of dictionaries containing the processed match data.
        """
        schedule = []

        for match in pandascore_response:
            try:
                match_data = {
                    "league": match["league"]["name"],
                    "Blue": match["opponents"][0]["opponent"]["name"],
                    "Red": match["opponents"][1]["opponent"]["name"],
                    "Start (UTC)": match["scheduled_at"],
                    "Best Of": match["number_of_games"],
                }
                schedule.append(match_data)
            except IndexError:
                print(match)

        return schedule

    def get_schedule(
        self, leagues: Optional[str], start_datetime: str, end_datetime: str
    ) -> pd.DataFrame:
        """
        Gets the schedule of upcoming matches.

        Parameters
        ----------
        leagues : str, optional
            An optional string containing leagues of interest.
            Multiple leagues can be specified as a comma-separated string.
            ex: LCK, LPL, LEC
        start_datetime : str
            The start datetime for the matches in 'YYYY-MM-DDTHH:MM:SSZ' format.
        end_datetime : str
            The end datetime for the matches in 'YYYY-MM-DDTHH:MM:SSZ' format.

        Returns
        -------
        upcoming : pd.DataFrame
            A pandas DataFrame containing the upcoming matches.
        """
        time_format = "%Y-%m-%dT%H:%M:%SZ"
        schedule = []
        i = 1

        # Establish Time Range
        start_datetime = dt.datetime.strptime(start_datetime, time_format)
        end_datetime = dt.datetime.strptime(end_datetime, time_format)

        if (end_datetime - start_datetime).days > 7:
            raise ValueError(
                "The time delta between start_datetime "
                "and end_datetime cannot be more than 7 days."
            )

        # Paginate Data Within Time Range
        while True:
            pandascore_response = self.fetch_data(i)
            match_data = self.process_response(pandascore_response)
            if not match_data:  # No more data to process
                break

            match_datetimes = [
                dt.datetime.strptime(game["Start (UTC)"], time_format)
                for game in match_data
            ]

            if (
                min(match_datetimes) > end_datetime
                or max(match_datetimes) < start_datetime
            ):
                break  # Data is out of the datetime range

            schedule.extend(match_data)
            i += 1

        # Filter Results To Time Range
        schedule = list(
            filter(
                lambda game: end_datetime
                >= dt.datetime.strptime(game["Start (UTC)"], time_format)
                >= start_datetime,
                schedule,
            )
        )

        # Filter Leagues of Interest
        if leagues:
            leagues = leagues.strip().split(",")
            schedule = list(filter(lambda game: game["league"] in leagues, schedule))

        # Convert to Pandas DataFrame
        schedule = pd.DataFrame(schedule)

        return schedule
