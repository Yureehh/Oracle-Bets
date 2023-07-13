"""
Oracle's Elixir

This script is designed to connect to Tim Sevenhuysen's Oracle's Elixir site to pull down and format data.
It is built to empower esports enthusiasts, data scientists, or anyone to leverage pro game data
for use in their own scripts and analytics.

Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""
# Housekeeping
import awswrangler as wr
import boto3
import datetime as dt
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


# Primary Functions
class OraclesElixir:
    def __init__(self,
                 session: Optional[boto3.Session],
                 bucket: str):
        """
        bucket : str
            Name of the bucket containing Oracle's Elixir S3 data.
        """
        self.oe_data = pd.DataFrame()
        self.session = session
        self.bucket = bucket

    def ingest_data(self,
                    years: Optional[Union[list, str, int]] = [dt.date.today().year]) -> pd.DataFrame:
        """
        Pull data from S3 based on the specified years
            and store it in the `oe_data` instance variable.

        Parameters
        ----------
        years : Union[list, str, int]
            A string or list of strings containing years (e.g. ["2019", "2020"]).
            If nothing is specified, returns the current year only by default.
        """
        if isinstance(years, (str, int)):
            years = [years]

        file_paths = [f"s3://{self.bucket}/{year}_LoL_esports_match_data_from_OraclesElixir.csv" for year in years]
        self.oe_data = wr.s3.read_csv(file_paths, boto3_session=self.session)

        return self.oe_data

    @staticmethod
    def get_opponent(column: pd.Series, entity: str) -> list:
        """
        Generate value for the opposing team or player.
        This can be used for utilities such as returning the opposing player or team's name.
        It can also be used to return opposing metrics, such as opponent's earned gold per minute, etc.
        Be sure that the input value is sorted to have consistent order of matches/positions.

        Parameters
        ----------
        column : Pandas Series
            Pandas DataFrame column representing entity data (see entity)
        entity : str
            'player' or 'team', entity to calculate opponent of.

        Returns
        -------
        opponent : list
            The opponent of the entities in the column provided; can be inserted as a column back into the dataframe.
        """
        opponent = []
        flag = 0

        # The gap represents how many rows separate a value from its opponent
        # Teams are 1 (Team A, Team B)
        # Players are 5 (ADC to opposing ADC is a 5 row gap)
        if entity == "player":
            gap = 5
        elif entity == "team":
            gap = 1
        else:
            raise ValueError("Entity must be either player or team.")

        for i, obj in enumerate(column):
            # If "Blue Side" - fetch opposing team/player below
            if flag < gap:
                opponent.append(column[i + gap])
                flag += 1
            # If "Red Side" - fetch opposing team/player above
            elif gap <= flag < (gap * 2):
                opponent.append(column[i - gap])
                flag += 1
            else:
                raise ValueError(f"Index {i} - Out Of Bounds")

            # After both sides are enumerated, reset the flag
            if flag >= (gap * 2):
                flag = 0
        return opponent

    def format_data_types(self) -> pd.DataFrame:
        self.oe_data["date"] = pd.to_datetime(self.oe_data["date"])
        self.oe_data[["gameid", "playerid", "teamid"]] = self.oe_data[
            ["gameid", "playerid", "teamid"]
        ].str.strip()
        replace_values = {"": np.nan, "nan": np.nan, "null": np.nan}
        self.oe_data = self.oe_data.replace(
            {
                "gameid": replace_values,
                "playerid": replace_values,
                "teamid": replace_values,
                "position": replace_values,
            }
        )
        return self.oe_data

    def format_ids(self) -> pd.DataFrame:
        self.oe_data = self.oe_data[
            (self.oe_data["gameid"].notna()) & (self.oe_data["position"].notna())
            ]
        return self.oe_data

    def remove_null_games(self) -> pd.DataFrame:
        return self.oe_data.dropna(subset=["gameid"])

    def drop_unknown_entities(self) -> pd.DataFrame:
        return self.oe_data[
            (~self.oe_data["playername"].isin(["unknown player"]))
            & (~self.oe_data["teamname"].isin(["unknown team"]))
            ]

    def drop_negative_earned_gpm(self) -> pd.DataFrame:
        return self.oe_data[self.oe_data["earned gpm"] >= 0]

    def normalize_names(
            self,
            team_replacements: Dict,
            player_replacements: Dict,
    ) -> pd.DataFrame:
        if team_replacements:
            self.oe_data["teamname"] = self.oe_data["teamname"].replace(team_replacements)
        if player_replacements:
            self.oe_data["playername"] = self.oe_data["playername"].replace(player_replacements)
        return self.oe_data

    def subset_data(self, split_on: Optional[str]) -> pd.DataFrame:
        if split_on == "team":
            return self.oe_data[
                [
                    "date",
                    "gameid",
                    "side",
                    "league",
                    "teamname",
                    "teamid",
                    "result",
                    "kills",
                    "deaths",
                    "assists",
                    "earned gpm",
                    "gamelength",
                    "ckpm",
                    "team kpm",
                    "firstblood",
                    "dragons",
                    "barons",
                    "towers",
                    "goldat15",
                    "xpat15",
                    "csat15",
                    "golddiffat15",
                    "xpdiffat15",
                    "csdiffat15",
                ]
            ]
        elif split_on == "player":
            return self.oe_data[
                [
                    "date",
                    "gameid",
                    "side",
                    "position",
                    "league",
                    "playername",
                    "playerid",
                    "teamname",
                    "teamid",
                    "result",
                    "kills",
                    "deaths",
                    "assists",
                    "total cs",
                    "earned gpm",
                    "earnedgoldshare",
                    "gamelength",
                    "ckpm",
                    "team kpm",
                    "goldat15",
                    "xpat15",
                    "csat15",
                    "killsat15",
                    "assistsat15",
                    "deathsat15",
                    "opp_killsat15",
                    "opp_assistsat15",
                    "opp_deathsat15",
                    "golddiffat15",
                    "xpdiffat15",
                    "csdiffat15",
                ]
            ]
        else:
            raise ValueError("Must split on either player or team.")

    def remove_inconsistent_games(self, split_on: Optional[str]) -> pd.DataFrame:
        counts = self.oe_data["gameid"].value_counts()
        inconsistent_games = counts[
            (counts != 2) if split_on == "team" else (counts < 1) | (counts > 10)
        ].index
        return self.oe_data[~self.oe_data["gameid"].isin(inconsistent_games)]

    def sort_data(self, split_on: Optional[str]) -> pd.DataFrame:
        if split_on == "player":
            return self.oe_data.sort_values(["league", "date", "gameid", "side", "position"])
        elif split_on == "team":
            return self.oe_data.sort_values(["league", "date", "gameid", "side"])

    def fill_null_team_ids(self, split_on: str) -> pd.DataFrame:
        if split_on == "player":
            self.oe_data["teamid"] = self.oe_data["teamid"].fillna(self.oe_data["teamname"])
        return self.oe_data

    def enrich_opponent_metrics(self, split_on: str) -> pd.DataFrame:
        if split_on == "player":
            self.oe_data["playerid"] = self.oe_data["playerid"].fillna(self.oe_data["playername"])
            self.oe_data["opponentteam"] = self.get_opponent(self.oe_data["teamname"].to_list(), split_on)
            self.oe_data["opponentteamid"] = self.get_opponent(self.oe_data["teamid"].to_list(), split_on)

        self.oe_data["opponentname"] = self.get_opponent(self.oe_data["playername"].to_list(), split_on)
        self.oe_data["opponentid"] = self.get_opponent(self.oe_data["playerid"].to_list(), split_on)
        self.oe_data["opponent_egpm"] = self.get_opponent(self.oe_data["earned gpm"].to_list(), split_on)

        return self.oe_data

    def clean_data(self, split_on: Optional[str], team_replacements: Optional[Dict] = None,
                   player_replacements: Optional[Dict] = None) -> pd.DataFrame:
        """
        Format and clean data from Oracle's Elixir.
        This function is optional, and provided as a convenience to help make the data more consistent and user-friendly

        The date column will be formatted appropriately as a datetime object.
        Any games with 'unknown team' or 'unknown player' will be dropped.
        Any games with null game ids will be dropped.
        Opponent metrics will be enriched into the dataframe.
        This function also subsets the dataset down to relevant columns for the entity you split on (team, player).
        Please note that this means not all columns from the initial data set are in the "cleaned" output.

        Parameters
        ----------
        split_on : 'team', 'player' or None
            Subset data for Team data or Player data. None for all data.
        team_replacements: Optional[dict]
            Replacement values to normalize team names in the data if a team name changes over time.
            The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}
        player_replacements: Optional[dict]
            Replacement values to normalize player names in the data if a player's name changes over time.
            The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}

        Returns
        -------
        A Pandas dataframe of formatted, subset Oracle's Elixir data matching
        the parameters provided above.
        """
        self.oe_data = self.format_data_types()
        self.oe_data = self.format_ids()
        self.oe_data = self.remove_null_games()
        self.oe_data = self.drop_unknown_entities()
        self.oe_data = self.drop_negative_earned_gpm()
        self.oe_data = self.normalize_names(team_replacements, player_replacements)
        self.oe_data = self.subset_data(split_on)
        self.oe_data = self.remove_inconsistent_games(split_on)
        self.oe_data = self.sort_data(split_on)
        self.oe_data = self.fill_null_team_ids(split_on)
        self.oe_data = self.enrich_opponent_metrics(split_on)
        return self.oe_data
