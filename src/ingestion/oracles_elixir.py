"""
Oracle's Elixir

This script is designed to connect to Tim Sevenhuysen's Oracle's Elixir data.
It is built to empower esports enthusiasts, data scientists, or anyone
to leverage pro game data for use in their own scripts and analytics.

Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""

import datetime as dt
import json
from dataclasses import dataclass
from typing import Optional, Union

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.logger import logger
from utils.paths import CONSIDERED_LEAGUES, IMPORT_COLUMNS, TEAM_REPLACEMENTS
from utils.utils import get_sorting_keys, json_loader

# Load environment variables from .env file
load_dotenv()
pd.set_option("future.no_silent_downcasting", True)  # TODO: Remove this line after fixing the warning

# Constants for the gap between players/teams for getting opponent data
GAP_PLAYER = 5
GAP_TEAM = 1

# Null value replacements
NULL_REPLACEMENTS = ["nan", "null", "unknown", "Unknown", "N/A"]


@dataclass
class OraclesElixir:
    session: Optional[boto3.Session]
    bucket: str

    def ingest_data(self, years: Optional[Union[list, str, int]] = None) -> pd.DataFrame:
        """Pull data from S3 based on the specified years and return as a DataFrame."""
        if years is None:
            years = [dt.date.today().year]
        if isinstance(years, (str, int)):
            years = [years]

        file_paths = [f"s3://{self.bucket}/{year}_LoL_esports_match_data_from_OraclesElixir.csv" for year in years]

        logger.info("Connecting to S3 bucket")
        try:
            oe_data = wr.s3.read_csv(file_paths, boto3_session=self.session, low_memory=False)
            logger.info(f"Successfully ingested data for years: {years}")
        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

        return oe_data

    @staticmethod
    def format_data_types(oe_data: pd.DataFrame) -> pd.DataFrame:
        """Format and clean data types, handling dates, null values, and game lengths."""
        logger.info("Formatting data types...")
        oe_data.loc[:, "date"] = pd.to_datetime(oe_data["date"], errors="coerce")

        # Normalize string columns by stripping whitespace and replacing null values
        id_cols = ["gameid", "playerid", "teamid", "league", "teamname", "playername"]
        oe_data.loc[:, id_cols] = oe_data[id_cols].apply(lambda x: x.str.strip()).replace("", np.nan)
        oe_data = oe_data.replace(NULL_REPLACEMENTS, pd.NA)
        oe_data["gamelength"] = oe_data["gamelength"].astype(float).div(60)
        logger.info("Data formatting completed.")
        return oe_data

    @staticmethod
    def remove_null_games(oe_data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where the 'gameid' is null to ensure dataset completeness."""
        if "gameid" not in oe_data.columns:
            raise ValueError("The dataframe does not contain the 'gameid' column.")

        cleaned_data = oe_data.dropna(subset=["gameid"])
        return cleaned_data

    @staticmethod
    def drop_unknown_entities(oe_data: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where player or team names are 'unknown'."""
        if "playername" not in oe_data or "teamname" not in oe_data:
            raise ValueError("Missing 'playername' or 'teamname' in dataframe.")

        return oe_data[
            ~oe_data["playername"].str.lower().isin(["unknown player"])
            & ~oe_data["teamname"].str.lower().isin(["unknown team"])
        ]

    @staticmethod
    def replace_team_names(oe_data: pd.DataFrame) -> pd.DataFrame:
        """Replace team names with consistent naming conventions."""
        with open(TEAM_REPLACEMENTS) as f:
            team_replacements = json.load(f)["team_replacements"]

        for pair in team_replacements:
            old, replacement = pair[0], pair[1]
            oe_data["teamname"] = oe_data["teamname"].replace(old["name"], replacement["name"])
            oe_data["teamid"] = oe_data["teamid"].replace(old["teamid"], replacement["teamid"])
        logger.info("Replaced incorrect team names with correct ones.\n")
        return oe_data

    @staticmethod
    def sort_data(oe_data: pd.DataFrame, split_on: Optional[str]) -> pd.DataFrame:
        """Sort Oracle's Elixir data by defined sorting keys based on the entity type."""
        if split_on not in ["player", "team"]:
            raise ValueError("split_on must be either 'player' or 'team'.")

        return oe_data.sort_values(get_sorting_keys(split_on))

    @staticmethod
    def fill_null_team_ids(oe_data: pd.DataFrame, split_on: str) -> pd.DataFrame:
        """Fill null team ids with team names."""
        oe_data["teamid"] = oe_data["teamid"].fillna(oe_data["teamname"])
        return oe_data

    @staticmethod
    def subset_data(oe_data: pd.DataFrame, split_on: str, columns: Optional[dict] = None) -> pd.DataFrame:
        """Subset the dataset down to relevant columns based on the specified entity (either 'team' or 'player')."""
        if columns is None:
            with open(IMPORT_COLUMNS) as file:
                columns = json.load(file)

        if split_on not in columns:
            raise ValueError("Must split on either 'player' or 'team'.")

        # Rename columns for consistency
        oe_data = oe_data.rename(
            columns={
                "earned gpm": "egpm",
                "team kpm": "team_kpm",
                "total cs": "total_cs",
            }
        )

        # Filter dataset by position and select relevant columns
        oe_data = (
            oe_data[oe_data["position"].str.lower() == split_on]
            if split_on == "team"
            else oe_data[oe_data["position"].str.lower() != "team"]
        )
        return oe_data[columns[split_on]]

    @staticmethod
    def remove_inconsistent_games(oe_data: pd.DataFrame, split_on: Optional[str] = "player") -> pd.DataFrame:
        """Remove entries from the input DataFrame with inconsistent game records based on gameID counts."""
        counts = oe_data["gameid"].value_counts()
        expected_count = 2 if split_on.lower() == "team" else 10
        inconsistent_games = counts[counts != expected_count].index
        return oe_data[~oe_data["gameid"].isin(inconsistent_games)]

    @staticmethod
    def enrich_opponent_metrics(oe_data: pd.DataFrame, split_on: str) -> pd.DataFrame:
        """Enrich the Oracle's Elixir data with opponent metrics."""
        metrics = {
            "teamid": oe_data["teamid"].fillna(oe_data["teamname"]),
            "opponentteam": get_opponent(oe_data["teamname"].to_list(), split_on),
            "opponentteamid": get_opponent(oe_data["teamid"].to_list(), split_on),
            "opp_egpm": get_opponent(oe_data["egpm"].to_list(), split_on),
        }
        # If split_on is "player", update metrics
        if split_on == "player":
            metrics.update(
                {
                    "playerid": oe_data["playerid"].fillna(oe_data["playername"]),
                    "opponentplayername": get_opponent(oe_data["playername"].to_list(), split_on),
                    "opponentplayerid": get_opponent(oe_data["playerid"].to_list(), split_on),
                }
            )

        # Assign new opponent metrics to the dataframe
        oe_data = oe_data.assign(**metrics)
        return oe_data

    @staticmethod
    def filter_leagues(oe_data: pd.DataFrame) -> pd.DataFrame:
        """Filter the dataset to include only the leagues considered relevant."""
        try:
            considered_leagues = json_loader(CONSIDERED_LEAGUES)["considered_leagues"]
            if considered_leagues:
                return oe_data[oe_data["league"].isin(considered_leagues)]
            else:
                raise ValueError("No leagues specified in the considered leagues list.")
        except KeyError:
            logger.error("Incorrect or missing 'considered_leagues' key in JSON configuration.")
            raise
        except FileNotFoundError:
            logger.error("League configuration file not found.")
            raise

    def clean_data(
        self,
        oe_data: pd.DataFrame,
        split_on: Optional[str],
    ) -> pd.DataFrame:
        """
        Format and clean data from Oracle's Elixir.
        This function makes the data more consistent and user-friendly

        The date column will be formatted appropriately as a datetime object.
        Any games with 'unknown team' or 'unknown player' will be dropped.
        Any games with null game ids will be dropped.
        Opponent metrics will be enriched into the dataframe.
        Subsets the dataset down to relevant columns for the entity you split on.
        NOTE: Not all data from the initial data set are in the "cleaned" output.
        """
        logger.info(f"Cleaning data for {split_on}s...")
        oe_data = self.format_data_types(oe_data)
        oe_data = self.remove_null_games(oe_data)
        oe_data = self.drop_unknown_entities(oe_data)
        oe_data = self.replace_team_names(oe_data)
        oe_data = self.sort_data(oe_data, split_on)
        oe_data = self.fill_null_team_ids(oe_data, split_on)
        oe_data = self.subset_data(oe_data, split_on)
        oe_data = self.remove_inconsistent_games(oe_data, split_on)
        oe_data = self.enrich_opponent_metrics(oe_data, split_on)
        oe_data = self.filter_leagues(oe_data)
        return oe_data


def get_opponent(column: pd.Series, entity: str) -> list:
    """
    Generate value for the opposing team or player.
    Used for utilities such as returning the opposing player/team's name.
    It can also return opposing metrics, ex: opponent's earned gold per minute
    Be sure that the input value is sorted to have consistent order in rows.
    """
    opponent = []
    flag = 0

    gap_dict = {"player": GAP_PLAYER, "team": GAP_TEAM}
    gap = gap_dict.get(entity)
    if gap is None:
        raise ValueError("Entity must be either player or team.")

    for i, _ in enumerate(column):
        # If "Blue Side" - fetch opposing team/player below
        if flag < gap:
            opponent.append(column[i + gap])
        # If "Red Side" - fetch opposing team/player above
        elif gap <= flag < (gap * 2):
            opponent.append(column[i - gap])
        else:
            raise ValueError(f"Index {i} - Out Of Bounds")

        flag += 1

        # After both sides of a game are enumerated, reset the flag
        if flag >= gap * 2:
            flag = 0
    return opponent
