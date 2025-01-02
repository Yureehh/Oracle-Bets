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
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import awswrangler as wr
import boto3
import fireducks.pandas as pd
from dotenv import load_dotenv

from src.utils.logger import logger
from src.utils.paths import CONSIDERED_LEAGUES, IMPORT_COLUMNS, TEAM_REPLACEMENTS
from src.utils.utils import get_sorting_keys, json_loader

warnings.simplefilter(action="ignore", category=FutureWarning)


# Load environment variables from .env file
load_dotenv()

# Constants for the gap between players/teams for getting opponent data
GAP_PLAYER = 5
GAP_TEAM = 1

# Null value replacements
NULL_REPLACEMENTS = ["nan", "null", "unknown", "Unknown", "N/A"]


@dataclass
class OraclesElixir:
    """Class to ingest, clean, and format data from Oracle's Elixir."""

    session: Optional[boto3.Session]
    bucket: str

    def ingest_data(self, years: Optional[Union[List[Union[str, int]], str, int]] = None) -> pd.DataFrame:
        """
        Pull data from S3 based on the specified years and return as a DataFrame.

        Args:
            years (Optional[Union[List[Union[str, int]], str, int]]):
                Years to ingest data for. Defaults to the current year.

        Returns:
            pd.DataFrame: Ingested data.
        """
        if years is None:
            years = [dt.date.today().year]
        elif isinstance(years, (str, int)):
            years = [years]

        # TODO: Remove this once the data is available for 2025
        YEAR_REPLACEMENTS = {"2025": "2022"}
        years = [YEAR_REPLACEMENTS.get(year, year) for year in years]
        file_paths = [f"s3://{self.bucket}/{year}_LoL_esports_match_data_from_OraclesElixir.csv" for year in years]

        logger.info("Connecting to S3 bucket")
        try:
            with ThreadPoolExecutor() as executor:
                dataframes = list(
                    executor.map(
                        lambda path: wr.s3.read_csv(path, boto3_session=self.session, low_memory=False), file_paths
                    )
                )
            oracles_elixir_data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Successfully ingested data for years: {years}")
        except Exception as e:
            logger.error(f"Failed to ingest data for years: {years}")
            logger.error(e)
        return oracles_elixir_data

    @staticmethod
    def format_data_length_types(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Format and clean data types, handling dates, null values, and game lengths.

        Args:
            oracles_elixir_data (pd.DataFrame): Raw data from Oracle's Elixir.

        Returns:
            pd.DataFrame: Formatted and cleaned data.
        """
        logger.info("Formatting data types...")

        # Ensure we are working on the original DataFrame
        oracles_elixir_data.loc[:, "date"] = pd.to_datetime(oracles_elixir_data["date"], errors="coerce")

        # Normalize string columns by stripping whitespace and replacing null values
        identifier_columns = ["gameid", "playerid", "teamid", "league", "teamname", "playername"]
        oracles_elixir_data.loc[:, identifier_columns] = (
            oracles_elixir_data[identifier_columns].apply(lambda x: x.str.strip()).replace("", pd.NA)
        )

        # Replace NULL_REPLACEMENTS safely
        oracles_elixir_data.loc[:, :] = oracles_elixir_data.replace(NULL_REPLACEMENTS, pd.NA)

        # Convert gamelength to minutes
        oracles_elixir_data.loc[:, "gamelength"] = (
            pd.to_numeric(oracles_elixir_data["gamelength"], errors="coerce") / 60
        ).astype(float)

        logger.info("Data formatting completed.")
        return oracles_elixir_data

    @staticmethod
    def remove_null_games(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where the 'gameid' is null to ensure dataset completeness.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame without null 'gameid'.
        """
        if "gameid" not in oracles_elixir_data.columns:
            raise ValueError("The dataframe does not contain the 'gameid' column.")

        initial_count = oracles_elixir_data.shape[0]
        cleaned_data = oracles_elixir_data.dropna(subset=["gameid"])
        removed_count = initial_count - cleaned_data.shape[0]
        logger.info(f"Removed {removed_count} rows with null 'gameid'.")
        return cleaned_data

    @staticmethod
    def drop_unknown_entities(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where player or team names are 'unknown'.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame without unknown entities.
        """
        required_columns = ["playername", "teamname"]
        for column in required_columns:
            if column not in oracles_elixir_data.columns:
                raise ValueError(f"Missing '{column}' in dataframe.")

        oracles_elixir_data = oracles_elixir_data[
            ~oracles_elixir_data["playername"].fillna("").str.lower().isin(["unknown player"])
            & ~oracles_elixir_data["teamname"].fillna("").str.lower().isin(["unknown team"])
        ]

        logger.info("Removed rows with unknown player or team names.")
        return oracles_elixir_data

    @staticmethod
    def replace_team_names(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace team names with consistent naming conventions.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to update.

        Returns:
            pd.DataFrame: DataFrame with replaced team names.
        """
        try:
            with open(TEAM_REPLACEMENTS) as file:
                team_name_replacements = json.load(file)["team_name_replacements"]
        except FileNotFoundError:
            logger.error(f"Team replacements file not found at {TEAM_REPLACEMENTS}.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {TEAM_REPLACEMENTS}: {e}")
            raise

        for old, replacement in team_name_replacements:
            oracles_elixir_data["teamname"] = oracles_elixir_data["teamname"].replace(old["name"], replacement["name"])
            oracles_elixir_data["teamid"] = oracles_elixir_data["teamid"].replace(old["teamid"], replacement["teamid"])

        logger.info("Replaced incorrect team names with correct ones.")
        return oracles_elixir_data

    @staticmethod
    def sort_data(oracles_elixir_data: pd.DataFrame, split_on: str) -> pd.DataFrame:
        """
        Sort Oracle's Elixir data by defined sorting keys based on the entity type.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to sort.
            split_on (str): Entity type to split on ('player' or 'team').

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        if split_on not in ["player", "team"]:
            raise ValueError("split_on must be either 'player' or 'team'.")

        sorting_keys = get_sorting_keys(split_on)
        oracles_elixir_data = oracles_elixir_data.sort_values(by=sorting_keys).reset_index(drop=True)
        logger.info(f"Sorted data by {sorting_keys}.")
        return oracles_elixir_data

    @staticmethod
    def fill_null_team_ids(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null team IDs with team names.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to update.

        Returns:
            pd.DataFrame: Updated DataFrame with filled team IDs.
        """
        oracles_elixir_data["teamname"] = oracles_elixir_data["teamname"].astype(str).fillna("")
        oracles_elixir_data["teamid"] = (
            oracles_elixir_data["teamid"].astype(str).fillna(oracles_elixir_data["teamname"])
        )
        logger.info("Filled null team IDs with team names.")
        return oracles_elixir_data

    @staticmethod
    def subset_data(
        oracles_elixir_data: pd.DataFrame, split_on: str, columns: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Subset the dataset down to relevant columns based on the specified entity (either 'team' or 'player').

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to subset.
            split_on (str): Entity type to split on ('player' or 'team').
            columns (Optional[Dict[str, List[str]]]): Column mappings. Defaults to None.

        Returns:
            pd.DataFrame: Subsetted DataFrame.
        """
        try:
            with open(IMPORT_COLUMNS) as file:
                columns = json.load(file)

            if split_on not in columns:
                raise ValueError("Must split on either 'player' or 'team'.")
        except FileNotFoundError:
            logger.error(f"Import columns file not found at {IMPORT_COLUMNS}.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {IMPORT_COLUMNS}: {e}")
            raise

        # Rename columns for consistency
        rename_mapping = {
            "earned gpm": "egpm",
            "team kpm": "team_kpm",
            "total cs": "total_cs",
        }
        oracles_elixir_data = oracles_elixir_data.rename(columns=rename_mapping)

        if "position" not in oracles_elixir_data.columns:
            raise ValueError("The dataframe does not contain the 'position' column.")
        oracles_elixir_data["position"] = oracles_elixir_data["position"].fillna("")

        # Filter dataset by position and select relevant columns
        position_filter = (
            oracles_elixir_data["position"].str.lower() == split_on
            if split_on == "team"
            else oracles_elixir_data["position"].str.lower() != "team"
        )
        oracles_elixir_data = oracles_elixir_data[position_filter]
        logger.info(f"Filtered data by {split_on}s.")
        return oracles_elixir_data[columns[split_on]]

    @staticmethod
    def remove_inconsistent_games(oracles_elixir_data: pd.DataFrame, split_on: str = "player") -> pd.DataFrame:
        """
        Remove entries from the input DataFrame with inconsistent game records based on gameID counts.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to clean.
            split_on (str, optional): Entity type to split on. Defaults to "player".

        Returns:
            pd.DataFrame: Cleaned DataFrame without inconsistent games.
        """
        game_counts = oracles_elixir_data["gameid"].value_counts()
        expected_count = 2 if split_on.lower() == "team" else 10
        inconsistent_game_ids = game_counts[game_counts != expected_count].index
        oracles_elixir_data = oracles_elixir_data[~oracles_elixir_data["gameid"].isin(inconsistent_game_ids)]
        logger.info(f"Removed {len(inconsistent_game_ids)} inconsistent games based on gameID counts.")
        return oracles_elixir_data

    @staticmethod
    def enrich_opponent_metrics(oracles_elixir_data: pd.DataFrame, split_on: str) -> pd.DataFrame:
        """
        Enrich the Oracle's Elixir data with opponent metrics.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to enrich.
            split_on (str): Entity type to split on ('player' or 'team').

        Returns:
            pd.DataFrame: Enriched DataFrame with opponent metrics.
        """
        metrics = {
            "teamid": oracles_elixir_data["teamid"].fillna(oracles_elixir_data["teamname"]),
            "opponentteam": get_opponent(oracles_elixir_data["teamname"].to_list(), split_on),
            "opponentteamid": get_opponent(oracles_elixir_data["teamid"].to_list(), split_on),
        }

        if split_on == "player":
            metrics.update(
                {
                    "playerid": oracles_elixir_data["playerid"].fillna(oracles_elixir_data["playername"]),
                    "opponentplayername": get_opponent(oracles_elixir_data["playername"].to_list(), split_on),
                    "opponentplayerid": get_opponent(oracles_elixir_data["playerid"].to_list(), split_on),
                }
            )

        # Assign new opponent metrics to the DataFrame
        oracles_elixir_data = oracles_elixir_data.assign(**metrics)
        logger.info("Enriched data with opponent metrics.")
        return oracles_elixir_data

    @staticmethod
    def filter_leagues(oracles_elixir_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataset to include only the leagues considered relevant.

        Args:
            oracles_elixir_data (pd.DataFrame): DataFrame to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame with relevant leagues.
        """
        logger.info("Filtering data for relevant leagues...")
        try:
            considered_leagues = json_loader(CONSIDERED_LEAGUES)["considered_leagues"]
            if not considered_leagues:
                raise ValueError("No leagues specified in the considered leagues list.")
        except KeyError:
            logger.error("Incorrect or missing 'considered_leagues' key in JSON configuration.")
            raise
        except FileNotFoundError:
            logger.error("League configuration file not found.")
            raise
        return oracles_elixir_data[oracles_elixir_data["league"].isin(considered_leagues)]

    def clean_data(
        self,
        oracles_elixir_data: pd.DataFrame,
        split_on: str,
    ) -> pd.DataFrame:
        """
        Format and clean data from Oracle's Elixir.

        This function makes the data more consistent and user-friendly.

        The date column will be formatted appropriately as a datetime object.
        Any games with 'unknown team' or 'unknown player' will be dropped.
        Any games with null game ids will be dropped.
        Opponent metrics will be enriched into the dataframe.
        Subsets the dataset down to relevant columns for the entity you split on.
        NOTE: Not all data from the initial dataset are in the "cleaned" output.

        Args:
            oracles_elixir_data (pd.DataFrame): Raw data to clean.
            split_on (str): Entity type to split on ('player' or 'team').

        Returns:
            pd.DataFrame: Cleaned and formatted data.
        """
        logger.info(f"Cleaning data for {split_on}s...")
        oracles_elixir_data = self.format_data_length_types(oracles_elixir_data)
        oracles_elixir_data = self.remove_null_games(oracles_elixir_data)
        oracles_elixir_data = self.drop_unknown_entities(oracles_elixir_data)
        oracles_elixir_data = self.replace_team_names(oracles_elixir_data)
        oracles_elixir_data = self.sort_data(oracles_elixir_data, split_on)
        oracles_elixir_data = self.fill_null_team_ids(oracles_elixir_data)
        oracles_elixir_data = self.subset_data(oracles_elixir_data, split_on)
        oracles_elixir_data = self.remove_inconsistent_games(oracles_elixir_data, split_on)
        oracles_elixir_data = self.enrich_opponent_metrics(oracles_elixir_data, split_on)
        oracles_elixir_data = self.filter_leagues(oracles_elixir_data)
        logger.info(f"Data cleaning for {split_on}s completed.\n")
        return oracles_elixir_data


def get_opponent(column: pd.Series, entity: str) -> pd.Series:
    """
    Generate values for the opposing team or player.

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
