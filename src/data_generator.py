"""
Data Generator.

This script is intended to represent the main function to update data once per day.
It takes the data that is daily stored in an S3 bucket from Oracle Elixir
and enriches it with additional data based on computed ratings.
The enriched data is then stored in the processed directory.


Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""

# Housekeeping
import datetime as dt
import json
from dataclasses import dataclass, field
from os import getenv

import boto3
import pandas as pd
from dotenv import load_dotenv

from src.data_ingest.oracles_elixir import OraclesElixir
from src.feature_engineering.impute_early_game_metrics import EarlyGameStatsImputer
from src.performance_features.performance_metrics import PerformanceMetrics
from src.ratings_features.rating_models import Ratings
from utils.logger import logger
from utils.paths import (
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
    INTERIM_DIR,
    INVALID_GAMES,
    PROCESSED_DIR,
    RAW_DIR,
)
from utils.utils import get_sorting_keys, json_loader

load_dotenv()


@dataclass
class DataGenerator:
    team_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    ts_lookup: dict = field(default_factory=dict)

    def __post_init__(self):
        self.s3_session = self.create_s3_session()
        self.oracle = OraclesElixir(
            session=self.s3_session, bucket=getenv("BUCKET_NAME")
        )
        self.imputer = EarlyGameStatsImputer()

    @staticmethod
    def create_s3_session():
        """
        Create a boto3 session to access the S3 bucket.
        """
        return boto3.Session(
            aws_access_key_id=getenv("ACCESS_ID"),
            aws_secret_access_key=getenv("SECRET_ID"),
        )

    @staticmethod
    def _get_years_to_process():
        """
        Get the years to process for the data ingestion.
        Those are the current year and the two previous years.
        """
        current_year = dt.date.today().year
        return [str(year) for year in range(current_year, current_year - 3, -1)]

    @staticmethod
    def _remove_buggy_games(data):
        """
        Remove games that have been identified as buggy.
        """
        with open(INVALID_GAMES, "r") as file:
            invalid_config = json.load(file)

        invalid_games = invalid_config["invalid_games"]
        data = data[~data.gameid.isin(invalid_games)].copy()
        logger.info("Removed buggy games.\n")
        return data

    def ingest_data_from_s3(self):
        """
        Ingest data from S3 bucket and store it in the interim directory.
        """
        try:
            # Define time frame for analytics
            years = self._get_years_to_process()

            # Ingest Data
            data = self.oracle.ingest_data(years=years)
            data.to_csv(RAW_DIR / "raw_data.csv", index=False)

            logger.info("Stored ingested data from S3.\n")

            # Remove Buggy Games
            data = self._remove_buggy_games(data)

            # Clean Data
            self.team_data = self.oracle.clean_data(data, split_on="team")
            self.player_data = self.oracle.clean_data(data, split_on="player")

            logger.info("Cleaned all data.\n")

            # Store Interim Data
            self.team_data.to_csv(INTERIM_DIR / "team_data.csv", index=False)
            self.player_data.to_csv(INTERIM_DIR / "player_data.csv", index=False)
            logger.info("Stored interim data.\n")
        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

    def _enrich_data_with_ratings(self):
        """
        Enrich data with all the associated ratings.
        """
        logger.info("Enriching data with ratings...")

        self.player_data = Ratings.compute_elo(df=self.player_data, entity="player")
        self.team_data = Ratings.compute_elo(df=self.team_data, entity="team")
        logger.info("Enriched data with elo.")

        self.player_data = Ratings.compute_plackett_luce(
            df=self.player_data, entity="player"
        )
        self.team_data = Ratings.compute_plackett_luce(df=self.team_data, entity="team")
        logger.info("Enriched data with plackett-luce.")

        self.player_data, self.team_data, self.ts_lookup = Ratings.compute_trueskill(
            player_data=self.player_data, team_data=self.team_data
        )
        logger.info("Enriched data with trueskill.")

        logger.info("Enriched data with ratings\n")

    def _enrich_data_with_performance_metrics(self):
        logger.info("Enriching data with performance metrics...")

        self.player_data = PerformanceMetrics.add_egpm_model(
            self.player_data, entity="player"
        )
        self.team_data = PerformanceMetrics.add_egpm_model(
            self.team_data, entity="team"
        )
        logger.info("Enriched data with EGPM.")

        self.player_data = PerformanceMetrics.add_entity_ema_statistics(
            self.player_data, entity="player"
        )
        self.team_data = PerformanceMetrics.add_entity_ema_statistics(
            self.team_data, entity="team"
        )
        logger.info("Enriched data with EMA statistics.")

        self.player_data = PerformanceMetrics.add_side_win_rate_ewm(
            self.player_data, entity="player"
        )
        self.team_data = PerformanceMetrics.add_side_win_rate_ewm(
            self.team_data, entity="team"
        )
        logger.info("Enriched data with side win rate.")

        logger.info("Enriched player and team data with performance metrics.")

    def enrich_datasets(self):
        """
        Compute all enrichment for team and player-based analytics and predictions.
        """
        try:
            # Load Data
            self.team_data = pd.read_csv(INTERIM_DIR / "team_data.csv")
            self.player_data = pd.read_csv(INTERIM_DIR / "player_data.csv")

            # Impute missing data for players only
            self.player_data = self.imputer.process_data(self.player_data)

            # Enrich Data with Ratings
            self._enrich_data_with_ratings()

            # Enrich Data with player performance metrics
            self._enrich_data_with_performance_metrics()

            # Sort dfs before storing them
            self.player_data.sort_values(get_sorting_keys("player"), inplace=True)
            self.team_data.sort_values(get_sorting_keys("team"), inplace=True)

            # Store Enriched Data
            self.team_data.to_csv(PROCESSED_DIR / "team_data.csv", index=False)
            self.player_data.to_csv(PROCESSED_DIR / "player_data.csv", index=False)
            logger.info("Stored enriched data.\n")
        except Exception as e:
            logger.error(f"Failed to enrich dataset: {e}")
            raise

    def flatten_team_data(self):
        """
        Flatten the team_data dataframe to get the most recent record per team.
        """

        flattened_team_config = json_loader(FLATTENED_TEAM_CONFIG)

        flattened_teams = (
            self.team_data.sort_values(["teamid", "date"])
            .groupby("teamid")
            .tail(1)
            .reset_index(drop=True)
        )
        flattened_teams = flattened_teams[flattened_team_config["flattened_cols"]]
        flattened_teams = flattened_teams.rename(
            columns=flattened_team_config["cols_renaming"]
        )
        flattened_teams.to_csv(PROCESSED_DIR / "flattened_teams.csv", index=False)
        logger.info("Stored flattened teams data.")

    def flatten_player_data(self):
        """
        Flatten the player_data dataframe to get the most recent record per player per team.
        """

        flattened_player_config = json_loader(FLATTENED_PLAYER_CONFIG)

        flattened_players = (
            self.player_data.sort_values(["playerid", "date"])
            .groupby("playerid")
            .tail(1)
            .reset_index(drop=True)
        )
        flattened_players = flattened_players[flattened_player_config["flattened_cols"]]
        flattened_players = flattened_players.rename(
            columns=flattened_player_config["cols_renaming"]
        )
        flattened_players.to_csv(PROCESSED_DIR / "flattened_players.csv", index=False)
        logger.info("Stored flattened players data.")

    def run(self):
        logger.info("Starting data generation.\n")
        self.ingest_data_from_s3()
        self.enrich_datasets()
        self.flatten_team_data()
        self.flatten_player_data()
        logger.info("Data generation completed.")


if __name__ == "__main__":
    generator = DataGenerator()
    logger.warning("Make sure to close any open CSV files!\n")

    start = dt.datetime.now()
    generator.run()
    end = dt.datetime.now()

    seconds_elapsed = (end - start).seconds
    logger.info(f"Data generation took {seconds_elapsed} seconds.\n")
