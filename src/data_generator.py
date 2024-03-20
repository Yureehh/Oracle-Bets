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
from dataclasses import dataclass
from os import getenv

import boto3
import pandas as pd
from dotenv import load_dotenv

from src.data_ingest.oracles_elixir import OraclesElixir
from src.feature_engineering.impute_early_game_metrics import EarlyGameStatsImputer
from src.performance_features.performance_metrics import PerformanceMetrics
from src.ratings_features.rating_models import Ratings
from utils.logger import logger
from utils.paths import INTERIM_DIR, INVALID_GAMES, PROCESSED_DIR, RAW_DIR
from utils.utils import get_sorting_keys

load_dotenv()


@dataclass
class DataGenerator:
    def __post_init__(self):
        self.s3_session = self.create_s3_session()
        self.oracle = OraclesElixir(
            session=self.s3_session, bucket=getenv("BUCKET_NAME")
        )
        self.imputer = EarlyGameStatsImputer()

    def create_s3_session(self):
        """
        Create a boto3 session to access the S3 bucket.
        """
        return boto3.Session(
            aws_access_key_id=getenv("ACCESS_ID"),
            aws_secret_access_key=getenv("SECRET_ID"),
        )

    def _get_years_to_process(self):
        """
        Get the years to process for the data ingestion.
        Those are the current year and the two previous years.
        """
        current_year = dt.date.today().year
        return [str(year) for year in range(current_year, current_year - 3, -1)]

    def _remove_buggy_games(self, data):
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

        - The function doesn't have parameters, but it reads the data from the S3 bucket
        - The function doesn't return anything, but it stores the data in the INTERIM_DIR
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
            team_data = self.oracle.clean_data(data, split_on="team")
            player_data = self.oracle.clean_data(data, split_on="player")

            logger.info("Cleaned all data.\n")

            # Store Interim Data
            team_data.to_csv(INTERIM_DIR / "team_data.csv", index=False)
            player_data.to_csv(INTERIM_DIR / "player_data.csv", index=False)
            logger.info("Stored interim data.\n")
            return team_data, player_data
        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

    def _enrich_data_with_ratings(self, player_data, team_data):
        """
        Enrich data with all the associated ratings.

        Parameters
        ----------
        player_data : pd.DataFrame
        team_data : pd.DataFrame

        Returns
        -------
        enriched_player_data : pd.DataFrame
        enriched_team_data : pd.DataFrame
        """

        logger.info("Enriching data with ratings...")

        player_data = Ratings.compute_elo(df=player_data, entity="player")
        team_data = Ratings.compute_elo(df=team_data, entity="team")
        logger.info("Enriched data with elo.")

        player_data = Ratings.compute_plackett_luce(df=player_data, entity="player")
        team_data = Ratings.compute_plackett_luce(df=team_data, entity="team")
        logger.info("Enriched data with plackett-luce.")

        player_data, team_data, ts_lookup = Ratings.compute_trueskill(
            player_data=player_data, team_data=team_data
        )
        logger.info("Enriched data with trueskill.")

        logger.info("Enriched data with ratings\n")
        return player_data, team_data, ts_lookup

    def _enrich_data_with_performance_metrics(self, player_data, team_data):
        logger.info("Enriching data with performance metrics...")

        player_data = PerformanceMetrics.add_egpm_model(player_data, entity="player")
        team_data = PerformanceMetrics.add_egpm_model(team_data, entity="team")
        logger.info("Enriched data with EGPM.")

        player_data = PerformanceMetrics.add_entity_ema_statistics(
            player_data, entity="player"
        )
        team_data = PerformanceMetrics.add_entity_ema_statistics(
            team_data, entity="team"
        )
        logger.info("Enriched data with EMA statistics.")

        player_data = PerformanceMetrics.add_side_win_rate_ewm(
            player_data, entity="player"
        )
        team_data = PerformanceMetrics.add_side_win_rate_ewm(team_data, entity="team")
        logger.info("Enriched data with side win rate.")

        logger.info("Enriched player and team data with performance metrics.")

        return player_data, team_data

    def enrich_datasets(self):
        """
        Compute all enrichment for team and player-based analytics and predictions.
        This includes Team and Player-based elo, TrueSkill, and EGPM dominance.

        - The function doesn't have parameters, but it reads the data from the INTERIM_DIR
        - The function doesn't return anything, but it stores the enriched data in the PROCESSED_DIR
        """
        try:
            # Load Data
            team_data = pd.read_csv(INTERIM_DIR / "team_data.csv")
            player_data = pd.read_csv(INTERIM_DIR / "player_data.csv")

            # Impute missing data for players only
            player_data = self.imputer.process_data(player_data)

            # Enrich Data with Ratings
            player_data, team_data, ts_lookup = self._enrich_data_with_ratings(
                player_data, team_data
            )

            # Enrich Data with player performance metrics
            player_data, team_data = self._enrich_data_with_performance_metrics(
                player_data, team_data
            )

            # Sort dfs before storing them
            player_data.sort_values(get_sorting_keys("player"), inplace=True)
            team_data.sort_values(get_sorting_keys("team"), inplace=True)

            # Store Enriched Data
            team_data.to_csv(PROCESSED_DIR / "team_data.csv", index=False)
            player_data.to_csv(PROCESSED_DIR / "player_data.csv", index=False)
            logger.info("Stored enriched data.\n")

            return team_data, player_data
        except Exception as e:
            logger.error(f"Failed to enrich dataset: {e}")
            raise

    def run(self):
        logger.info("Starting data generation.\n")
        self.ingest_data_from_s3()
        self.enrich_datasets()
        logger.info("Data generation completed.")


if __name__ == "__main__":
    generator = DataGenerator()

    start = dt.datetime.now()
    generator.run()
    end = dt.datetime.now()

    seconds_elapsed = (end - start).seconds
    logger.info(f"Data generation took {seconds_elapsed} seconds.\n")
