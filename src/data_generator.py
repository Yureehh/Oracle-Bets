"""
Data Generator.

This script is intended to represent the main function to update data once per day.
It takes the data that is daily stored in an S3 bucket from Oracle Elixir
and enriches it with additional data based on computed ratings.
The enriched data is then stored in the processed directory.


Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from os import getenv

import boto3
import pandas as pd
from dotenv import load_dotenv

from src.feature_engineering.features_generator import FeatureGenerator
from src.feature_engineering.impute_early_game_metrics import EarlyGameStatsImputer
from src.feature_engineering.performance_features.performance_metrics import PerformanceMetrics
from src.feature_engineering.ratings_features.rating_models import Ratings
from src.ingestion.oracles_elixir import OraclesElixir
from utils.logger import logger
from utils.paths import (
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
    INTERIM_PLAYER_DATA,
    INTERIM_TEAM_DATA,
    INVALID_GAMES,
    LEAGUE_ELO,
    PROCESSED_DIR,
    PROCESSED_PLAYERS,
    PROCESSED_TEAMS,
    RAW_DATA,
    TEAM_LEAGUES_MAPPING,
    TRAINING_PLAYER_CONFIG,
    TRAINING_TEAM_CONFIG,
    YEARS_RANGE_PATH,
)
from utils.utils import get_sorting_keys, json_loader

# Load environment variables from .env file
load_dotenv()

# Constants
BUCKET_NAME_ENV = "BUCKET_NAME"
ACCESS_ID_ENV = "ACCESS_ID"
SECRET_ID_ENV = "SECRET_ID"
MAX_EXPECTED_PLAYERS = 10
MAX_EXPECTED_TEAMS = 2
MAX_EXPECTED_ROWS = 12


@dataclass
class DataGenerator:
    team_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    bucket_name: str = field(init=False)

    def __post_init__(self):
        """Initialize DataGenerator with S3 session and feature generation components."""
        self.bucket_name = getenv(BUCKET_NAME_ENV)
        if not self.bucket_name:
            logger.error("Bucket name not specified in the environment variables.")
            raise ValueError("BUCKET_NAME environment variable not set.")
        self.s3_session = self.create_s3_session()
        self.oracle = OraclesElixir(session=self.s3_session, bucket=self.bucket_name)
        self.imputer = EarlyGameStatsImputer()
        self.feature_generator = FeatureGenerator()
        self.rating_models = Ratings()

    @staticmethod
    def create_s3_session():
        """Create a boto3 session to access the S3 bucket using environment variables."""
        access_key = getenv(ACCESS_ID_ENV)
        secret_key = getenv(SECRET_ID_ENV)
        if not access_key or not secret_key:
            logger.error("AWS access credentials are not set in environment variables.")
            raise RuntimeError("Missing AWS credentials in environment variables.")

        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @staticmethod
    def get_years_to_process():
        """Get the years to process for data ingestion: current and the previous two years."""
        years_range = json_loader(YEARS_RANGE_PATH)["years_range"]
        current_year = dt.date.today().year
        return [str(year) for year in range(current_year, current_year - years_range, -1)]

    @staticmethod
    def _detect_buggy_games(data: pd.DataFrame) -> set:
        """Identify buggy games within the dataset."""
        incorrect_rows = set(data.groupby("gameid").filter(lambda x: len(x) != MAX_EXPECTED_ROWS)["gameid"].unique())
        incorrect_teams = set(
            data.groupby("gameid")
            .filter(lambda x: x["teamid"].nunique() != MAX_EXPECTED_TEAMS or "unknown team" in x["teamname"].values)[
                "gameid"
            ]
            .unique()
        )
        incorrect_players = set(
            data.groupby("gameid")
            .filter(
                lambda x: x["playerid"].nunique() != MAX_EXPECTED_PLAYERS or "unknown player" in x["playername"].values
            )["gameid"]
            .unique()
        )

        buggy_games = incorrect_rows.union(incorrect_teams, incorrect_players)
        return buggy_games

    @staticmethod
    def remove_buggy_games(data: pd.DataFrame) -> pd.DataFrame:
        """Remove games identified as buggy based on criteria in the INVALID_GAMES config and internal checks."""
        try:
            with open(INVALID_GAMES) as file:
                invalid_config = json.load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file for invalid games not found: {INVALID_GAMES}")
            raise FileNotFoundError(f"Configuration file not found: {INVALID_GAMES}") from None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {INVALID_GAMES}")
            raise json.JSONDecodeError("Invalid JSON format in the configuration file.") from None

        logger.info("Removing buggy games based on predefined criteria and additional checks.")
        invalid_games = invalid_config["invalid_games"]
        other_invalid_games = DataGenerator._detect_buggy_games(data)
        cleaned_data = data[~data["gameid"].isin(set(invalid_games).union(other_invalid_games))]

        logger.info(f"Removed buggy games. Remaining games: {len(cleaned_data)}\n")
        return cleaned_data

    def clean_and_store_data(self, data: pd.DataFrame):
        """Clean, sort, and store team and player data in interim directory."""
        self.team_data = self.oracle.clean_data(data, split_on="team")
        self.player_data = self.oracle.clean_data(data, split_on="player")

        self.team_data.sort_values(get_sorting_keys("team"), inplace=True)
        self.player_data.sort_values(get_sorting_keys("player"), inplace=True)

        self.team_data.to_parquet(INTERIM_TEAM_DATA, index=False)
        self.player_data.to_parquet(INTERIM_PLAYER_DATA, index=False)
        logger.info("Cleaned and stored interim data.\n")

    def ingest_data_from_s3(self):
        """Ingest data from S3 bucket and store it in the interim directory."""
        try:
            logger.info("Starting data ingestion from S3...")
            years = self.get_years_to_process()
            data = self.oracle.ingest_data(years=years)
            data.to_parquet(RAW_DATA, index=False)
            logger.info("Data ingestion completed and stored.\n")

            data = DataGenerator.remove_buggy_games(data)
            self.clean_and_store_data(data)

            return self.team_data, self.player_data

        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

    def _enrich_data_with_ratings(self):
        """Enrich data with all the associated ratings."""
        logger.info("Enriching data with ratings...")

        self.team_data = self.rating_models.compute_elo(df=self.team_data, entity="team")
        self.player_data = self.rating_models.compute_elo(df=self.player_data, entity="player")
        logger.info("Enriched data with elo.")

        self.team_data = self.rating_models.compute_glicko2(df=self.team_data, entity="team")
        self.player_data = self.rating_models.compute_glicko2(df=self.player_data, entity="player")
        logger.info("Enriched data with glicko2.")

        self.team_data = self.rating_models.compute_plackett_luce(df=self.team_data, entity="team")
        self.player_data = self.rating_models.compute_plackett_luce(df=self.player_data, entity="player")
        logger.info("Enriched data with plackett-luce.")

        self.team_data = self.rating_models.compute_trueskill(df=self.team_data, entity="team")
        self.player_data = self.rating_models.compute_trueskill(df=self.player_data, entity="player")
        logger.info("Enriched data with trueskill.")

        # self.team_data = self.rating_models.compute_whr(df=self.team_data, entity="team")
        # logger.info("Enriched team data with WHR.")

        self.team_data, belonging_league, league_elos = self.rating_models.compute_leagues_elo(
            df=self.team_data, entity="team"
        )

        league_elos.to_parquet(LEAGUE_ELO, index=False)
        belonging_league.to_parquet(TEAM_LEAGUES_MAPPING, index=False)
        logger.info("Enriched team data with leagues elo and stored the ratings.")

        logger.info("Completed enriching data with all ratings.\n")

    def _enrich_data_with_performance_metrics(self):
        """Enrich data with performance metrics."""
        logger.info("Enriching data with performance metrics...")

        self.team_data = PerformanceMetrics.add_entity_ema_statistics(self.team_data, entity="team")
        self.player_data = PerformanceMetrics.add_entity_ema_statistics(self.player_data, entity="player")
        logger.info("Enriched data with EMA statistics.")

        self.team_data = PerformanceMetrics.add_side_win_rate_ewm(self.team_data, entity="team")
        self.team_data = PerformanceMetrics.add_patch_win_rate_ewm(self.team_data, entity="team")
        self.team_data = PerformanceMetrics.add_season_win_rate_ewm(self.team_data, entity="team")
        logger.info("Enriched data with side and patch win rate.")

        logger.info("Completed enriching player and team data with all performance metrics.\n")

    def enrich_datasets(self):
        """Load, enrich, and store datasets for team and player-based analytics and predictions."""
        try:
            self.load_and_sort_data()

            # Impute missing data and generate new features
            self.generate_features_and_impute_data()

            # Enrich data with ratings and performance metrics
            self._enrich_data_with_ratings()
            self._enrich_data_with_performance_metrics()

            # Store the enriched data
            self.store_enriched_data()

            return self.team_data, self.player_data
        except Exception:
            raise

    def load_and_sort_data(self):
        """Load data from parquet and sort it based on predefined keys."""
        self.team_data = pd.read_parquet(INTERIM_TEAM_DATA)
        self.player_data = pd.read_parquet(INTERIM_PLAYER_DATA)
        self.team_data.sort_values(get_sorting_keys("team"), inplace=True)
        self.player_data.sort_values(get_sorting_keys("player"), inplace=True)

    def generate_features_and_impute_data(self):
        """Generate new features for team and player data and impute missing values where necessary."""
        self.team_data = self.feature_generator.generate_new_team_features(self.team_data)
        self.team_data = self.imputer.impute_data(self.team_data, "Team")

        self.player_data = self.feature_generator.generate_new_player_features(self.player_data)
        self.player_data = self.imputer.impute_data(self.player_data, "Player")

    def store_enriched_data(self):
        """Store enriched team and player data to parquet files."""
        self.team_data.to_parquet(PROCESSED_TEAMS, index=False)
        self.player_data.to_parquet(PROCESSED_PLAYERS, index=False)
        logger.info("Stored enriched data.")

    def extract_team_inference_data(self):
        """Extract the most recent team data for inference and store it."""
        config = json_loader(TRAINING_TEAM_CONFIG)
        self.extract_inference_data(self.team_data, config, "team")

    def extract_player_inference_data(self):
        """Extract the most recent player data for inference and store it."""
        config = json_loader(TRAINING_PLAYER_CONFIG)
        self.extract_inference_data(self.player_data, config, "player")

    def extract_inference_data(self, data: pd.DataFrame, config: dict, entity_type: str):
        """General method to extract inference data based on the specified configuration."""
        training_cols = config[f"{entity_type}_features"]
        before_cols = [col for col in training_cols if "_before" in col]
        inference_data = data[training_cols]
        inference_data = inference_data.rename(columns={col: col.replace("_before", "") for col in before_cols})

        inference_data.to_parquet(PROCESSED_DIR / f"training_{entity_type}_data.parquet", index=False)
        logger.info(f"Stored training {entity_type} data.")

    def extract_training_data(self):
        """Extract training data from the enriched datasets by invoking the extraction of team and player data."""
        logger.info("Extracting training data for teams and players...")
        self.extract_team_inference_data()
        self.extract_player_inference_data()
        logger.info("Completed extraction of training data.\n")

    def flatten_data(self, data: pd.DataFrame, config: str, entity_type: str):
        """Flatten the data to get the most recent record per entity."""
        flattened_entity_config = json_loader(config)
        flattened_cols = flattened_entity_config["flattened_cols"]

        # Filter columns that end with '_after'
        after_cols = list({col for col in flattened_cols if "_after" in col})

        # Get the most recent record for each entity
        flattened_entities = (
            data.sort_values([f"{entity_type}id", "date"])
            .groupby(f"{entity_type}id")
            .tail(1)
            .reset_index(drop=True)[flattened_cols]
        )

        # Rename columns by removing '_after' suffix
        flattened_entities = flattened_entities.rename(columns={col: col.replace("_after", "") for col in after_cols})

        # Define output path
        output_path = PROCESSED_DIR / f"flattened_{entity_type}s.parquet"

        # Store data to Parquet
        try:
            flattened_entities.to_parquet(output_path, index=False, engine="pyarrow")
        except Exception:
            # TODO!: find a way to avoid needing the fallback
            logger.warning("Failed to save to Parquet. Fallback to CSV and re-read.")
            fallback_csv_path = PROCESSED_DIR / f"flattened_{entity_type}s.csv"
            flattened_entities.to_csv(fallback_csv_path, index=False)
            read_back_data = pd.read_csv(fallback_csv_path)
            read_back_data.to_parquet(output_path, index=False, engine="pyarrow")
            os.remove(fallback_csv_path)

    def flatten_team_data(self):
        """Flatten the team_data dataframe to get the most recent record per team."""
        logger.info("Flattening team data...")
        self.flatten_data(self.team_data, FLATTENED_TEAM_CONFIG, "team")

    def flatten_player_data(self):
        """Flatten the player_data dataframe to get the most recent record per player per team."""
        logger.info("Flattening player data...")
        self.flatten_data(self.player_data, FLATTENED_PLAYER_CONFIG, "player")

    def flatten_inference_data(self):
        """Flatten both the team and player dataframes to get the most recent records."""
        logger.info("Starting to flatten inference data for teams and players...")
        self.flatten_team_data()
        self.flatten_player_data()
        logger.info("Completed flattening of inference data.\n")

    def run(self):
        """
        Run the complete data generation process including data ingestion, enrichment,
        training data extraction, and inference data flattening.
        """
        try:
            logger.info("Starting data generation process.\n")
            self.ingest_data_from_s3()
            self.enrich_datasets()
            self.extract_training_data()
            self.flatten_inference_data()
            logger.info("Data generation process completed successfully.")
        except Exception:
            raise


if __name__ == "__main__":
    generator = DataGenerator()
    logger.warning("Make sure all necessary configurations are set and all files are closed before starting.\n")

    start_time = dt.datetime.now()
    try:
        generator.run()
    except Exception as e:
        logger.error(f"Failed to complete the data generation process: {e}")
    else:
        elapsed_time = (dt.datetime.now() - start_time).total_seconds()
        logger.info(f"Data generation took {elapsed_time:.2f} seconds.\n")
