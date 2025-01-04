"""
Data Generator.

This script updates data once per day by ingesting data stored in an S3 bucket from Oracle Elixir,
enriching it with additional data based on computed ratings, and storing the enriched data.
Please visit and support www.oracleselixir.com. Tim provides an invaluable service to the League community.
"""

import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

import boto3
import fireducks.pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from src.feature_engineering.features_generator import FeatureGenerator
from src.feature_engineering.performance_features.performance_metrics import PerformanceMetrics
from src.feature_engineering.ratings_features.rating_models import Ratings
from src.ingestion.oracles_elixir import OraclesElixir
from src.utils.logger import instantiate_conf_logger, logger
from src.utils.paths import (
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
    INTERIM_PLAYER_DATA,
    INTERIM_TEAM_DATA,
    INVALID_GAMES,
    PROCESSED_DIR,
    PROCESSED_PLAYERS,
    PROCESSED_TEAMS,
    RAW_DATA,
    TRAINING_PLAYER_CONFIG,
    TRAINING_TEAM_CONFIG,
    YEARS_RANGE_PATH,
)
from src.utils.utils import get_sorting_keys, json_loader, safe_store_df_as_parquet

# -------------------------------------------------------------------------------------------
# 1. Load environment variables from .env file
# -------------------------------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------------------------------
# 2. Centralized Configuration (Constants) and Logging
# -------------------------------------------------------------------------------------------
BUCKET_NAME_ENV = "BUCKET_NAME"
ACCESS_ID_ENV = "ACCESS_ID"
SECRET_ID_ENV = "SECRET_ID"  # pragma: allowlist secret
MAX_EXPECTED_PLAYERS = 10
MAX_EXPECTED_TEAMS = 2
MAX_EXPECTED_ROWS = 12
data_pipeline_logger = instantiate_conf_logger("data_pipeline")


# -------------------------------------------------------------------------------------------
# 3. Logging Decorator
# -------------------------------------------------------------------------------------------
def log_function_call(logger_instance):
    """Decorator to wrap function calls with consistent logging before/after execution."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger_instance.info(f"Starting {func.__name__}...")
            try:
                result = func(*args, **kwargs)
                logger_instance.info(f"Completed {func.__name__}.\n")
                return result
            except Exception:
                logger_instance.error(f"Error in {func.__name__}")
                raise

        return wrapper

    return decorator


# -------------------------------------------------------------------------------------------
# 4. Parallel Enrichment Helper
# -------------------------------------------------------------------------------------------
def parallelize_enrichment(func, df_team: pd.DataFrame, df_player: pd.DataFrame, entity_team: str, entity_player: str):
    """
    Runs a given function in parallel on team and player data when the function signature
    is the same except for the 'entity' parameter.
    """
    with ThreadPoolExecutor() as executor:
        future_team = executor.submit(func, df_team, entity=entity_team)
        future_player = executor.submit(func, df_player, entity=entity_player)
        team_result = future_team.result()
        player_result = future_player.result()
    return team_result, player_result


# -------------------------------------------------------------------------------------------
# 5. Helper: Check Missing Columns
# -------------------------------------------------------------------------------------------
def check_missing_columns(data: pd.DataFrame, required_columns: List[str], entity_type: str) -> None:
    """Checks if the required columns are present in the DataFrame. Raises ValueError if columns are missing."""
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing columns for {entity_type}: {missing_cols}")
        raise ValueError(f"Missing columns in data for {entity_type}: {missing_cols}")


# -------------------------------------------------------------------------------------------
# 6. Main DataGenerator Class
# -------------------------------------------------------------------------------------------
@dataclass
class DataGenerator:
    """Class responsible for ingesting, cleaning, enriching, and storing League of Legends data."""

    team_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    bucket_name: str = field(init=False)
    s3_session: boto3.Session = field(init=False)
    oracle: OraclesElixir = field(init=False)
    feature_generator: FeatureGenerator = field(init=False)
    rating_models: Ratings = field(init=False)

    def __post_init__(self):
        """Initialize DataGenerator with S3 session and feature generation components."""
        try:
            self.load_bucket()
            self.s3_session = self.create_s3_session()
            self.oracle = OraclesElixir(session=self.s3_session, bucket=self.bucket_name)
            self.feature_generator = FeatureGenerator()
            self.rating_models = Ratings()
            logger.info("DataGenerator initialized successfully.\n")
        except Exception as e:
            logger.error(f"Failed to initialize DataGenerator: {e}")
            raise

    def load_bucket(self) -> None:
        """Load configuration from environment variables."""
        self.bucket_name = os.getenv(BUCKET_NAME_ENV)
        if not self.bucket_name:
            logger.error("Bucket name not specified in environment variables.")
            raise ValueError("BUCKET_NAME environment variable not set.")
        logger.info(f"Loaded bucket name: {self.bucket_name}")

    @staticmethod
    def create_s3_session() -> boto3.Session:
        """
        Create a boto3 session to access the S3 bucket using environment variables.

        Raises:
            RuntimeError: If AWS credentials are missing.
        """
        access_key = os.getenv(ACCESS_ID_ENV)
        secret_key = os.getenv(SECRET_ID_ENV)
        if not access_key or not secret_key:
            logger.error("AWS access credentials are not set in environment variables.")
            raise RuntimeError("Missing AWS credentials in environment variables.")

        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        logger.info("Created AWS S3 session successfully.")
        return session

    @staticmethod
    def get_years_to_process() -> List[str]:
        """
        Get the years to process for data ingestion: current and the previous N years.

        Returns:
            List[str]: List of years to process.

        Raises:
            Exception: If years range configuration fails to load.
        """
        try:
            years_config = json_loader(YEARS_RANGE_PATH)
            years_range = years_config["years_range"]
            current_year = dt.date.today().year
            years = [str(year) for year in range(current_year, current_year - years_range, -1)]
            logger.info(f"Years to process: {years}")
            return years
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading years range configuration: {e}")
            raise ValueError(f"Failed to load years range configuration: {e}") from e

    @staticmethod
    def _detect_buggy_games(data: pd.DataFrame) -> Set[str]:
        """
        Identify buggy games within the dataset using vectorized logic
        to minimize repeated filtering passes.
        """
        try:
            grouped = data.groupby("gameid")
            buggy_games = set(
                grouped.filter(
                    lambda x: (
                        len(x) != MAX_EXPECTED_ROWS
                        or x["teamid"].nunique() != MAX_EXPECTED_TEAMS
                        or x["playerid"].nunique() != MAX_EXPECTED_PLAYERS
                        or "unknown team" in x["teamname"].values
                        or "unknown player" in x["playername"].values
                    )
                )["gameid"].unique()
            )
            logger.info(f"Detected {len(buggy_games)} buggy games.")
            return buggy_games
        except Exception as e:
            logger.error(f"Error detecting buggy games: {e}")
            raise

    @staticmethod
    def remove_buggy_games(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove games identified as buggy based on criteria in the INVALID_GAMES config and internal checks.

        Raises:
            Exception: If the invalid games configuration fails to load.
        """
        try:
            invalid_config = json_loader(INVALID_GAMES)
            invalid_games = set(invalid_config["invalid_games"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading configuration file {INVALID_GAMES}: {e}")
            raise KeyError(f"Failed to load invalid games configuration: {e}") from e

        other_invalid_games = DataGenerator._detect_buggy_games(data)
        all_invalid_games = invalid_games.union(other_invalid_games)
        cleaned_data = data[~data["gameid"].isin(all_invalid_games)].reset_index(drop=True)

        logger.info(f"Removed {len(all_invalid_games)} invalid games. Remaining rows: {len(cleaned_data)}")
        return cleaned_data

    @log_function_call(logger)
    def clean_and_store_data(self, data: pd.DataFrame) -> None:
        """Clean, sort, and store team and player data in interim directory."""
        cleaned_data = self.remove_buggy_games(data)

        # Oracle clean_data method
        self.team_data = self.oracle.clean_data(cleaned_data, split_on="team").sort_values(get_sorting_keys("team"))
        self.player_data = self.oracle.clean_data(cleaned_data, split_on="player").sort_values(
            get_sorting_keys("player")
        )

        safe_store_df_as_parquet(self.team_data, INTERIM_TEAM_DATA, logger)
        safe_store_df_as_parquet(self.player_data, INTERIM_PLAYER_DATA, logger)
        logger.info("Cleaned and stored interim data.")

    def ingest_data_from_s3(self) -> pd.DataFrame:
        """Ingest data from S3 bucket and return the raw data."""
        try:
            logger.info("Starting data ingestion from S3...")
            years = self.get_years_to_process()
            data = self.oracle.ingest_data(years=years)
            safe_store_df_as_parquet(data, RAW_DATA, logger)
            logger.info("Data ingestion completed and stored.\n")
            return data
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS S3 error during data ingestion: {e}")
            raise ClientError(f"Failed to ingest data from S3 due to AWS error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

    def _enrich_data_with_ratings(self) -> None:
        """Enrich data with all the associated ratings in parallel where possible."""
        try:
            logger.info("Enriching data with ratings...")
            # Some rating computations depend on others (e.g. compute_leagues_elo first),
            # so we run them sequentially where needed, then parallelize others.

            # (1) League ELO
            self.team_data = self.rating_models.compute_leagues_elo(self.team_data)
            logger.info("Completed enriching data with Leagues ELO.")
            data_pipeline_logger.info("Completed enriching data with Leagues ELO.")

            # (2) ELO
            self.team_data, self.player_data = parallelize_enrichment(
                self.rating_models.compute_elo,
                self.team_data,
                self.player_data,
                "team",
                "player",
            )
            logger.info("Completed enriching data with ELO.")
            data_pipeline_logger.info("Completed enriching data with ELO.")

            # (3) Glicko2
            self.team_data, self.player_data = parallelize_enrichment(
                self.rating_models.compute_glicko2,
                self.team_data,
                self.player_data,
                "team",
                "player",
            )
            logger.info("Completed enriching data with Glicko2.")
            data_pipeline_logger.info("Completed enriching data with Glicko2.")

            # (4) Plackett-Luce
            self.team_data, self.player_data = parallelize_enrichment(
                self.rating_models.compute_plackett_luce,
                self.team_data,
                self.player_data,
                "team",
                "player",
            )
            logger.info("Completed enriching data with Plackett-Luce.")
            data_pipeline_logger.info("Completed enriching data with Plackett-Luce.")

            # (5) TrueSkill
            self.team_data, self.player_data = parallelize_enrichment(
                self.rating_models.compute_trueskill,
                self.team_data,
                self.player_data,
                "team",
                "player",
            )
            logger.info("Completed enriching data with TrueSkill.")
            data_pipeline_logger.info("Completed enriching data with TrueSkill.")

            logger.info("Completed enriching data with all ratings.")
        except Exception as e:
            logger.error(f"Failed to enrich data with ratings: {e}")
            raise

    def _enrich_data_with_performance_metrics(self) -> None:
        """Enrich data with performance metrics in parallel where possible."""
        try:
            logger.info("Enriching data with performance metrics...")

            # Example of parallelizing the same function calls
            self.team_data, self.player_data = parallelize_enrichment(
                PerformanceMetrics.add_entity_ema_statistics, self.team_data, self.player_data, "team", "player"
            )
            logger.info("Completed enriching data with EMA statistics.")
            data_pipeline_logger.info("Completed enriching data with EMA statistics.")

            # Some metrics are only for teams in this example
            self.team_data = PerformanceMetrics.add_side_win_rate_ewm(self.team_data, entity="team")
            self.team_data = PerformanceMetrics.add_patch_win_rate_ewm(self.team_data, entity="team")
            self.team_data = PerformanceMetrics.add_season_win_rate_ewm(self.team_data, entity="team")
            logger.info("Completed enriching data with win rate EWM metrics.")
            data_pipeline_logger.info("Completed enriching data with win rate EWM metrics.")

            logger.info("Completed enriching data with performance metrics.")
        except Exception as e:
            logger.error(f"Failed to enrich data with performance metrics: {e}")
            raise

    @log_function_call(logger)
    def enrich_datasets(self) -> None:
        """Load, enrich, and store datasets for team and player-based analytics and predictions."""
        self.load_and_sort_data()
        self.generate_features()
        self._enrich_data_with_ratings()
        self._enrich_data_with_performance_metrics()
        self.store_enriched_data()

    def load_and_sort_data(self) -> None:
        """Load data from parquet and sort it based on predefined keys."""
        try:
            if not INTERIM_TEAM_DATA.exists() or not INTERIM_PLAYER_DATA.exists():
                raise FileNotFoundError("Interim data files not found.")

            self.team_data = pd.read_parquet(INTERIM_TEAM_DATA, engine="fastparquet")
            self.player_data = pd.read_parquet(INTERIM_PLAYER_DATA, engine="fastparquet")

            self.team_data.sort_values(get_sorting_keys("team"), inplace=True)
            self.player_data.sort_values(get_sorting_keys("player"), inplace=True)
            logger.info("Loaded and sorted interim data.")
        except Exception as e:
            logger.error(f"Failed to load and sort data: {e}")
            raise

    def generate_features(self) -> None:
        """Generate new features for team and player data."""
        try:
            self.team_data = self.feature_generator.generate_new_team_features(self.team_data)
            self.player_data = self.feature_generator.generate_new_player_features(self.player_data)
            logger.info("Generated new features for team and player data.\n")
        except Exception as e:
            logger.error(f"Failed to generate features: {e}")
            raise

    def store_enriched_data(self) -> None:
        """Store enriched team and player data to parquet files."""
        try:
            safe_store_df_as_parquet(self.team_data, PROCESSED_TEAMS, logger)
            safe_store_df_as_parquet(self.player_data, PROCESSED_PLAYERS, logger)
            logger.info("Stored enriched data.")
        except Exception as e:
            logger.error(f"Failed to store enriched data: {e}")
            raise

    # -------------------------------------------------------
    # Combined Logic for Extracting Training / Flattening Data
    # -------------------------------------------------------
    def extract_inference_data(
        self, data: pd.DataFrame, config_path: Path, entity_type: str, output_prefix: str
    ) -> None:
        """
        Extract or flatten data based on the specified configuration file.
        Use 'output_prefix' to define where to store the resulting parquet.
        """
        try:
            config = json_loader(config_path)

            # Decide which columns we want: training or flattened
            required_cols_key = "flattened_cols" if "flattened" in output_prefix else f"{entity_type}_features"
            required_cols = config[required_cols_key]

            # Check for missing columns
            check_missing_columns(data, required_cols, entity_type)

            # Distinguish between "flatten" vs "training" logic
            if "flattened" in output_prefix:
                # Flatten approach
                after_cols = [col for col in required_cols if "_after" in col]
                # Sort and group to get the most recent
                flattened = (
                    data.sort_values([f"{entity_type}id", "date"])
                    .groupby(f"{entity_type}id")
                    .tail(1)
                    .reset_index(drop=True)[required_cols]
                )
                # Rename columns to remove "_after"
                flattened = flattened.rename(columns={col: col.replace("_after", "") for col in after_cols})
                output_path = PROCESSED_DIR / f"{output_prefix}_{entity_type}s.parquet"
                safe_store_df_as_parquet(flattened, output_path, logger)
                logger.info(f"Stored flattened {entity_type} data.")
            else:
                # Training approach
                before_cols = [col for col in required_cols if "_before" in col]
                inference_data = data[required_cols].copy()
                inference_data = inference_data.rename(columns={col: col.replace("_before", "") for col in before_cols})
                output_path = PROCESSED_DIR / f"{output_prefix}_{entity_type}_data.parquet"
                safe_store_df_as_parquet(inference_data, output_path, logger)
                logger.info(f"Stored training {entity_type} data.")

        except Exception as e:
            logger.error(f"Failed to process inference data for {entity_type}: {e}")
            raise

    @log_function_call(logger)
    def extract_both_training_data(self) -> None:
        """Extract training data for both teams and players."""
        self.extract_inference_data(
            data=self.team_data, config_path=TRAINING_TEAM_CONFIG, entity_type="team", output_prefix="training"
        )
        self.extract_inference_data(
            data=self.player_data, config_path=TRAINING_PLAYER_CONFIG, entity_type="player", output_prefix="training"
        )

    @log_function_call(logger)
    def flatten_both_inference_data(self) -> None:
        """Flatten both the team and player dataframes to get the most recent records."""
        self.extract_inference_data(
            data=self.team_data, config_path=FLATTENED_TEAM_CONFIG, entity_type="team", output_prefix="flattened"
        )
        self.extract_inference_data(
            data=self.player_data, config_path=FLATTENED_PLAYER_CONFIG, entity_type="player", output_prefix="flattened"
        )

    @log_function_call(logger)
    def run(self) -> None:
        """
        Run the complete data generation process including data ingestion, cleaning,
        enrichment, training data extraction, and inference data flattening.
        """
        # raw_data = self.ingest_data_from_s3()
        raw_data = pd.read_parquet(RAW_DATA, engine="fastparquet")
        self.clean_and_store_data(raw_data)
        self.enrich_datasets()
        self.extract_both_training_data()
        self.flatten_both_inference_data()
        logger.info("Data generation process completed successfully.")


# -------------------------------------------------------------------------------------------
# 7. Entry Point: Optionally Integrate cProfile or memory_profiler if needed
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    generator = DataGenerator()
    data_pipeline_logger.info("Data Pipeline Logger initialized.")

    start_time = dt.datetime.now()
    try:
        generator.run()
    except Exception:
        logger.error("Data generation process failed")
    else:
        elapsed_time = (dt.datetime.now() - start_time).total_seconds()
        logger.info(f"Data generation took {elapsed_time:.2f} seconds.\n")
