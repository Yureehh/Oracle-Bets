"""
Paths Module.

This module defines and manages file system paths used across the project.
It ensures that the necessary directories are created and available for data storage.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.

    Args:
        directory (Path): The directory path to create.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise e


# Retrieve the base directory from an environment variable or default to the current working directory
BASE_DIR: Path = Path(os.getenv("BASE_DIR", os.getcwd())).resolve()

# Directories for data storage
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
CONFIG_DIR: Path = BASE_DIR / "config"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
LOGS_DIR: Path = BASE_DIR / "logs"

# Data storage subdirectories
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PROCESSED_TEAMS_DIR: Path = PROCESSED_DIR / "teams"
PROCESSED_PLAYERS_DIR: Path = PROCESSED_DIR / "players"

# Configuration file paths
DISCORD_CONFIG: Path = CONFIG_DIR / "discord_config.json"
DATA_INGESTION_DIR: Path = CONFIG_DIR / "data_ingestion"
TARGET_FEATURES: Path = CONFIG_DIR / "target_features.json"
TRAINING_AND_INPUT_COLS_DIR: Path = CONFIG_DIR / "training"

# Data Ingestion paths
YEARS_RANGE_PATH: Path = DATA_INGESTION_DIR / "years_range.json"
INVALID_GAMES: Path = DATA_INGESTION_DIR / "invalid_games.json"
IMPORT_COLUMNS: Path = DATA_INGESTION_DIR / "import_columns.json"
TEAM_REPLACEMENTS: Path = DATA_INGESTION_DIR / "team_name_replacements.json"
CONSIDERED_LEAGUES: Path = DATA_INGESTION_DIR / "considered_leagues.json"

# Raw data
RAW_DIR: Path = DATA_DIR / "raw"
RAW_DATA: Path = RAW_DIR / "raw_data.parquet"

# Interim data
INTERIM_TEAM_DATA: Path = INTERIM_DIR / "team_data.parquet"
INTERIM_PLAYER_DATA: Path = INTERIM_DIR / "player_data.parquet"

# Processed data
PROCESSED_TEAMS: Path = PROCESSED_TEAMS_DIR / "team_data.parquet"
PROCESSED_PLAYERS: Path = PROCESSED_PLAYERS_DIR / "player_data.parquet"
TEAM_LEAGUES_MAPPING: Path = PROCESSED_DIR / "team_league_mapping.parquet"
LEAGUE_ELO: Path = PROCESSED_DIR / "league_elo.parquet"
SCHEDULE: Path = PROCESSED_DIR / "schedule.parquet"

# Flattened data
FLATTENED_TEAMS: Path = PROCESSED_TEAMS_DIR / "flattened_teams.parquet"
FLATTENED_PLAYERS: Path = PROCESSED_PLAYERS_DIR / "flattened_players.parquet"

# Training data
TRAINING_TEAM_DATA: Path = PROCESSED_TEAMS_DIR / "training_team_data.parquet"
TRAINING_PLAYER_DATA: Path = PROCESSED_PLAYERS_DIR / "training_player_data.parquet"

# Training and input columns for outcome prediction
TRAINING_TEAM_CONFIG: Path = TRAINING_AND_INPUT_COLS_DIR / "training_team_config.json"
FLATTENED_TEAM_CONFIG: Path = TRAINING_AND_INPUT_COLS_DIR / "flattened_team_config.json"
TRAINING_PLAYER_CONFIG: Path = TRAINING_AND_INPUT_COLS_DIR / "training_player_config.json"
FLATTENED_PLAYER_CONFIG: Path = TRAINING_AND_INPUT_COLS_DIR / "flattened_player_config.json"

# Artifacts storage
MODEL_ARTIFACTS: Path = MODELS_DIR / "artifacts"
WHOLE_HISTORY_RATING_PATH: Path = MODEL_ARTIFACTS / "whr.pkl"
OUTCOME_PREDICTION_MODEL_PATH: Path = MODEL_ARTIFACTS / "OutcomePrediction.pkl"
OUTCOME_PREDICTION_CATEGORICAL_FEATURES: Path = MODEL_ARTIFACTS / "OutcomePrediction_categorical_features.pkl"
OUTCOME_PREDICTION_FINAL_FEATURES: Path = MODEL_ARTIFACTS / "OutcomePrediction_final_features.pkl"
OUTCOME_PREDICTION_BEST_HYPERPARAMETERS: Path = MODEL_ARTIFACTS / "OutcomePrediction_best_hyperparameters.pkl"
GAMELENGTH_PREDICTION_MODEL_PATH: Path = MODEL_ARTIFACTS / "GamelengthPrediction.pkl"
GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES: Path = MODEL_ARTIFACTS / "GamelengthPrediction_categorical_features.pkl"
GAMELENGTH_PREDICTION_FINAL_FEATURES: Path = MODEL_ARTIFACTS / "GamelengthPrediction_final_features.pkl"
GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS: Path = MODEL_ARTIFACTS / "GamelengthPrediction_best_hyperparameters.pkl"

# Hyperparams
HYPERPARAMETERS: Path = MODELS_DIR / "hyperparameters"
DEFAULT_MODELS_PARAMETERS: Path = HYPERPARAMETERS / "default_models_parameters.json"
BEST_HYPERPARAMETERS: Path = HYPERPARAMETERS / "best_hyperparams"
LEAGUES_ELO_HYPERPARAMETERS: Path = BEST_HYPERPARAMETERS / "leagues_elo_hyperparameters.json"
ENTITY_ELO_HYPERPARAMETERS: Path = BEST_HYPERPARAMETERS / "entity_elo_hyperparameters.json"

# Reports storage
FIGURES_DIR: Path = REPORTS_DIR / "figures"
FEATURE_IMP_DIR: Path = FIGURES_DIR / "feature_importances"
INSIGHTS_DIR: Path = REPORTS_DIR / "evaluation_insights"

# List of directories to ensure existence
directories: List[Path] = [
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CONFIG_DIR,
    NOTEBOOKS_DIR,
    LOGS_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROCESSED_TEAMS_DIR,
    PROCESSED_PLAYERS_DIR,
    RAW_DIR,
    MODEL_ARTIFACTS,
    FIGURES_DIR,
    FEATURE_IMP_DIR,
    INSIGHTS_DIR,
    DATA_INGESTION_DIR,
    TRAINING_AND_INPUT_COLS_DIR,
    HYPERPARAMETERS,
    BEST_HYPERPARAMETERS,
    FEATURE_IMP_DIR,
    INSIGHTS_DIR,
]

# Create directories if they do not exist
for directory in directories:
    create_directory(directory)
