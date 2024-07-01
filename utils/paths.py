"""
Paths Folder

This file contains the paths to the data directories.
It is used to ensure that the necessary directories are created and available for data storage.
"""

import os
from pathlib import Path

# Define the base directory path relative to this file's location
BASE_DIR = Path(os.getcwd())

# Directories for data storage
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
REPORTS_DIR = BASE_DIR / "reports"

# Data storage directories
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Config file path

DISCORD_CONFIG = CONFIG_DIR / "discord_config.json"
DATA_INGESTION_DIR = CONFIG_DIR / "data_ingestion"
DEFAULT_MODELS_PARAMETERS = CONFIG_DIR / "default_models_parameters.json"
OUTCOME_TRAINING_AND_INPUT_COLS_DIR = CONFIG_DIR / "train_and_input_cols" / "outcome"
GAMELENGTH_TRAINING_AND_INPUT_COLS_DIR = CONFIG_DIR / "train_and_input_cols" / "gamelength"

# Data Ingestion
YEARS_RANGE_PATH = DATA_INGESTION_DIR / "years_to_consider.json"
INVALID_GAMES = DATA_INGESTION_DIR / "invalid_games.json"
IMPORT_COLUMNS = DATA_INGESTION_DIR / "import_columns.json"
TEAM_REPLACEMENTS = DATA_INGESTION_DIR / "team_replacements.json"
CONSIDERED_LEAGUES = DATA_INGESTION_DIR / "considered_leagues.json"
FEATURES_TO_IMPUTE = DATA_INGESTION_DIR / "features_to_impute.json"

# External Bets
EXTERNAL_BETS = EXTERNAL_DIR / "bets.csv"

# Raw data
RAW_DATA = RAW_DIR / "raw_data.parquet"

# Interim data
INTERIM_TEAM_DATA = INTERIM_DIR / "team_data.parquet"
INTERIM_PLAYER_DATA = INTERIM_DIR / "player_data.parquet"

# Training data
TRAINING_TEAM_DATA = PROCESSED_DIR / "training_team_data.parquet"
TRAINING_PLAYER_DATA = PROCESSED_DIR / "training_player_data.parquet"

# Processed data
PROCESSED_TEAMS = PROCESSED_DIR / "team_data.parquet"
PROCESSED_PLAYERS = PROCESSED_DIR / "player_data.parquet"
TEAM_LEAGUES_MAPPING = PROCESSED_DIR / "team_league_mapping.parquet"
LEAGUE_ELO = PROCESSED_DIR / "league_elo.parquet"
SCHEDULE = PROCESSED_DIR / "schedule.parquet"

# Flattened data
FLATTENED_TEAMS = PROCESSED_DIR / "flattened_teams.parquet"
FLATTENED_PLAYERS = PROCESSED_DIR / "flattened_players.parquet"

# Training and input cols for outcome prediction
TRAINING_TEAM_CONFIG = OUTCOME_TRAINING_AND_INPUT_COLS_DIR / "training_team_config.json"
FLATTENED_TEAM_CONFIG = OUTCOME_TRAINING_AND_INPUT_COLS_DIR / "flattened_team_config.json"
TRAINING_PLAYER_CONFIG = OUTCOME_TRAINING_AND_INPUT_COLS_DIR / "training_player_config.json"
FLATTENED_PLAYER_CONFIG = OUTCOME_TRAINING_AND_INPUT_COLS_DIR / "flattened_player_config.json"

# Game length prediction
GAMELENGTH_TRAINING_TEAM_CONFIG = GAMELENGTH_TRAINING_AND_INPUT_COLS_DIR / "training_team_config.json"
GAMELENGTH_FLATTENED_TEAM_CONFIG = GAMELENGTH_TRAINING_AND_INPUT_COLS_DIR / "flattened_team_config.json"
GAMELENGTH_TRAINING_PLAYER_CONFIG = GAMELENGTH_TRAINING_AND_INPUT_COLS_DIR / "training_player_config.json"
GAMELENGTH_FLATTENED_PLAYER_CONFIG = GAMELENGTH_TRAINING_AND_INPUT_COLS_DIR / "flattened_player_config.json"

# Models storage
WHOLE_HISTORY_RATING_PATH = MODELS_DIR / "whr.pkl"
PREDICTION_MODEL_PATH = MODELS_DIR / "LightGBM.pkl"
LIGHTGBM_CATEGORICAL_FEATURES = MODELS_DIR / "LightGBM_categorical_features.pkl"
LIGHTGBM_FINAL_FEATURES = MODELS_DIR / "LightGBM_final_features.pkl"
LIGHTGBM_BEST_HYPERPARAMETERS = MODELS_DIR / "LightGBM_best_hyperparameters.pkl"

# Reports storage
LOGS_DIR = REPORTS_DIR / "logs"
FIGURES_DIR = REPORTS_DIR / "figures"
FEATURE_IMP_DIR = FIGURES_DIR / "feature_importances"
INSIGHTS_DIR = REPORTS_DIR / "insights"

# Notebooks storage
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


# Logs storage
MODELS_LOGS = LOGS_DIR / "gbdt_models.log"
EARLY_GAME_INPUTING_LOGS = LOGS_DIR / "early_game_inputing.log"

# for every directory, check if it exists, if not, create it
for directory in [
    DATA_DIR,
    MODELS_DIR,
    CONFIG_DIR,
    REPORTS_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    DATA_INGESTION_DIR,
    OUTCOME_TRAINING_AND_INPUT_COLS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    FEATURE_IMP_DIR,
    INSIGHTS_DIR,
    NOTEBOOKS_DIR,
]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
