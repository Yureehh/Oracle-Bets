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
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Config file path
CONFIG_DIR = BASE_DIR / "config"
INVALID_GAMES = CONFIG_DIR / "invalid_games.json"
IMPORT_COLUMNS = CONFIG_DIR / "import_columns.json"
DEFAULT_PARAMETERS = CONFIG_DIR / "default_parameters.json"
FLATTENED_TEAM_CONFIG = CONFIG_DIR / "flattened_team_config.json"
FLATTENED_PLAYER_CONFIG = CONFIG_DIR / "flattened_player_config.json"
CONSIDERED_LEAGUES = CONFIG_DIR / "considered_leagues.json"
DISCORD_CONFIG = CONFIG_DIR / "discord_config.json"

# Models storage
MODELS_DIR = BASE_DIR / "models"
EGPM_DOM_LOGISTIC = MODELS_DIR / "egpm_dom_logistic_regression.pkl"
MIXED_VALIDATOR_WEIGHTS = MODELS_DIR / "mixed_validator_weights.pkl"

# Reports storage
REPORTS_DIR = BASE_DIR / "reports"

# Figures storage
FIGURES_DIR = REPORTS_DIR / "figures"

# Model metrics storage
METRICS_DIR = REPORTS_DIR / "metrics"

# Logs storage
LOGS_DIR = REPORTS_DIR / "logs"
EARLY_GAME_INPUTING_LOGS = LOGS_DIR / "early_game_inputing.log"
MODELS_LOGS = LOGS_DIR / "models_ensemble.log"
