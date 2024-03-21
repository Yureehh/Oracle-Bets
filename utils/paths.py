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

# Models storage
MODELS_DIR = BASE_DIR / "models"
