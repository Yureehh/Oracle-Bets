"""
Paths Folder

This file contains the paths to the data directories.
It is used to ensure that the necessary directories are created and available for data storage.
"""

import os
from pathlib import Path


def find_project_root(current_dir=None):
    current_dir = current_dir or os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
        # We found the marker file, return the current directory
        return current_dir
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        # We've reached the filesystem root without finding the marker
        raise FileNotFoundError("Project root marker not found")
    return Path(find_project_root(parent_dir))


# Define the base directory path relative to this file's location
BASE_DIR = find_project_root()

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

# Models storage
MODELS_DIR = BASE_DIR / "models"
