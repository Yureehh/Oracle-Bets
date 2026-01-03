"""
Paths module.

Centralises every on-disk location used by Oracle-Bets and eagerly ensures the
directories exist.  Nothing outside this module should hard-code a filesystem
path string.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
load_dotenv()  # makes BASE_DIR overridable via .env / process env


# --------------------------------------------------------------------------- #
# Helpers / explicit error model
# --------------------------------------------------------------------------- #
class DirectoryCreationError(OSError):
    """Raised when a required directory cannot be created."""


def _create_directory(directory: Path) -> None:
    """Create *directory* (recursively) iff it does not already exist."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"Error creating directory: {directory}"
        raise DirectoryCreationError(msg) from exc


# --------------------------------------------------------------------------- #
# Base & top-level dirs
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(os.getenv("BASE_DIR", Path.cwd())).resolve()

DATA_DIR: Final = BASE_DIR / "data"
CONFIG_DIR: Final = BASE_DIR / "config"
MODELS_DIR: Final = BASE_DIR / "models"
NOTEBOOKS_DIR: Final = BASE_DIR / "notebooks"
REPORTS_DIR: Final = BASE_DIR / "reports"
LOGS_DIR: Final = BASE_DIR / "logs"

# --------------------------------------------------------------------------- #
# Data sub-directories
# --------------------------------------------------------------------------- #
RAW_DIR: Final = DATA_DIR / "raw"
INTERIM_DIR: Final = DATA_DIR / "interim"
PROCESSED_DIR: Final = DATA_DIR / "processed"
PROCESSED_TEAMS_DIR: Final = PROCESSED_DIR / "teams"
PROCESSED_PLAYERS_DIR: Final = PROCESSED_DIR / "players"

# --------------------------------------------------------------------------- #
# Configuration files
# --------------------------------------------------------------------------- #
DISCORD_CONFIG: Final = CONFIG_DIR / "discord_config.json"
DATA_INGESTION_DIR: Final = CONFIG_DIR / "data_ingestion"
LEAGUES_INGESTION: Final = DATA_INGESTION_DIR / "leagues_handling"
TARGET_FEATURES: Final = CONFIG_DIR / "target_features.json"
TRAINING_AND_INPUT_COLS_DIR: Final = CONFIG_DIR / "training"

YEARS_RANGE_PATH: Final = DATA_INGESTION_DIR / "years_range.json"
IMPORT_COLUMNS: Final = DATA_INGESTION_DIR / "import_columns.json"
TEAM_REPLACEMENTS_AND_INVALID_GAMES: Final = (
    DATA_INGESTION_DIR / "team_name_replacements_and_invalid_games.json"
)
CONSIDERED_LEAGUES: Final = LEAGUES_INGESTION / "considered_leagues.json"
LEAGUE_PRIOR_SETTINGS: Final = LEAGUES_INGESTION / "league_prior_settings.json"

# --------------------------------------------------------------------------- #
# Concrete data artefacts
# --------------------------------------------------------------------------- #
RAW_DATA: Final = RAW_DIR / "raw_data.parquet"

INTERIM_TEAM_DATA: Final = INTERIM_DIR / "team_data.parquet"
INTERIM_PLAYER_DATA: Final = INTERIM_DIR / "player_data.parquet"

PROCESSED_TEAMS: Final = PROCESSED_TEAMS_DIR / "team_data.parquet"
PROCESSED_PLAYERS: Final = PROCESSED_PLAYERS_DIR / "player_data.parquet"
SCHEDULE: Final = PROCESSED_DIR / "schedule.parquet"

FLATTENED_TEAMS: Final = PROCESSED_TEAMS_DIR / "flattened_teams.parquet"
FLATTENED_PLAYERS: Final = PROCESSED_PLAYERS_DIR / "flattened_players.parquet"

TRAINING_TEAM_DATA: Final = PROCESSED_TEAMS_DIR / "training_teams.parquet"
TRAINING_PLAYER_DATA: Final = PROCESSED_PLAYERS_DIR / "training_players.parquet"

TRAINING_TEAM_CONFIG: Final = TRAINING_AND_INPUT_COLS_DIR / "training_team_config.json"
TRAINING_TEAM_CONFIG_COMPACT: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_team_config_compact.json"
)
FLATTENED_TEAM_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "flattened_team_config.json"
)
TRAINING_PLAYER_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_player_config.json"
)
TRAINING_PLAYER_CONFIG_COMPACT: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_player_config_compact.json"
)
FLATTENED_PLAYER_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "flattened_player_config.json"
)

# --------------------------------------------------------------------------- #
# Model artefacts & hyper-parameters
# --------------------------------------------------------------------------- #
MODEL_ARTIFACTS: Final = MODELS_DIR / "artifacts"
LEAGUE_ELO: Final = MODEL_ARTIFACTS / "league_elo.parquet"
TEAM_LEAGUES_MAPPING: Final = MODEL_ARTIFACTS / "team_league_mapping.parquet"
LEAGUE_STRENGTH_PRIORS: Final = MODEL_ARTIFACTS / "league_strength_priors.json"
WHOLE_HISTORY_RATING_PATH: Final = MODEL_ARTIFACTS / "whr.pkl"
OUTCOME_PREDICTION_MODEL_PATH: Final = (
    MODEL_ARTIFACTS / "OutcomePrediction" / "OutcomePrediction.pkl"
)
OUTCOME_PREDICTION_CATEGORICAL_FEATURES: Final = (
    MODEL_ARTIFACTS / "OutcomePrediction" / "OutcomePrediction_categorical_features.pkl"
)
OUTCOME_PREDICTION_FINAL_FEATURES: Final = (
    MODEL_ARTIFACTS / "OutcomePrediction" / "OutcomePrediction_final_features.pkl"
)
OUTCOME_PREDICTION_BEST_HYPERPARAMETERS: Final = (
    MODEL_ARTIFACTS / "OutcomePrediction" / "OutcomePrediction_best_hyperparameters.pkl"
)
GAMELENGTH_PREDICTION_MODEL_PATH: Final = (
    MODEL_ARTIFACTS / "GamelengthPrediction" / "GamelengthPrediction.pkl"
)
GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES: Final = (
    MODEL_ARTIFACTS
    / "GamelengthPrediction"
    / "GamelengthPrediction_categorical_features.pkl"
)
GAMELENGTH_PREDICTION_FINAL_FEATURES: Final = (
    MODEL_ARTIFACTS / "GamelengthPrediction" / "GamelengthPrediction_final_features.pkl"
)
GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS: Final = (
    MODEL_ARTIFACTS
    / "GamelengthPrediction"
    / "GamelengthPrediction_best_hyperparameters.pkl"
)
TOTAL_KILLS_PREDICTION_MODEL_PATH: Final = (
    MODEL_ARTIFACTS / "TotalKillsPrediction" / "TotalKillsPrediction.pkl"
)
TOTAL_KILLS_PREDICTION_CATEGORICAL_FEATURES: Final = (
    MODEL_ARTIFACTS
    / "TotalKillsPrediction"
    / "TotalKillsPrediction_categorical_features.pkl"
)
TOTAL_KILLS_PREDICTION_FINAL_FEATURES: Final = (
    MODEL_ARTIFACTS / "TotalKillsPrediction" / "TotalKillsPrediction_final_features.pkl"
)
TOTAL_KILLS_PREDICTION_BEST_HYPERPARAMETERS: Final = (
    MODEL_ARTIFACTS
    / "TotalKillsPrediction"
    / "TotalKillsPrediction_best_hyperparameters.pkl"
)
TOTAL_TOWERS_PREDICTION_MODEL_PATH: Final = (
    MODEL_ARTIFACTS / "TotalTowersPrediction" / "TotalTowersPrediction.pkl"
)
TOTAL_TOWERS_PREDICTION_CATEGORICAL_FEATURES: Final = (
    MODEL_ARTIFACTS
    / "TotalTowersPrediction"
    / "TotalTowersPrediction_categorical_features.pkl"
)
TOTAL_TOWERS_PREDICTION_FINAL_FEATURES: Final = (
    MODEL_ARTIFACTS / "TotalTowersPrediction" / "TotalTowersPrediction_final_features.pkl"
)
TOTAL_TOWERS_PREDICTION_BEST_HYPERPARAMETERS: Final = (
    MODEL_ARTIFACTS
    / "TotalTowersPrediction"
    / "TotalTowersPrediction_best_hyperparameters.pkl"
)

# Hyper-parameter grids
HYPERPARAMETERS: Final = MODELS_DIR / "hyperparameters"
DEFAULT_MODELS_PARAMETERS: Final = HYPERPARAMETERS / "default_models_parameters.json"
BEST_HYPERPARAMETERS: Final = HYPERPARAMETERS / "best_hyperparams"
LEAGUES_ELO_HYPERPARAMETERS: Final = (
    BEST_HYPERPARAMETERS / "leagues_elo_hyperparameters.json"
)
ENTITY_ELO_HYPERPARAMETERS: Final = (
    BEST_HYPERPARAMETERS / "entity_elo_hyperparameters.json"
)
ENTITY_GLICKO_HYPERPARAMETERS: Final = (
    BEST_HYPERPARAMETERS / "entity_glicko_hyperparameters.json"
)
ENTITY_PL_HYPERPARAMETERS: Final = (
    BEST_HYPERPARAMETERS / "entity_pl_hyperparameters.json"
)
ENTITY_TRUESKILL_HYPERPARAMETERS: Final = (
    BEST_HYPERPARAMETERS / "entity_trueskill_hyperparameters.json"
)

# --------------------------------------------------------------------------- #
# Reports & figures
# --------------------------------------------------------------------------- #
FIGURES_DIR: Final = REPORTS_DIR / "figures"
INSIGHTS_DIR: Final = REPORTS_DIR / "evaluation_insights"

# --------------------------------------------------------------------------- #
# Directory bootstrap
# --------------------------------------------------------------------------- #
_directories: list[Path] = [
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROCESSED_TEAMS_DIR,
    PROCESSED_PLAYERS_DIR,
    MODELS_DIR,
    MODEL_ARTIFACTS,
    REPORTS_DIR,
    FIGURES_DIR,
    FIGURES_DIR,
    INSIGHTS_DIR,
    CONFIG_DIR,
    DATA_INGESTION_DIR,
    TRAINING_AND_INPUT_COLS_DIR,
    NOTEBOOKS_DIR,
    LOGS_DIR,
    HYPERPARAMETERS,
    BEST_HYPERPARAMETERS,
]

for _d in _directories:
    _create_directory(_d)
