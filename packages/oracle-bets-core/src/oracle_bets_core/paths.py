"""
Paths module.

Centralises every on-disk location used by Oracle-Bets and eagerly ensures the
directories exist.  Nothing outside this module should hard-code a filesystem
path string.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Model Type Configuration
# --------------------------------------------------------------------------- #
ModelType = Literal["LightGBM", "TabNet"]

# Change this to switch between model types for inference
ACTIVE_MODEL_TYPE: ModelType = "LightGBM"

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
load_dotenv()


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
def _discover_suite_root() -> Path:
    """Find the repository/suite root from this package location."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").exists():
            return parent
    return Path.cwd()


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else None


def _scoped_dir(kind: str) -> Path:
    """Resolve a LoL-owned top-level directory, with optional env override."""
    override = _env_path(f"ORACLE_BETS_{kind.upper()}_DIR")
    if override is not None:
        return override

    if LOL_HOME is not None:
        return LOL_HOME / kind
    return SUITE_ROOT / kind / "lol"


SUITE_ROOT: Final[Path] = (
    _env_path("ORACLE_BETS_HOME") or _discover_suite_root()
).resolve()

LOL_HOME: Final = _env_path("ORACLE_BETS_LOL_HOME")

DATA_DIR: Final = _scoped_dir("data")
CONFIG_DIR: Final = _scoped_dir("config")
MODELS_DIR: Final = _scoped_dir("models")
NOTEBOOKS_DIR: Final = _scoped_dir("notebooks")
REPORTS_DIR: Final = _scoped_dir("reports")
LOGS_DIR: Final = _scoped_dir("logs")

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
EXTRAS_DIR: Final = REPORTS_DIR / "ingestion"
TARGET_FEATURES: Final = CONFIG_DIR / "target_features.json"
TRAINING_AND_INPUT_COLS_DIR: Final = CONFIG_DIR / "training"

YEARS_RANGE_PATH: Final = DATA_INGESTION_DIR / "years_range.json"
IMPORT_COLUMNS: Final = DATA_INGESTION_DIR / "import_columns.json"
TEAM_REPLACEMENTS_AND_INVALID_GAMES: Final = (
    DATA_INGESTION_DIR / "team_name_replacements_and_invalid_games.json"
)
CONSIDERED_LEAGUES: Final = DATA_INGESTION_DIR / "considered_leagues.json"
LEAGUE_TAXONOMY: Final = DATA_INGESTION_DIR / "league_taxonomy.json"

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
TRAINING_COMPACT_TEAM_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_compact_team_config.json"
)
FLATTENED_TEAM_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "flattened_team_config.json"
)
TRAINING_PLAYER_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_player_config.json"
)
TRAINING_COMPACT_PLAYER_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "training_compact_player_config.json"
)
FLATTENED_PLAYER_CONFIG: Final = (
    TRAINING_AND_INPUT_COLS_DIR / "flattened_player_config.json"
)

# --------------------------------------------------------------------------- #
# Model outputs (scoped under models/lol by default)
# --------------------------------------------------------------------------- #
LEAGUE_ELO: Final = MODELS_DIR / "league_elo.parquet"
TEAM_LEAGUES_MAPPING: Final = MODELS_DIR / "team_league_mapping.parquet"
WHOLE_HISTORY_RATING_PATH: Final = MODELS_DIR / "whr.pkl"


# --------------------------------------------------------------------------- #
# Dynamic model path helpers (uses ACTIVE_MODEL_TYPE)
# --------------------------------------------------------------------------- #
def _model_dir(base_name: str) -> Path:
    """Get model directory with type suffix (e.g., OutcomePrediction_TabNet/)."""
    return MODELS_DIR / f"{base_name}_{ACTIVE_MODEL_TYPE}"


def _model_path(base_name: str) -> Path:
    """Get path to the main model file."""
    full_name = f"{base_name}_{ACTIVE_MODEL_TYPE}"
    return _model_dir(base_name) / f"{full_name}.pkl"


def _model_artifact(base_name: str, artifact: str) -> Path:
    """Get path to a model artifact (e.g., categorical_features, feature_pipeline)."""
    full_name = f"{base_name}_{ACTIVE_MODEL_TYPE}"
    return _model_dir(base_name) / f"{full_name}_{artifact}.pkl"


# --------------------------------------------------------------------------- #
# Outcome Prediction Model Paths
# --------------------------------------------------------------------------- #
OUTCOME_PREDICTION_MODEL_PATH: Path = _model_path("OutcomePrediction")
OUTCOME_PREDICTION_CATEGORICAL_FEATURES: Path = _model_artifact(
    "OutcomePrediction", "categorical_features"
)
OUTCOME_PREDICTION_FINAL_FEATURES: Path = _model_artifact(
    "OutcomePrediction", "final_features"
)
OUTCOME_PREDICTION_BEST_HYPERPARAMETERS: Path = _model_artifact(
    "OutcomePrediction", "best_hyperparameters"
)
OUTCOME_PREDICTION_FEATURE_PIPELINE: Path = _model_artifact(
    "OutcomePrediction", "feature_pipeline"
)
OUTCOME_PREDICTION_CATEGORICAL_ENCODINGS: Path = _model_artifact(
    "OutcomePrediction", "categorical_encodings"
)

# --------------------------------------------------------------------------- #
# Gamelength Prediction Model Paths
# --------------------------------------------------------------------------- #
GAMELENGTH_PREDICTION_MODEL_PATH: Path = _model_path("GamelengthPrediction")
GAMELENGTH_PREDICTION_CATEGORICAL_FEATURES: Path = _model_artifact(
    "GamelengthPrediction", "categorical_features"
)
GAMELENGTH_PREDICTION_FINAL_FEATURES: Path = _model_artifact(
    "GamelengthPrediction", "final_features"
)
GAMELENGTH_PREDICTION_BEST_HYPERPARAMETERS: Path = _model_artifact(
    "GamelengthPrediction", "best_hyperparameters"
)
GAMELENGTH_PREDICTION_FEATURE_PIPELINE: Path = _model_artifact(
    "GamelengthPrediction", "feature_pipeline"
)
GAMELENGTH_PREDICTION_CATEGORICAL_ENCODINGS: Path = _model_artifact(
    "GamelengthPrediction", "categorical_encodings"
)

# --------------------------------------------------------------------------- #
# Total Kills Prediction Model Paths
# --------------------------------------------------------------------------- #
TOTAL_KILLS_PREDICTION_MODEL_PATH: Path = _model_path("TotalKillsPrediction")
TOTAL_KILLS_PREDICTION_CATEGORICAL_FEATURES: Path = _model_artifact(
    "TotalKillsPrediction", "categorical_features"
)
TOTAL_KILLS_PREDICTION_FINAL_FEATURES: Path = _model_artifact(
    "TotalKillsPrediction", "final_features"
)
TOTAL_KILLS_PREDICTION_BEST_HYPERPARAMETERS: Path = _model_artifact(
    "TotalKillsPrediction", "best_hyperparameters"
)
TOTAL_KILLS_PREDICTION_FEATURE_PIPELINE: Path = _model_artifact(
    "TotalKillsPrediction", "feature_pipeline"
)
TOTAL_KILLS_PREDICTION_CATEGORICAL_ENCODINGS: Path = _model_artifact(
    "TotalKillsPrediction", "categorical_encodings"
)

# --------------------------------------------------------------------------- #
# Total Towers Prediction Model Paths
# --------------------------------------------------------------------------- #
TOTAL_TOWERS_PREDICTION_MODEL_PATH: Path = _model_path("TotalTowersPrediction")
TOTAL_TOWERS_PREDICTION_CATEGORICAL_FEATURES: Path = _model_artifact(
    "TotalTowersPrediction", "categorical_features"
)
TOTAL_TOWERS_PREDICTION_FINAL_FEATURES: Path = _model_artifact(
    "TotalTowersPrediction", "final_features"
)
TOTAL_TOWERS_PREDICTION_BEST_HYPERPARAMETERS: Path = _model_artifact(
    "TotalTowersPrediction", "best_hyperparameters"
)
TOTAL_TOWERS_PREDICTION_FEATURE_PIPELINE: Path = _model_artifact(
    "TotalTowersPrediction", "feature_pipeline"
)
TOTAL_TOWERS_PREDICTION_CATEGORICAL_ENCODINGS: Path = _model_artifact(
    "TotalTowersPrediction", "categorical_encodings"
)

# --------------------------------------------------------------------------- #
# Hyper-parameter configs (scoped under config/lol by default)
# --------------------------------------------------------------------------- #
HYPERPARAMETERS: Final = CONFIG_DIR / "hyperparameters"
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
    REPORTS_DIR,
    FIGURES_DIR,
    INSIGHTS_DIR,
    CONFIG_DIR,
    DATA_INGESTION_DIR,
    EXTRAS_DIR,
    TRAINING_AND_INPUT_COLS_DIR,
    HYPERPARAMETERS,
    BEST_HYPERPARAMETERS,
    NOTEBOOKS_DIR,
    LOGS_DIR,
]

for _d in _directories:
    _create_directory(_d)
