"""
Models Training

Initializes and trains the project's models using team & player training tables.
Outputs serialized models to MODELS_DIR.

Models covered:
- Outcome prediction (classification)
# - Gamelength prediction (regression)
# - Total kills prediction (regression)
# - Total towers prediction (regression)

Usage:
    python src/2_models_training.py
    # Configure MODEL_TYPE and FEATURE_SELECTION variables below before running
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prediction_models.lightgbm_model import ModelFactory
from utils.io_utils import load_training_data, store_model
from utils.logger import logger
from utils.paths import (
    MODEL_ARTIFACTS,
    TRAINING_PLAYER_DATA,
    TRAINING_TEAM_DATA,
)

if TYPE_CHECKING:
    from pathlib import Path

    from utils.pd import pd

# ───────────────────────────────  types / config  ─────────────────────────────

ProblemType = Literal["classification", "regression"]
ModelType = Literal["lightgbm", "tabnet"]
FeatureSelectionMethod = Literal["none", "importance", "cumulative", "rfecv", "boruta"]
MODEL_FILE_EXTENSION = "pkl"

# ─────────────────────────  TRAINING CONFIGURATION  ───────────────────────────
# Modify these variables before running the script

MODEL_TYPE: ModelType = "tabnet"  # "lightgbm" or "tabnet"
FEATURE_SELECTION: FeatureSelectionMethod = (
    "none"  # "none", "importance", "cumulative", "rfecv", "boruta"
)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    target_column: str
    problem_type: ProblemType
    validate: bool = True


# Enable/disable models here
MODELS_TO_TRAIN: tuple[ModelConfig, ...] = (
    ModelConfig(
        model_name="OutcomePrediction",
        target_column="result",
        problem_type="classification",
    ),
    # ModelConfig("GamelengthPrediction", "gamelength", "regression"),  # noqa: ERA001
    # ModelConfig("TotalKillsPrediction", "total_kills", "regression"),  # noqa: ERA001
    # ModelConfig("TotalTowersPrediction", "total_towers", "regression"),  # noqa: ERA001
)


# ───────────────────────────────  helpers  ───────────────────────────────────


def _model_path(name: str, ext: str = MODEL_FILE_EXTENSION) -> Path:
    # Always store under models/<Name>/<Name>.<ext>
    return MODEL_ARTIFACTS / name / f"{name}.{ext}"


def _check_target_presence(team_df: pd.DataFrame, target: str) -> None:
    """
    Minimal guard: team targets (result / gamelength / totals) must exist
    in the team training table before we hand off to the model pipeline.
    """
    if target not in team_df.columns:
        msg = (
            f"Target column '{target}' not found in team training data "
            f"(available: {len(team_df.columns)} columns)."
        )
        raise ValueError(msg)


# ───────────────────────────────  core training  ─────────────────────────────


def _get_model_name_with_suffix(base_name: str, model_type: ModelType) -> str:
    """Generate model name with type suffix (e.g., OutcomePrediction_TabNet)."""
    suffix_map: dict[ModelType, str] = {
        "lightgbm": "LightGBM",
        "tabnet": "TabNet",
    }
    return f"{base_name}_{suffix_map[model_type]}"


def initialize_and_train_model(
    cfg: ModelConfig,
    training_team_data: pd.DataFrame,
    training_player_data: pd.DataFrame,
    model_type: ModelType = "lightgbm",
    feature_selection: FeatureSelectionMethod = "none",
) -> Path | None:
    """
    Initialize, train (and optionally validate) a model defined by `cfg`.
    Returns the stored model path on success.
    """
    _check_target_presence(training_team_data, cfg.target_column)

    # Add model type suffix to prevent overwriting different model types
    full_model_name = _get_model_name_with_suffix(cfg.model_name, model_type)

    logger.info(
        f"Initializing '{full_model_name}' "
        f"({cfg.problem_type}, model={model_type}) with target='{cfg.target_column}'…"
    )

    model = ModelFactory.create_model(
        model_name=full_model_name,
        problem_type=cfg.problem_type,
        training_team_data=training_team_data,
        training_player_data=training_player_data,
        model_type=model_type,
    )

    start = dt.datetime.now()
    # Allow model to do its own splits/joins/feature selection internally
    model.preprocess_data(target_col=cfg.target_column)

    logger.info(
        f"Training '{full_model_name}' (validate={cfg.validate}, "
        f"feature_selection={feature_selection})…"
    )
    trained_model = model.train_and_validate_model(
        target_col=cfg.target_column,
        validate=cfg.validate,
        feature_selection=feature_selection,
    )
    elapsed = (dt.datetime.now() - start).total_seconds()

    logger.info(f"'{full_model_name}' training complete in {elapsed:.2f}s.")

    path = _model_path(full_model_name)
    store_model(path, trained_model, full_model_name, logger)
    logger.info(f"Stored trained model: {path}\n")

    return path


def train_models(
    model_type: ModelType = "lightgbm",
    feature_selection: FeatureSelectionMethod = "none",
) -> None:
    """Train all configured models using shared training tables."""
    try:
        logger.info("Loading training data…")
        team_df, player_df = load_training_data(
            TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA, logger
        )
    except Exception as e:
        logger.exception(f"Failed to load training data: {e}")
        raise

    trained: list[str] = []
    failed: list[str] = []

    for cfg in MODELS_TO_TRAIN:
        try:
            initialize_and_train_model(
                cfg=cfg,
                training_team_data=team_df,
                training_player_data=player_df,
                model_type=model_type,
                feature_selection=feature_selection,
            )
            trained.append(cfg.model_name)
        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user during '{cfg.model_name}'")
            raise
        except Exception as e:
            failed.append(cfg.model_name)
            logger.exception(f"Training failed for '{cfg.model_name}': {e}")

    # Summary
    if trained:
        logger.info(f"Successfully trained: {', '.join(trained)}")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
    logger.info("All model training tasks finished.\n")


# ───────────────────────────────  entrypoint  ─────────────────────────────────

if __name__ == "__main__":
    try:
        logger.info(
            f"Starting training with model_type={MODEL_TYPE}, "
            f"feature_selection={FEATURE_SELECTION}"
        )
        train_models(
            model_type=MODEL_TYPE,
            feature_selection=FEATURE_SELECTION,
        )
    except (KeyboardInterrupt, Exception) as e:
        logger.exception(f"Unexpected error during model training: {e}")
