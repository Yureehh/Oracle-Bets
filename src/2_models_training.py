"""
Models Training

Initializes and trains the project's models using team & player training tables.
Outputs serialized models to MODELS_DIR.

Models covered:
- Outcome prediction (classification)
# - Gamelength prediction (regression)
# - Total kills prediction (regression)
# - Total towers prediction (regression)
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

    import pandas as pd

# ───────────────────────────────  types / config  ─────────────────────────────

ProblemType = Literal["classification", "regression"]
MODEL_FILE_EXTENSION = "pkl"


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
    # Always store under models/artifacts/<Name>/<Name>.<ext>
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


def initialize_and_train_model(
    cfg: ModelConfig,
    training_team_data: pd.DataFrame,
    training_player_data: pd.DataFrame,
) -> Path | None:
    """
    Initialize, train (and optionally validate) a model defined by `cfg`.
    Returns the stored model path on success.
    """
    _check_target_presence(training_team_data, cfg.target_column)

    logger.info(
        f"Initializing '{cfg.model_name}' "
        f"({cfg.problem_type}) with target='{cfg.target_column}'…"
    )

    model = ModelFactory.create_model(
        model_name=cfg.model_name,
        problem_type=cfg.problem_type,
        training_team_data=training_team_data,
        training_player_data=training_player_data,
    )

    start = dt.datetime.now()
    # Allow model to do its own splits/joins/feature selection internally
    model.preprocess_data(target_col=cfg.target_column)

    logger.info(f"Training '{cfg.model_name}' (validate={cfg.validate})…")
    trained_model = model.train_and_validate_model(
        target_col=cfg.target_column, validate=cfg.validate
    )
    elapsed = (dt.datetime.now() - start).total_seconds()

    logger.info(f"'{cfg.model_name}' training complete in {elapsed:.2f}s.")

    path = _model_path(cfg.model_name)
    store_model(path, trained_model, cfg.model_name, logger)
    logger.info(f"Stored trained model: {path}\n")

    return path


def train_models() -> None:
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
            )
            trained.append(cfg.model_name)
        except (Exception, KeyboardInterrupt) as e:
            failed.append(cfg.model_name)
            logger.exception(f"Training failed for '{cfg.model_name}': {e}")

    # Summary
    if trained:
        logger.info(f"Successfully trained: {', '.join(trained)}")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
    logger.info("All model training tasks finished.\n")


# ───────────────────────────────  cli entrypoint  ─────────────────────────────

if __name__ == "__main__":
    try:
        train_models()
    except (KeyboardInterrupt, Exception) as e:
        logger.exception(f"Unexpected error during model training: {e}")
