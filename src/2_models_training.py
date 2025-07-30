"""
Models Training

This script initializes and trains needed models for the project.
It loads training data, initializes and trains the models, and stores the trained models.
The models are outcome prediction, game length prediction, total kills prediction, and total towers prediction.
"""

import datetime as dt

import pandas as pd

from prediction_models.lightgbm_model import ModelFactory
from utils.logger import logger
from utils.paths import MODELS_DIR, TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA
from utils.utils import load_training_data, store_model

# Constants
MODEL_FILE_EXTENSION = "pkl"

# List of models to train with their configurations
MODELS_TO_TRAIN = [
    {
        "model_name": "OutcomePrediction",
        "target_column": "result",
        "problem_type": "classification",
    },
    # {
    #     "model_name": "GamelengthPrediction",
    #     "target_column": "gamelength",
    #     "problem_type": "regression",
    # },
    # {
    #     "model_name": "TotalKillsPrediction",
    #     "target_column": "total_kills",
    #     "problem_type": "regression",
    # },
    # {
    #     "model_name": "TotalTowersPrediction",
    #     "target_column": "total_towers",
    #     "problem_type": "regression",
    # },
]


def initialize_and_train_model(
    model_name: str,
    training_team_data: pd.DataFrame,
    training_player_data: pd.DataFrame,
    target_column_name: str,
    problem_type: str,
    validate: bool = True,
) -> None:
    """
    Initialize and train the model.

    Args:
        model_name (str): Name of the model to be trained.
        training_team_data (pd.DataFrame): DataFrame containing team-level features.
        training_player_data (pd.DataFrame): DataFrame containing player-level features.
        target_column_name (str): The name of the target column in the data.
        problem_type (str): Type of the problem ('classification' or 'regression').
        validate (bool): Whether to perform model validation. Defaults to True.

    Raises:
        Exception: If training fails.

    """
    try:
        model = ModelFactory.create_model(
            model_name=model_name,
            problem_type=problem_type,
            training_team_data=training_team_data,
            training_player_data=training_player_data,
        )

        logger.info(
            f"Initialized {model_name} model for {problem_type} problem. Starting training process..."
        )
        start_time = dt.datetime.now()

        model.preprocess_data(target_col=target_column_name)
        trained_model = model.train_and_validate_model(
            target_col=target_column_name, validate=validate
        )
        elapsed_time = (dt.datetime.now() - start_time).total_seconds()

        logger.info(
            f"{model_name} model training and evaluation completed in {elapsed_time:.2f} seconds"
        )

        model_path = MODELS_DIR / f"{model_name}.{MODEL_FILE_EXTENSION}"
        store_model(model_path, trained_model, model_name, logger)
        logger.info(f"Stored trained model at {model_path}\n")

    except Exception as e:
        logger.error(f"Failed to train {model_name} model: {e}")
        raise


def train_models() -> None:
    """Train all models defined in MODELS_TO_TRAIN."""
    try:
        logger.info("Loading training data...")
        training_team_data, training_player_data = load_training_data(
            TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA, logger
        )
        logger.info("Training data loaded successfully.\n")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

    for model_info in MODELS_TO_TRAIN:
        model_name = model_info["model_name"]
        target_column = model_info["target_column"]
        problem_type = model_info["problem_type"]

        try:
            initialize_and_train_model(
                model_name=model_name,
                training_team_data=training_team_data,
                training_player_data=training_player_data,
                target_column_name=target_column,
                problem_type=problem_type,
            )
        except Exception as e:
            logger.error(f"An error occurred while training {model_name}: {e}")
            continue  # Proceed to the next model

    logger.info("All models training process completed.\n")


if __name__ == "__main__":
    try:
        train_models()
    except KeyboardInterrupt:
        logger.error("Training models interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
