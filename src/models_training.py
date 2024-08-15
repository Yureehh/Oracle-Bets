"""
Models Training

This script initializes and trains needed models for the project.
It loads training data, initializes and trains the models, and stores the trained model.
The models are outcome prediction, gamelength prediction, total kills prediction, and total towers prediction.
"""

import datetime as dt

import pandas as pd

from prediction_models.lightgbm_model import ModelFactory
from src.utils.logger import logger, models_logger
from src.utils.paths import MODELS_DIR, TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA
from src.utils.utils import load_training_data, store_model

# Constants
OUTCOME_MODEL_NAME = "OutcomePrediction"
GAMELENGTH_MODEL_NAME = "GamelengthPrediction"
TOTAL_KILLS_MODEL_NAME = "TotalKillsPrediction"
TOTAL_TOWERS_MODEL_NAME = "TotalTowersPrediction"
MODEL_FILE_EXTENSION = "pkl"


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

    Parameters:
        model_name (str): Name of the model to be trained.
        training_team_data (pd.DataFrame): DataFrame containing team-level features.
        training_player_data (pd.DataFrame): DataFrame containing player-level features.
        target_column_name (str): The name of the target column in the data.
        problem_type (str): Type of the problem ('classification' or 'regression').
        validate (bool): Whether to perform model validation. Defaults to True.
    """
    model = ModelFactory.create_model(model_name, problem_type, training_team_data, training_player_data)

    logger.info(f"Initialized {model_name} model for {problem_type} problem. Starting training process...")
    start = dt.datetime.now()

    model.preprocess_data(target_column_name)
    trained_model = model.train_and_validate_model(target_column_name, validate)

    end = dt.datetime.now()
    elapsed_time = end - start

    logger.info(f"{model_name} model training and evaluation completed in {elapsed_time} seconds")
    models_logger.info(f"{model_name} model training and evaluation completed in {elapsed_time} seconds")

    model_path = MODELS_DIR / f"{model_name}.{MODEL_FILE_EXTENSION}"
    store_model(model_path, trained_model, model_name, logger, models_logger)


def train_models() -> None:
    """Train all models."""
    logger.info("Loading training data...")
    models_logger.info("Loading training data...")
    training_team_data, training_player_data = load_training_data(TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA)

    # Train the outcome prediction model (classification)
    initialize_and_train_model(
        model_name=OUTCOME_MODEL_NAME,
        training_team_data=training_team_data,
        training_player_data=training_player_data,
        target_column_name="result",
        problem_type="classification",
    )

    # Train regression models
    for model_name, target_column in [
        (GAMELENGTH_MODEL_NAME, "gamelength"),
        (TOTAL_KILLS_MODEL_NAME, "total_kills"),
        (TOTAL_TOWERS_MODEL_NAME, "total_towers"),
    ]:
        initialize_and_train_model(
            model_name=model_name,
            training_team_data=training_team_data,
            training_player_data=training_player_data,
            target_column_name=target_column,
            problem_type="regression",
        )

    logger.info("All models trained and validated.")
    models_logger.info("All models trained and validated.")


if __name__ == "__main__":
    try:
        train_models()
    except KeyboardInterrupt:
        logger.error("Training models interrupted by user.")
        models_logger.error("Training models interrupted by user.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        models_logger.error(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        models_logger.error(f"Value error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while training models: {e}")
        models_logger.error(f"An unexpected error occurred while training models: {e}")
