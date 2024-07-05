"""
Models Training

This script initializes and trains needed models for the project.
It loads training data, initializes and trains the models, and stores the trained model.
The models are outcome prediction, gamelength prediction, total kills prediction, and total towers prediction.
"""

import datetime as dt

import pandas as pd

from prediction_models.lightgbm_model import LightGBMModel
from utils.logger import logger, models_logger
from utils.paths import MODELS_DIR, TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA
from utils.utils import load_training_data, store_model

# Constants
OUTCOME_MODEL_NAME = "OutcomePrediction"
MODEL_FILE_EXTENSION = "pkl"


def initialize_and_train_model(
    model_name: str,
    training_team_data: pd.DataFrame,
    training_player_data: pd.DataFrame,
    target_column_name: str = "result",
) -> LightGBMModel:
    """
    Initialize and train the LightGBM model.
    """
    lightgbm_model = LightGBMModel(model_name, training_team_data, training_player_data)

    logger.info(f"Initialized {model_name} model. Starting training process...\n")
    start = dt.datetime.now()

    lightgbm_model.preprocess_data(target_column_name)
    trained_model = lightgbm_model.train_and_validate_model(target_column_name)

    end = dt.datetime.now()
    elapsed_time = end - start

    logger.info(f"{model_name} model training and evaluation completed in {elapsed_time}")
    models_logger.info(f"{model_name} model training and evaluation completed in {elapsed_time}")

    return trained_model


def main():
    """
    Main function to run the model training process.
    """
    try:
        logger.info("Starting the training process...\n")
        models_logger.info("Starting the training process...\n")
        training_team_data, training_player_data = load_training_data(TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA, logger)
        outcome_prediction_model = initialize_and_train_model(
            OUTCOME_MODEL_NAME, training_team_data, training_player_data, "result"
        )
        model_path = MODELS_DIR / f"{OUTCOME_MODEL_NAME}.{MODEL_FILE_EXTENSION}"
        store_model(model_path, outcome_prediction_model, OUTCOME_MODEL_NAME, logger, models_logger)

        logger.info("Training process completed successfully.\n")
        models_logger.info("Training process completed successfully.\n\n\n")
    except Exception as e:
        error_message = f"An error occurred during the training process: {e}\n\n\n"
        logger.error(error_message)
        models_logger.error(error_message)


if __name__ == "__main__":
    main()
