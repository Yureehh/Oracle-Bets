import datetime as dt
import pickle

import pandas as pd

from prediction_models.lightgbm_model import LightGBMModel
from utils.logger import logger, models_logger
from utils.paths import MODELS_DIR, TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA

# Constants
MODEL_NAME = "lightgbm"
MODEL_FILE_EXTENSION = "pkl"


def load_training_data():
    """Load training data for the model."""
    try:
        training_team_data = pd.read_parquet(TRAINING_TEAM_DATA)
        training_player_data = pd.read_parquet(TRAINING_PLAYER_DATA)
        logger.info("Training data loaded successfully.")
        return training_team_data, training_player_data
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


def initialize_and_train_model(model_name: str, training_team_data: pd.DataFrame, training_player_data: pd.DataFrame):
    """Initialize and train the LightGBM model."""
    try:
        lightgbm_model = LightGBMModel(model_name, training_team_data, training_player_data)

        logger.info(f"Initialized {model_name} model. Starting training process...\n")
        start = dt.datetime.now()

        lightgbm_model.preprocess_data()
        lightgbm_model.train_and_validate_model()

        end = dt.datetime.now()
        elapsed_time = end - start

        logger.info(f"{model_name} model training and evaluation completed in {elapsed_time}")
        models_logger.info(f"{model_name} model training and evaluation completed in {elapsed_time}\n")

        return lightgbm_model
    except Exception as e:
        logger.error(f"Failed to initialize and train the model: {e}")
        models_logger.error(f"Failed to initialize and train the model: {e}\n")
        raise


def store_model(model, model_name: str):
    """Store the trained model to a file."""
    model_path = MODELS_DIR / f"{model_name.lower()}.{MODEL_FILE_EXTENSION}"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Stored {model_name} model to {model_path}")
        models_logger.info(f"Stored {model_name} model to {model_path}\n")
    except Exception as e:
        logger.error(f"Failed to store the model: {e}")
        models_logger.error(f"Failed to store the model: {e}\n")
        raise


def main():
    """Main function to run the model training process."""
    try:
        training_team_data, training_player_data = load_training_data()
        lightgbm_model = initialize_and_train_model(MODEL_NAME, training_team_data, training_player_data)
        store_model(lightgbm_model, MODEL_NAME)

        logger.info("Training process completed successfully.\n")
        models_logger.info("Training process completed successfully.\n")
    except Exception as e:
        error_message = f"An error occurred during the training process: {e}"
        logger.error(error_message)
        models_logger.error(error_message + "\n")


if __name__ == "__main__":
    main()
