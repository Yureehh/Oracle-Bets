"""
Utility functions

This script defines utility functions for the project.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Union

import pandas as pd


def get_sorting_keys(entity: str) -> List[str]:
    """
    Get sorting keys for the specified entity.

    Parameters:
        entity (str): The entity type, either 'team' or 'player'.

    Returns:
        List[str]: A list of sorting keys.
    """
    keys = {
        "team": ["date", "league", "gameid", "side"],
        "player": ["date", "league", "gameid", "side", "teamid", "position"],
    }
    entity_lower = entity.lower()
    if entity_lower in keys:
        return keys[entity_lower]
    else:
        raise ValueError(f"Entity must be either 'player' or 'team', not '{entity}'.")


def json_loader(file_path: str) -> Any:
    """
    Load the JSON file from the specified file path.
    """
    return load_file(file_path, file_type="json")


def csv_loader(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file from the specified file path.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded CSV data.
    """
    return load_file(file_path, file_type="csv")


def parquet_loader(file_path: str) -> pd.DataFrame:
    """
    Load the parquet file from the specified file path.

    Parameters:
        file_path (str): The path to the parquet file.

    Returns:
        pd.DataFrame: The loaded parquet data.
    """
    return pd.read_parquet(file_path)


def load_file(file_path: str, file_type: str = "json") -> Union[Any, pd.DataFrame]:
    """
    Generic file loading function to handle JSON, CSV, and parquet files.

    Parameters:
        file_path (str): The path to the file.
        file_type (str): The type of the file ('json', 'csv', 'parquet').

    Returns:
        Union[Any, pd.DataFrame]: The loaded file data.
    """
    try:
        if file_type == "json":
            with open(file_path) as file:
                return json.load(file)
        elif file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: '{file_type}'")
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file: '{file_path}'") from None
    except (json.JSONDecodeError, pd.errors.ParserError) as e:
        error_msg = "JSON" if file_type == "json" else "CSV"
        raise type(e)(f"Error parsing {error_msg} file: '{file_path}'. {e}") from e


def get_identity(entity: str) -> str:
    """
    Get the identity column name based on the entity type.

    Parameters:
        entity (str): The entity type, either 'player' or 'team'.

    Returns:
        str: The identity column name.
    """
    entity_lower = entity.lower()
    if entity_lower in ["player", "team"]:
        return f"{entity_lower}id"
    raise ValueError("Entity must be either 'player' or 'team'.")


def load_model(filepath: str) -> Any:
    """
    Load a machine learning model from a file.

    Parameters:
        filepath (str): The path to the model file.

    Returns:
        Any: The loaded model.
    """
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: '{filepath}'") from None
    except pickle.PickleError as e:
        raise pickle.PickleError(f"Error loading model from '{filepath}': {e}") from e


def store_model(path: Path, model, model_name: str, logger: logging.Logger, models_logger: logging.Logger):
    """Store the trained model to a file."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Stored {model_name} model to {path}")
        models_logger.info(f"Stored {model_name} model to {path}\n")
    except Exception as e:
        logger.error(f"Failed to store the model: {e}")
        models_logger.error(f"Failed to store the model: {e}\n")
        raise


def load_training_data(
    training_team_data: str, training_player_data: str, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data for the model."""
    try:
        training_team_data = pd.read_parquet(training_team_data)
        training_player_data = pd.read_parquet(training_player_data)
        logger.info("Training data loaded successfully.")
        return training_team_data, training_player_data
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise
