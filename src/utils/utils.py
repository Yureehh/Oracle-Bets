"""
Utility functions.

This script defines utility functions for the project, including file loaders, model handlers,
and data storage utilities. These functions facilitate tasks such as loading various file types,
managing entity-specific operations, and handling model serialization/deserialization.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def get_sorting_keys(entity: str) -> list[str]:
    """
    Get sorting keys for the specified entity.

    Args:
        entity (str): The entity type, either 'team' or 'player'.

    Returns:
        List[str]: A list of sorting keys.

    """
    SORTING_KEYS: dict[str, list[str]] = {
        "team": ["date", "league", "gameid", "side"],
        "player": ["date", "league", "gameid", "side", "position"],
    }
    entity_lower = entity.lower()
    if entity_lower in SORTING_KEYS:
        return SORTING_KEYS[entity_lower]
    msg = f"Entity must be either 'player' or 'team', not '{entity}'."
    raise ValueError(msg)


def load_file(file_path: str | Path, file_type: str = "json") -> Any | pd.DataFrame:
    """
    File loading function to handle JSON, CSV, and Parquet files.

    Args:
        file_path (Union[str, Path]): The path to the file.
        file_type (str): The type of the file ('json', 'csv', 'parquet').

    Returns:
        Union[Any, pd.DataFrame]: The loaded file data.

    """
    file_path = Path(file_path)
    try:
        if file_type == "json":
            with file_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        elif file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "parquet":
            return pd.read_parquet(file_path)
        else:
            msg = (
                f"Unsupported file type: '{file_type}'. "
                "Supported types are 'json', 'csv', 'parquet'."
            )
            raise ValueError(msg)
    except FileNotFoundError as e:
        msg = f"No such file: '{file_path}'"
        raise FileNotFoundError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Error parsing JSON file '{file_path}': {e}"
        raise ValueError(msg) from e
    except ValueError as e:
        msg = f"Error parsing Parquet file '{file_path}': {e}"
        raise ValueError(msg) from e
    except Exception as e:
        msg = f"Unexpected error loading file '{file_path}': {e}"
        raise ValueError(msg) from e


def json_loader(file_path: str | Path) -> Any:
    """
    Load a JSON file from the specified file path.

    Args:
        file_path (Union[str, Path]): The path to the JSON file.

    Returns:
        Any: The loaded JSON data.

    """
    return load_file(file_path, file_type="json")


def csv_loader(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file from the specified file path.

    Args:
        file_path (Union[str, Path]): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded CSV data.

    """
    return load_file(file_path, file_type="csv")


def parquet_loader(file_path: str | Path) -> pd.DataFrame:
    """
    Load a Parquet file from the specified file path.

    Args:
        file_path (Union[str, Path]): The path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded Parquet data.

    """
    return load_file(file_path, file_type="parquet")


def get_identity(entity: str) -> str:
    """
    Get the identity column name based on the entity type.

    Args:
        entity (str): The entity type, either 'player' or 'team'.

    Returns:
        str: The identity column name (e.g., 'playerid' or 'teamid').

    """
    entity_lower = entity.lower()
    if entity_lower in {"player", "team"}:
        return f"{entity_lower}id"
    msg = "Entity must be either 'player' or 'team'."
    raise ValueError(msg)


def load_model(filepath: str | Path) -> Any:
    """
    Load a machine learning model from a file using pickle.

    Args:
        filepath (Union[str, Path]): The path to the model file.

    Returns:
        Any: The loaded model.

    """
    filepath = Path(filepath)
    try:
        with filepath.open("rb") as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        msg = f"Model file not found: '{filepath}'"
        raise FileNotFoundError(msg) from e
    except pickle.UnpicklingError as e:
        msg = f"Error loading model from '{filepath}': {e}"
        raise pickle.UnpicklingError(msg) from e
    except Exception as e:
        msg = f"Unexpected error loading model from '{filepath}': {e}"
        raise ValueError(msg) from e


def store_model(
    path: str | Path, model: Any, model_name: str, logger: logging.Logger
) -> None:
    """
    Store the trained model to a file using pickle.

    Args:
        path (Union[str, Path]): The file path to store the model.
        model (Any): The trained model to be stored.
        model_name (str): The name of the model (for logging purposes).
        logger (logging.Logger): The logger for logging messages.

    """
    path = Path(path)
    try:
        with path.open("wb") as f:
            pickle.dump(model, f)
        logger.info(f"Stored '{model_name}' model to '{path}'.")
    except pickle.PicklingError as e:
        logger.exception(f"Failed to store the model '{model_name}' at '{path}': {e}")
        msg = f"Error storing model '{model_name}' at '{path}': {e}"
        raise pickle.PicklingError(msg) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error storing the model '{model_name}' at '{path}': {e}"
        )
        msg = f"Unexpected error storing model '{model_name}' at '{path}': {e}"
        raise ValueError(msg) from e


def load_training_data(
    training_team_data: str | Path,
    training_player_data: str | Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data for the model from Parquet files.

    Args:
        training_team_data (Union[str, Path]): The path to the team's training data Parquet file.
        training_player_data (Union[str, Path]): The path to the player's training data Parquet file.
        logger (logging.Logger): The logger for logging messages.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the team and player training DataFrames.

    """
    try:
        team_df = parquet_loader(training_team_data)
        player_df = parquet_loader(training_player_data)
        logger.info("Training data loaded successfully.")
        return team_df, player_df
    except Exception as e:
        logger.exception(f"Failed to load training data: {e}")
        msg = f"Error loading training data: {e}"
        raise ValueError(msg) from e


def safe_store_df_as_parquet(
    df: pd.DataFrame, output_path: str | Path, loggers: list[logging.Logger]
) -> None:
    """
    Store a Pandas DataFrame to Parquet format.

    Args:
        df (pd.DataFrame): The DataFrame to be stored.
        output_path (Union[str, Path]): The file path to store the Parquet file.
        logger (logging.Logger): The logger for logging messages.

    """
    output_path = Path(output_path)
    try:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_parquet(output_path, compression="gzip")
        for logger in loggers:
            logger.info(f"Successfully saved DataFrame to Parquet at '{output_path}'.")
    except Exception:
        for logger in loggers:
            logger.exception(
                "Failed to save DataFrame to Parquet. Falling back to Polar"
            )
        try:
            import polars as pl

            pl_df = pl.DataFrame(df)
            pl_df.write_parquet(output_path, compression="gzip")
            logger.info(f"Successfully saved DataFrame to Parquet at '{output_path}'.")
        except Exception as e:
            logger.exception(
                f"Failed to store DataFrame as Parquet at '{output_path}': {e}"
            )
            msg = f"Failed to store DataFrame as Parquet at '{output_path}': {e}"
            raise ValueError(msg) from e
