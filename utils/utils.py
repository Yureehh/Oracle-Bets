"""
Utility functions

This script defines utility functions for the project.
"""

import json
import pickle
from typing import List


def get_sorting_keys(entity: str) -> List[str]:
    """
    Get sorting keys for the specified entity.

    Parameters
    ----------
    entity : str
        Entity for which sorting keys are to be computed.

    Returns
    -------
    List[str]
        List of sorting keys.
    """
    if entity.lower() == "team":
        return ["date", "league", "gameid", "side"]
    return ["date", "league", "gameid", "teamid", "side", "position"]


def json_loader(file_path):
    """Load the JSON file from the specified file path."""
    with open(file_path) as file:
        return json.load(file)


def get_identity(entity):
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")
    return "playerid" if entity.lower() == "player" else "teamid"


def setup_pandas(pd):
    """
    Set up pandas display options for better readability.
    """
    pd.options.display.float_format = "{:,.4f}".format
    pd.set_option("display.max_rows", None, "display.max_columns", None)


def load_model(filepath: str):
    """
    Load a machine learning model from a file.
    """
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model
