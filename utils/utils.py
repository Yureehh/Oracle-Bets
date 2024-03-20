import json
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
    with open(file_path, "r") as file:
        return json.load(file)


def get_identity(entity):
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")
    return "playerid" if entity.lower() == "player" else "teamid"
