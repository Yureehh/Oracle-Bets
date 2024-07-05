"""
Side win rate

This module provides functionality to compute the side win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from typing import Union

import pandas as pd

import src.ingestion.oracles_elixir as oe
from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_side(df: pd.DataFrame, side: str, identity: str, half_life: float) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for a specific side (Red or Blue).

    Parameters:
        df (pd.DataFrame): The input DataFrame containing match data.
        side (str): The side to compute EWM for ('Red' or 'Blue').
        identity (str): The identity column to group by (e.g., 'player' or 'team').
        half_life (float): The half-life for the EWM calculation.

    Returns:
        pd.DataFrame: DataFrame with computed EWM for the specified side.
    """
    side_df = df[df["side"] == side].copy()
    ema_col = f"ema_{side.lower()}_side"
    side_df[f"{ema_col}_before"] = side_df.groupby(identity)["result"].transform(
        lambda x: x.ewm(halflife=half_life, ignore_na=True).mean().shift().bfill()
    )
    side_df[f"{ema_col}_after"] = side_df.groupby(identity)["result"].transform(
        lambda x: x.ewm(halflife=half_life, ignore_na=True).mean()
    )
    return side_df


def calculate_side_win_likelihood(row: pd.Series, epsilon: float = EPSILON) -> Union[float, None]:
    """
    Calculate the EMA side win percentage.

    Parameters:
        row (pd.Series): A row from the DataFrame containing match data.
        epsilon (float): A small constant to prevent division by zero.

    Returns:
        Union[float, None]: Calculated EMA side win likelihood, or None if not computable.
    """
    side = row["side"].lower()  # Ensure the side value is lowercase for consistency.
    opposite_side = "red" if side == "blue" else "blue"  # Determine the opposite side.

    ema_side_col = f"ema_{side}_side_before"
    ema_opp_side_col = f"opp_ema_{opposite_side}_side_before"

    # Check if the required columns have non-null values.
    if pd.notnull(row[ema_side_col]) and pd.notnull(row[ema_opp_side_col]):
        # Calculate the EMA side win percentage with epsilon to avoid division by zero.
        return round(row[ema_side_col] / (row[ema_side_col] + row[ema_opp_side_col] + epsilon), 3)
    return None


def side_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing match data.
        entity (str): The entity type ('player' or 'team').

    Returns:
        pd.DataFrame: DataFrame with computed EWM for side win rates.

    Raises:
        ValueError: If the entity is not 'player' or 'team'.
    """
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")

    df.sort_values(get_sorting_keys(entity), inplace=True)
    identity = get_identity(entity)

    # Compute EMA for both Red and Blue sides
    red_side = compute_ema_side(df, "Red", identity, HALF_LIFE)
    blue_side = compute_ema_side(df, "Blue", identity, HALF_LIFE)

    # Merge and process
    merged = pd.concat([red_side, blue_side], ignore_index=True)
    merged.sort_values(get_sorting_keys(entity), inplace=True)
    merged.reset_index(drop=True, inplace=True)

    columns = [f"ema_{color}_side_before" for color in ["blue", "red"]] + [
        f"ema_{color}_side_after" for color in ["blue", "red"]
    ]

    # Apply forward and back fill within each group for the specified EWM columns
    for column in columns:
        # This row is needed to "carry forward" the metrics of opponent side when dealing with the "other" one.
        # So basically this carries forward red side metrics when dealing with blue side and vice versa.
        merged[column] = merged.groupby([identity])[column].transform(lambda x: x.ffill().bfill())

    merged.sort_values(get_sorting_keys(entity), inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Compute Opponent Columns
    for color in ["red", "blue"]:
        # Here I get the last value of the opponent side EMA before the processed row.
        # It will basically use last value of the opponent EMA  before the current row.
        merged[f"opp_ema_{color}_side_before"] = oe.get_opponent(merged[f"ema_{color}_side_before"].to_list(), entity)

    # Calculate win likelihood based on EMA
    merged["side_win_likelihood"] = merged.apply(lambda row: calculate_side_win_likelihood(row), axis=1)

    return merged.reset_index(drop=True)
