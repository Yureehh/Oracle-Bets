"""
Side Win Rate Module

This module provides functionality to compute the side win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from typing import Optional

import fireducks.pandas as pd

from ingestion.oracles_elixir import get_opponent
from src.utils.paths import DEFAULT_MODELS_PARAMETERS
from src.utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_side(df: pd.DataFrame, side: str, identity: str, half_life: float) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for a specific side ('Red' or 'Blue').

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        side (str): The side to compute EWM for ('Red' or 'Blue').
        identity (str): The identity column to group by (e.g., 'playerid' or 'teamid').
        half_life (float): The half-life for the EWM calculation.

    Returns:
        pd.DataFrame: DataFrame with computed EWM for the specified side.
    """
    side_df = df[df["side"] == side].copy()
    side_lower = side.lower()
    ema_col = f"ema_{side_lower}_side"

    # Compute EMA before and after
    group = side_df.groupby(identity)["result"]
    side_df[f"{ema_col}_before"] = (
        group.transform(lambda x: x.ewm(halflife=half_life, adjust=False, ignore_na=True).mean()).shift().bfill()
    )
    side_df[f"{ema_col}_after"] = group.transform(
        lambda x: x.ewm(halflife=half_life, adjust=False, ignore_na=True).mean()
    )
    return side_df


def calculate_side_win_likelihood(row: pd.Series) -> Optional[float]:
    """
    Calculate the EMA side win likelihood for a given row.

    Args:
        row (pd.Series): A row from the DataFrame containing match data.

    Returns:
        Optional[float]: Calculated EMA side win likelihood, or None if not computable.
    """
    side = row["side"].lower()
    opposite_side = "red" if side == "blue" else "blue"

    ema_side_col = f"ema_{side}_side_before"
    opp_ema_side_col = f"opp_ema_{opposite_side}_side_before"

    ema_side = row.get(ema_side_col)
    opp_ema_side = row.get(opp_ema_side_col)

    if pd.notnull(ema_side) and pd.notnull(opp_ema_side):
        total = ema_side + opp_ema_side + EPSILON
        likelihood = ema_side / total
        return round(likelihood, 3)
    return None


def side_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        entity (str): The entity type ('player' or 'team').

    Returns:
        pd.DataFrame: DataFrame with computed EWM for side win rates.

    Raises:
        ValueError: If the entity is not 'player' or 'team'.
    """
    if entity.lower() not in {"player", "team"}:
        raise ValueError("Entity must be either 'player' or 'team'.")

    identity = get_identity(entity)
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Compute EMA for both Red and Blue sides
    red_side_df = compute_ema_side(df, "Red", identity, HALF_LIFE)
    blue_side_df = compute_ema_side(df, "Blue", identity, HALF_LIFE)

    # Combine the dataframes
    combined_df = pd.concat([red_side_df, blue_side_df], ignore_index=True)
    combined_df = combined_df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Fill missing EMA values within each group
    ema_columns = [f"ema_{color}_side_before" for color in ["blue", "red"]] + [
        f"ema_{color}_side_after" for color in ["blue", "red"]
    ]
    combined_df[ema_columns] = (
        combined_df.groupby([identity])[ema_columns].apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
    )

    # Compute opponent EMA side values
    for color in ["red", "blue"]:
        ema_col = f"ema_{color}_side_before"
        opp_ema_col = f"opp_{ema_col}"
        combined_df[opp_ema_col] = get_opponent(combined_df[ema_col].to_list(), entity=entity)

    # Calculate side win likelihood
    # Calculate win likelihood based on EMA
    combined_df["side_win_likelihood"] = combined_df.apply(lambda row: calculate_side_win_likelihood(row), axis=1)

    combined_df = combined_df.reset_index(drop=True)
    return combined_df
