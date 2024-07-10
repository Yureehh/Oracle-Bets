"""
Patch win rate

This module provides functionality to compute the patch games count and win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from typing import Union

import pandas as pd

import src.ingestion.oracles_elixir as oe
from src.utils.paths import DEFAULT_MODELS_PARAMETERS
from src.utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_patch(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute total games, wins, win rate, and EWM for win rates grouped by patch.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing match data.
        identity (str): The identity column to group by (e.g., 'player' or 'team').

    Returns:
        pd.DataFrame: DataFrame with computed EWM for patch win rates.
    """
    # Adding total games and wins calculation directly in the EMA computation
    grouped = df.groupby([identity, "patch"])

    # Compute EMA before and after for win rate
    df["ema_patch_win_rate_before"] = grouped["result"].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean().shift().bfill()
    )
    df["ema_patch_win_rate_after"] = grouped["result"].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean()
    )
    return df


def calculate_patch_win_likelihood(ema_win_rate: float, opp_ema_win_rate: float) -> Union[float, None]:
    """
    Calculate the EMA patch win likelihood.

    Parameters:
        ema_win_rate (float): The EMA win rate for the entity.
        opp_ema_win_rate (float): The EMA win rate for the opponent.

    Returns:
        float: The calculated patch win likelihood.
    """
    if pd.notnull(ema_win_rate) and pd.notnull(opp_ema_win_rate):
        return round(ema_win_rate / (ema_win_rate + opp_ema_win_rate + EPSILON), 3)
    return None


def patch_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Compute patch-wise EWM computation integrated with games count and win rates.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing match data.
        entity (str): The entity type ('player' or 'team').

    Returns:
        pd.DataFrame: DataFrame with computed patch win rates and EWM.

    Raises:
        ValueError: If the entity is not 'player' or 'team'.
    """
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")

    identity = get_identity(entity)
    df.sort_values(get_sorting_keys(entity), inplace=True)

    # Compute EMA along with games and win rates
    df = compute_ema_patch(df, identity)

    # Compute Opponent Columns
    df["opp_ema_patch_win_rate_before"] = oe.get_opponent(df["ema_patch_win_rate_before"].tolist(), entity)

    # Calculate win likelihood based on EMA
    df["patch_win_likelihood"] = df.apply(
        lambda row: calculate_patch_win_likelihood(
            row["ema_patch_win_rate_before"], row["opp_ema_patch_win_rate_before"]
        ),
        axis=1,
    )

    return df.reset_index(drop=True)
