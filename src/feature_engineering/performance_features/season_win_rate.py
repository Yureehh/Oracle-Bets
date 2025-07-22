"""
Season Win Rate Module

This module provides functionality to compute the season games count and win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

import pandas as pd

from ingestion.oracles_elixir import get_opponent
from src.utils.paths import DEFAULT_MODELS_PARAMETERS
from src.utils.utils import get_identity, get_sorting_keys, json_loader

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_season(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for season win rates grouped by identity and season.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        identity (str): The identity column to group by (e.g., 'playerid' or 'teamid').

    Returns:
        pd.DataFrame: DataFrame with computed EWM for season win rates.
                     Columns: 'ema_season_win_rate_before', 'ema_season_win_rate_after'.

    """
    df = df.copy()
    grouped = df.groupby([identity, "season"])["result"]

    # "before" uses shift() to avoid leaking current match result
    df["ema_season_win_rate_before"] = (
        grouped.transform(
            lambda x: x.ewm(halflife=HALF_LIFE, adjust=False, ignore_na=True).mean()
        )
        .shift()
        .bfill()
    )  # bfill ensures no NaNs. Consider leaving as NaN to avoid any data leak in first row.

    # "after" includes the current row's result
    df["ema_season_win_rate_after"] = grouped.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=False, ignore_na=True).mean()
    )

    return df


def calculate_season_win_likelihood(
    ema_win_rate: pd.Series, opp_ema_win_rate: pd.Series
) -> pd.Series:
    """
    Calculate the EMA season win likelihood.

    Args:
        ema_win_rate (pd.Series): The EMA win rate for the entity.
        opp_ema_win_rate (pd.Series): The EMA win rate for the opponent.

    Returns:
        pd.Series: The calculated season win likelihood per row.

    """
    total_win_rate = ema_win_rate + opp_ema_win_rate + EPSILON
    win_likelihood = ema_win_rate / total_win_rate
    return round(win_likelihood, 3)


def season_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Compute season-wise EWM computation integrated with games count and win rates.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        entity (str): The entity type ('player' or 'team').

    Returns:
        pd.DataFrame: A DataFrame with columns for EWM of season win rates (before/after),
                      opponent's EWM, and a 'season_win_likelihood' measure.

    """
    if entity.lower() not in {"player", "team"}:
        msg = "Entity must be either 'player' or 'team'."
        raise ValueError(msg)

    identity = get_identity(entity)
    # Sort the DataFrame for correct chronological or logical EWM calculation
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Compute EWM for the season win rate
    df = compute_ema_season(df, identity)

    # Compute Opponent EWM columns (using "before" as a baseline)
    df["opp_ema_season_win_rate_before"] = get_opponent(
        df["ema_season_win_rate_before"].tolist(), entity=entity
    )

    # Calculate season win likelihood based on EWM
    df["season_win_likelihood"] = df.apply(
        lambda row: calculate_season_win_likelihood(
            row["ema_season_win_rate_before"], row["opp_ema_season_win_rate_before"]
        ),
        axis=1,
    )

    # Re-index after transformations
    return df.reset_index(drop=True)
