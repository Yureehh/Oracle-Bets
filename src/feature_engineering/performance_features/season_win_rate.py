"""
Season Win Rate Module

This module provides functionality to compute the season games count and win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

import fireducks.pandas as pd

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
    """
    df = df.copy()
    grouped = df.groupby([identity, "season"])["result"]

    # Compute EMA before and after for win rate
    df["ema_season_win_rate_before"] = grouped.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=False, ignore_na=True).mean().shift().bfill()
    )
    df["ema_season_win_rate_after"] = grouped.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=False, ignore_na=True).mean()
    )
    return df


def calculate_season_win_likelihood(ema_win_rate: pd.Series, opp_ema_win_rate: pd.Series) -> pd.Series:
    """
    Calculate the EMA season win likelihood.

    Args:
        ema_win_rate (pd.Series): The EMA win rate for the entity.
        opp_ema_win_rate (pd.Series): The EMA win rate for the opponent.

    Returns:
        pd.Series: The calculated season win likelihood.
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
        pd.DataFrame: DataFrame with computed season win rates and EWM.

    Raises:
        ValueError: If the entity is not 'player' or 'team'.
    """
    if entity.lower() not in {"player", "team"}:
        raise ValueError("Entity must be either 'player' or 'team'.")

    identity = get_identity(entity)
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Compute EMA along with games and win rates
    df = compute_ema_season(df, identity)

    # Compute Opponent Columns
    df["opp_ema_season_win_rate_before"] = get_opponent(df["ema_season_win_rate_before"].tolist(), entity=entity)

    # Calculate win likelihood based on EMA
    df["season_win_likelihood"] = df.apply(
        lambda row: calculate_season_win_likelihood(
            row["ema_season_win_rate_before"], row["opp_ema_season_win_rate_before"]
        ),
        axis=1,
    )

    df = df.reset_index(drop=True)
    return df
