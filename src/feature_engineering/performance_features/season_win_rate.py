"""
Season win rate

This module provides functionality to compute the season games count and win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from typing import Union

import numpy as np
import pandas as pd

import src.ingestion.oracles_elixir as oe
from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.utils import get_identity, get_sorting_keys, json_loader

# Load configuration parameters
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE = config_params["half_life"]
EPSILON = 1e-8  # Small constant to prevent division by zero


def compute_ema_season(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute total games, wins, win rate, and EWM for win rates grouped by season.
    """
    # Adding total games and wins calculation directly in the EMA computation
    grouped = df.groupby([identity, "season"])
    df["season_total_games"] = grouped["gameid"].transform("size")
    df["season_wins"] = grouped["result"].transform("sum")
    df["season_win_rate"] = df["season_wins"] / df["season_total_games"]

    # Compute EMA before and after for win rate
    df["ema_season_win_rate_before"] = grouped["result"].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean().shift().bfill()
    )
    df["ema_season_win_rate_after"] = grouped["result"].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean()
    )
    df["ema_season_win_rate_before"] = grouped["ema_season_win_rate_before"].transform(lambda x: x.bfill().ffill())

    return df


def calculate_ema_season_win_likelihood(row: pd.Series) -> Union[float, np.nan]:
    """
    Calculate the EMA season win likelihood.
    """
    ema_win_rate = row["ema_season_win_rate_before"]
    opp_ema_win_rate = row["opp_season_ema_win_rate_before"]
    if pd.notnull(ema_win_rate) and pd.notnull(opp_ema_win_rate):
        return round(ema_win_rate / (ema_win_rate + opp_ema_win_rate + EPSILON), 3)
    return np.nan


def season_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Compute season-wise EWM computation integrated with games count and win rates.
    """
    if entity.lower() not in ["player", "team"]:
        raise ValueError("Entity must be either 'player' or 'team'.")

    identity = get_identity(entity)
    df.sort_values(get_sorting_keys(entity), inplace=True)

    # Compute EMA along with games and win rates
    df = compute_ema_season(df, identity)

    # Compute Opponent Columns
    df["opp_season_ema_win_rate_before"] = oe.get_opponent(df["ema_season_win_rate_before"].to_list(), entity)

    # Calculate win likelihood based on EMA
    df["ema_season_win_likelihood"] = df.apply(calculate_ema_season_win_likelihood, axis=1)

    return df.reset_index(drop=True)
