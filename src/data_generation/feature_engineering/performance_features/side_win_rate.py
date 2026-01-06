"""
Side Win Rate Module

This module provides functionality to compute the side win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from __future__ import annotations

import numpy as np

from data_generation.ingestion.oracles_elixir import get_opponent
from utils.io_utils import get_identity, get_sorting_keys, json_loader
from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.pd import pd

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE: float = float(config_params["half_life"])
EPSILON = 1e-8  # Small constant to prevent division by zero


def _validate_inputs(df: pd.DataFrame, identity: str) -> None:
    """Ensure required columns exist and result is numeric 0/1."""
    required = {identity, "side", "result"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Input DataFrame is missing required columns: {missing}"
        raise ValueError(msg)

    if not pd.api.types.is_numeric_dtype(df["result"]):
        valmap = {
            "W": 1,
            "Win": 1,
            "win": 1,
            "Won": 1,
            "won": 1,
            True: 1,
            "L": 0,
            "Loss": 0,
            "loss": 0,
            "Lose": 0,
            "lose": 0,
            False: 0,
        }
        df["result"] = df["result"].map(valmap).astype("float64")


def _compute_side_ema_all(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute EWM side win rates (before/after) and EMA-weighted games counts
    for both 'Blue' and 'Red' sides in a single pass.

    Adds:
      - ema_blue_side_before / after
      - ema_red_side_before / after
      - ema_blue_side_games_before / after
      - ema_red_side_games_before / after
    """
    df = df.copy()
    _validate_inputs(df, identity)

    # Group by (identity, side), assuming proper chronological sorting upstream
    g = df.groupby([identity, "side"], sort=False)["result"]

    # EMA win-rate per group
    ema_after_all = g.transform(
        lambda s: s.ewm(halflife=HALF_LIFE, adjust=True, ignore_na=True).mean()
    )
    ema_before_all = g.transform(
        lambda s: s.ewm(halflife=HALF_LIFE, adjust=True, ignore_na=True).mean().shift()
    )

    # EMA "games" (trust) per group
    games_after_all = g.transform(
        lambda s: pd.Series(1.0, index=s.index)
        .ewm(halflife=HALF_LIFE, adjust=True)
        .sum()
    )
    # IMPORTANT: shift per-group (not globally)
    games_before_all = g.transform(
        lambda s: pd.Series(1.0, index=s.index)
        .ewm(halflife=HALF_LIFE, adjust=True)
        .sum()
        .shift()
    )

    # Allocate side-specific columns, then assign by mask
    for col in [
        "ema_blue_side_before",
        "ema_blue_side_after",
        "ema_red_side_before",
        "ema_red_side_after",
        "ema_blue_side_games_before",
        "ema_blue_side_games_after",
        "ema_red_side_games_before",
        "ema_red_side_games_after",
    ]:
        df[col] = np.nan

    blue_mask = df["side"].eq("Blue")
    red_mask = df["side"].eq("Red")

    df.loc[blue_mask, "ema_blue_side_before"] = ema_before_all[blue_mask]
    df.loc[blue_mask, "ema_blue_side_after"] = ema_after_all[blue_mask]
    df.loc[blue_mask, "ema_blue_side_games_before"] = games_before_all[blue_mask]
    df.loc[blue_mask, "ema_blue_side_games_after"] = games_after_all[blue_mask]

    df.loc[red_mask, "ema_red_side_before"] = ema_before_all[red_mask]
    df.loc[red_mask, "ema_red_side_after"] = ema_after_all[red_mask]
    df.loc[red_mask, "ema_red_side_games_before"] = games_before_all[red_mask]
    df.loc[red_mask, "ema_red_side_games_after"] = games_after_all[red_mask]

    # Carry past values of the *other* side forward (ffill only to avoid future leak)
    by_id = df.groupby(identity, sort=False)
    ffill_cols = [
        "ema_blue_side_before",
        "ema_blue_side_after",
        "ema_red_side_before",
        "ema_red_side_after",
        "ema_blue_side_games_before",
        "ema_blue_side_games_after",
        "ema_red_side_games_before",
        "ema_red_side_games_after",
    ]
    df[ffill_cols] = by_id[ffill_cols].ffill()

    # Neutral priors where history is missing
    df["ema_blue_side_before"] = df["ema_blue_side_before"].fillna(0.5)
    df["ema_red_side_before"] = df["ema_red_side_before"].fillna(0.5)
    df["ema_blue_side_games_before"] = df["ema_blue_side_games_before"].fillna(0.0)
    df["ema_red_side_games_before"] = df["ema_red_side_games_before"].fillna(0.0)

    return df


def _side_likelihood_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    P(win) = my_side_before / (my_side_before + opp_side_before + EPSILON)
    Uses opponent’s EMA for the *opposite* side.
    """
    is_blue = df["side"].eq("Blue")
    my_before = np.where(is_blue, df["ema_blue_side_before"], df["ema_red_side_before"])
    opp_before = np.where(
        is_blue, df["opp_ema_red_side_before"], df["opp_ema_blue_side_before"]
    )
    denom = my_before + opp_before + EPSILON
    return (my_before / denom).clip(0.0, 1.0)


def side_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.

    Adds columns:
      - ema_blue_side_before / after
      - ema_red_side_before / after
      - ema_blue_side_games_before / after
      - ema_red_side_games_before / after
      - opp_ema_blue_side_before / opp_ema_red_side_before
      - side_win_likelihood
    """
    if entity.lower() not in {"player", "team"}:
        msg = "Entity must be either 'player' or 'team'."
        raise ValueError(msg)

    identity = get_identity(entity)
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Single-pass EMA features for both sides
    df = _compute_side_ema_all(df, identity)

    # Opponent EMA “before” columns
    df["opp_ema_blue_side_before"] = get_opponent(
        df["ema_blue_side_before"].tolist(), entity=entity
    )
    df["opp_ema_red_side_before"] = get_opponent(
        df["ema_red_side_before"].tolist(), entity=entity
    )

    # Vectorized likelihood
    df["side_win_likelihood"] = _side_likelihood_vectorized(df)

    return df.reset_index(drop=True)
