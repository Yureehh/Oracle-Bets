"""
Patch Win Rate Module

This module provides functionality to compute the patch games count and win rate for a given entity,
using an Exponentially Weighted Mean (EWM) model.
"""

from __future__ import annotations

from data_generation.ingestion.oracles_elixir import get_opponent
from utils.io_utils import get_identity, get_sorting_keys, json_loader
from utils.paths import DEFAULT_MODELS_PARAMETERS
from utils.pd import pd

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE: float = float(config_params["half_life"])
EPSILON = 1e-8  # Small constant to prevent division by zero


def _validate_inputs(df: pd.DataFrame, identity: str) -> None:
    required = {identity, "patch", "result"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Input DataFrame is missing required columns: {missing}"
        raise ValueError(msg)
    # Ensure result is numeric 0/1
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


def compute_ema_patch(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for patch win rates grouped by identity and patch.

    Returns columns:
      - ema_patch_win_rate_before: EMA prior to the current game (leak-free; shifted per-group)
      - ema_patch_win_rate_after:  EMA including the current game
      - ema_patch_games_before:   EMA-weighted games count prior to the current game
      - ema_patch_games_after:    EMA-weighted games count including the current game
    """
    df = df.copy()
    _validate_inputs(df, identity)

    # Grouping (assumes df is already sorted in calling function)
    g_result = df.groupby([identity, "patch"], sort=False)["result"]

    # Win rate EMA
    ema_after = g_result.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=True, ignore_na=True).mean()
    )
    # >>> change 1: shift WITHIN group, not globally
    ema_before = ema_after.groupby([df[identity], df["patch"]], sort=False).shift()

    # A neutral prior for the first game in the patch (no history): 0.5
    df["ema_patch_win_rate_before"] = ema_before.fillna(0.5)
    df["ema_patch_win_rate_after"] = ema_after

    # EMA-weighted "games count" (trust signal)
    ones = pd.Series(1.0, index=df.index)
    g_ones = ones.groupby([df[identity], df["patch"]], sort=False)
    games_ema_after = g_ones.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=True).sum()
    )
    # >>> change 2: shift WITHIN group, not globally
    games_ema_before = (
        games_ema_after.groupby([df[identity], df["patch"]], sort=False)
        .shift()
        .fillna(0.0)
    )

    df["ema_patch_games_before"] = games_ema_before
    df["ema_patch_games_after"] = games_ema_after

    return df


def calculate_patch_win_likelihood(
    ema_win_rate: pd.Series, opp_ema_win_rate: pd.Series
) -> pd.Series:
    """
    Calculate the EMA patch win likelihood (symmetric ratio).
    NOTE: We keep full precision (no rounding) for modeling.
    """
    denom = ema_win_rate.add(opp_ema_win_rate).add(EPSILON)
    return ema_win_rate.div(denom).clip(0.0, 1.0)


def patch_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Compute patch-wise EWM computation integrated with games count and win rates.

    Adds columns:
      - ema_patch_win_rate_before / after
      - ema_patch_games_before / after
      - opp_ema_patch_win_rate_before
      - patch_win_likelihood
    """
    if entity.lower() not in {"player", "team"}:
        msg = "Entity must be either 'player' or 'team'."
        raise ValueError(msg)

    identity = get_identity(entity)
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # Compute EMA features
    df = compute_ema_patch(df, identity)

    # Opponent columns (keep existing helper signature)
    df["opp_ema_patch_win_rate_before"] = get_opponent(
        df["ema_patch_win_rate_before"].tolist(), entity=entity
    )

    # Vectorized likelihood (no per-row apply)
    df["patch_win_likelihood"] = calculate_patch_win_likelihood(
        df["ema_patch_win_rate_before"], df["opp_ema_patch_win_rate_before"]
    )

    return df.reset_index(drop=True)
