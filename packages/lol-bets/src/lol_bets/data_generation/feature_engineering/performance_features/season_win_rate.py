"""
Season Win Rate Module

This module provides functionality to compute the season games count and win rate
for a given entity, using an Exponentially Weighted Mean (EWM) model.
"""

from __future__ import annotations

from lol_bets.data_generation.ingestion.oracles_elixir import get_opponent
from oracle_bets_core.io_utils import get_identity, get_sorting_keys, json_loader
from oracle_bets_core.paths import DEFAULT_MODELS_PARAMETERS
from oracle_bets_core.pd import pd

# Constants
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE: float = float(config_params["half_life"])
EPSILON = 1e-8  # Small constant to prevent division by zero


def _validate_inputs(df: pd.DataFrame, identity: str) -> None:
    """Ensure required columns exist and result is numeric 0/1."""
    required = {identity, "season", "result"}
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


def compute_ema_season(df: pd.DataFrame, identity: str) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Mean (EWM) for season win rates grouped by identity and season.

    Returns columns:
      - ema_season_win_rate_before: EMA prior to the current game (leak-free; shifted, neutral prior 0.5)
      - ema_season_win_rate_after:  EMA including the current game
      - ema_season_games_before:   EMA-weighted games count prior to the current game
      - ema_season_games_after:    EMA-weighted games count including the current game
    """
    df = df.copy()
    _validate_inputs(df, identity)

    # Grouped result (assumes df sorted by get_sorting_keys(entity) upstream)
    g_result = df.groupby([identity, "season"], sort=False)["result"]

    # Win-rate EMA
    ema_after = g_result.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=True, ignore_na=True).mean()
    )
    ema_before = ema_after.groupby([df[identity], df["season"]], sort=False).shift()
    df["ema_season_win_rate_before"] = ema_before.fillna(0.5)  # neutral prior
    df["ema_season_win_rate_after"] = ema_after

    # EMA-weighted games count (trust signal)
    ones = pd.Series(1.0, index=df.index)
    g_ones = ones.groupby([df[identity], df["season"]], sort=False)
    games_after = g_ones.transform(
        lambda x: x.ewm(halflife=HALF_LIFE, adjust=True).sum()
    )
    games_before = (
        games_after.groupby([df[identity], df["season"]], sort=False)
        .shift()
        .fillna(0.0)
    )

    df["ema_season_games_before"] = games_before
    df["ema_season_games_after"] = games_after

    return df


def calculate_season_win_likelihood(
    ema_win_rate: pd.Series, opp_ema_win_rate: pd.Series
) -> pd.Series:
    """
    Calculate the EMA season win likelihood (symmetric ratio).
    We keep full precision for modeling and clip to [0, 1].
    """
    denom = ema_win_rate.add(opp_ema_win_rate).add(EPSILON)
    return ema_win_rate.div(denom).clip(0.0, 1.0)


def season_win_rate_ewm_performance(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Compute season-wise EWM computation integrated with games count and win rates.

    Adds columns:
      - ema_season_win_rate_before / after
      - ema_season_games_before / after
      - opp_ema_season_win_rate_before
      - season_win_likelihood
    """
    if entity.lower() not in {"player", "team"}:
        msg = "Entity must be either 'player' or 'team'."
        raise ValueError(msg)

    identity = get_identity(entity)
    df = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)

    # EMA features
    df = compute_ema_season(df, identity)

    # Opponent EMA (mirror the 'before' values)
    df["opp_ema_season_win_rate_before"] = get_opponent(
        df["ema_season_win_rate_before"].tolist(), entity=entity
    )

    # Vectorized likelihood
    df["season_win_likelihood"] = calculate_season_win_likelihood(
        df["ema_season_win_rate_before"], df["opp_ema_season_win_rate_before"]
    )

    return df.reset_index(drop=True)
