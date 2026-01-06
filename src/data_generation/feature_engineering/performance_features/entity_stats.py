"""
Entity Statistics Module

This module contains functions to calculate entity-specific EMA statistics
(e.g., KDA, KP, ratios, efficiencies), producing leak-free 'before' and
updated 'after' columns. It intentionally avoids recomputing EMAs that are
handled elsewhere (side, patch, season win rates).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tqdm import tqdm

from data_generation.ingestion.oracles_elixir import get_opponent
from utils.io_utils import get_identity, get_sorting_keys, json_loader
from utils.paths import (
    DEFAULT_MODELS_PARAMETERS,
    FLATTENED_PLAYER_CONFIG,
    FLATTENED_TEAM_CONFIG,
)
from utils.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
config_params = json_loader(DEFAULT_MODELS_PARAMETERS)
HALF_LIFE: float = float(config_params["half_life"])

# Columns whose EMAs are computed in dedicated modules (do not redo here)
_ALREADY_HANDLED_EMAS = {
    "ema_red_side_after",
    "ema_blue_side_after",
    "ema_patch_win_rate_after",
    "ema_season_win_rate_after",
}

# Ratings / league-aggregates that should NEVER be EMA'd here (future-proof)
_DISALLOWED_BASES = {
    "elo",
    "league_elo",
    "glicko2_mu",
    "glicko2_phi",
    "pl_mu",
    "pl_sigma",
    "trueskill_mu",
    "trueskill_sigma",
}

# Neutral priors ONLY for a few rate-like stats; applied to *_before to avoid NaN on first obs.
NEUTRAL_PRIORS: dict[str, float] = {
    "kill_participation": 0.5,
    "ka_ratio": 0.5,
}


# ---------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------
def _load_flattened_config_path(entity: str) -> str:
    if entity == "team":
        return FLATTENED_TEAM_CONFIG
    if entity == "player":
        return FLATTENED_PLAYER_CONFIG
    msg = "Entity must be either 'team' or 'player'."
    raise ValueError(msg)


def _select_base_columns_from_flattened_config(
    entity: str, df: pd.DataFrame
) -> list[str]:
    """
    Read flattened config, find columns that look like 'ema_*_after', drop the ones
    we intentionally don't recompute here (side/patch/season EMAs), then derive the
    base metric name by stripping the leading 'ema_' and trailing '_after'.

    Finally, keep only base columns that exist in `df`, are numeric, and not disallowed.
    """
    cfg_path = _load_flattened_config_path(entity)
    flattened_cols: list[str] = json_loader(cfg_path)["flattened_cols"]

    # Only EMA-after candidates and not in the "already-handled" skip list
    ema_after_cols = [
        c
        for c in flattened_cols
        if c.startswith("ema_")
        and c.endswith("_after")
        and c not in _ALREADY_HANDLED_EMAS
    ]

    # Derive base names via anchored regex (avoid accidental replacements inside names)
    base_candidates = [
        re.sub(r"_after$", "", re.sub(r"^ema_", "", c)) for c in ema_after_cols
    ]

    # Drop disallowed bases (ratings, league elo, etc.)
    base_candidates = [b for b in base_candidates if b not in _DISALLOWED_BASES]

    # Keep only those present and numeric in the current dataframe
    numeric_like = {col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    base_cols = [c for c in base_candidates if c in numeric_like]
    return sorted(set(base_cols))


# ---------------------------------------------------------------------
# EMA computation
# ---------------------------------------------------------------------
def _ema(series: pd.Series) -> pd.Series:
    return series.ewm(halflife=HALF_LIFE, adjust=True, ignore_na=True).mean()


def apply_ema(df: pd.DataFrame, identity: str, columns: Iterable[str]) -> pd.DataFrame:
    """
    Apply EMA ('before' and 'after') for each base column in `columns`,
    grouped by the entity identity. 'before' is shifted to be leak-free.
    """
    out = df.copy()
    new_cols = {}  # <- collect here

    for col in tqdm(list(columns), desc="Calculating EMA"):
        grp = out.groupby(identity, sort=False, observed=True)[col]
        ema_after = grp.transform(_ema)
        ema_before = ema_after.groupby(out[identity], sort=False).shift()

        new_cols[f"ema_{col}_after"] = ema_after
        new_cols[f"ema_{col}_before"] = ema_before

    # single concat avoids fragmentation
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


# ---------------------------------------------------------------------
# Opponent mirroring
# ---------------------------------------------------------------------
def apply_opponent_stats(
    df: pd.DataFrame, entity: str, columns: Iterable[str]
) -> pd.DataFrame:
    """
    For each 'ema_{col}_before', add 'opp_ema_{col}_before' using the
    game-join logic in `get_opponent`. Keeps order and index.
    """
    out = df.copy()
    opp_cols = {}

    for col in columns:
        before_name = f"ema_{col}_before"
        if before_name not in out.columns:
            continue
        vals = pd.Series(out[before_name].to_numpy().flatten())
        opp_cols[f"opp_{before_name}"] = get_opponent(vals, entity=entity)

    if opp_cols:
        out = pd.concat([out, pd.DataFrame(opp_cols, index=out.index)], axis=1)
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def enrich_entity_ema_statistics(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    High-level entry point:
      1) sort (for chronological stability),
      2) choose base columns from flattened config (minus side/patch/season EMAs and ratings),
      3) compute EMAs (before/after),
      4) add opponent 'before' EMAs.

    Notes
    -----
    - We do not fill 'before' NaNs; first observations per entity will be NaN by design.
      Keep them or impute later (neutral priors), but do not bfill to avoid leakage.

    """
    if entity not in {"team", "player"}:
        msg = "Entity must be either 'team' or 'player'."
        raise ValueError(msg)

    out = df.sort_values(get_sorting_keys(entity)).reset_index(drop=True)
    identity = get_identity(entity)

    base_cols = _select_base_columns_from_flattened_config(entity, out)
    if not base_cols:
        return out

    out = apply_ema(out, identity, base_cols)

    # neutral priors for a few rate-like *_before EMAs
    for base, prior in NEUTRAL_PRIORS.items():
        col = f"ema_{base}_before"
        if col in out.columns:
            out[col] = out[col].fillna(prior)

    return apply_opponent_stats(out, entity, base_cols)
