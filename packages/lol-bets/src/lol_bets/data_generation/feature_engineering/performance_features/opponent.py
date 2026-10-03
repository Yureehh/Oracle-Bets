"""Opponent feature pairing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from oracle_bets_core.pd import pd

OPPOSITE_SIDE = {"Blue": "Red", "Red": "Blue"}


def add_opponent_columns(
    df: pd.DataFrame,
    *,
    entity: str,
    source_columns: Iterable[str],
) -> pd.DataFrame:
    """Pair source columns from the opposite side in the same game."""
    if entity not in {"team", "player"}:
        msg = "Entity must be either 'team' or 'player'."
        raise ValueError(msg)

    keys = ["gameid", "side"]
    if entity == "player":
        keys.append("position")

    source = list(source_columns)
    missing = set(keys + source) - set(df.columns)
    if missing:
        msg = f"Missing columns for opponent pairing: {sorted(missing)}"
        raise ValueError(msg)

    left = df.copy()
    left["_row_id"] = range(len(left))
    right = df[[*keys, *source]].copy()
    right["side"] = right["side"].map(OPPOSITE_SIDE)
    right = right.rename(columns={col: f"opp_{col}" for col in source})

    out = left.merge(right, on=keys, how="left", validate="many_to_one")
    out = out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"])
    out.index = df.index

    opp_cols = [f"opp_{col}" for col in source]
    if out[opp_cols].isna().any(axis=None):
        msg = f"Could not pair all {entity} opponent columns by game/side."
        raise ValueError(msg)
    return out
