"""Shared safeguards for temporally valid rating features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from oracle_bets_core.pd import pd

from lol_bets.data_generation.feature_engineering.performance_features.opponent import (
    add_opponent_columns,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def freeze_same_date_rating_inputs(
    frame: pd.DataFrame,
    *,
    entity: str,
    rating_columns: Iterable[str],
) -> pd.DataFrame:
    """Give every same-date appearance the rating available before that date."""
    if entity not in {"team", "player"}:
        raise ValueError("entity must be 'team' or 'player'")
    out = frame.copy()
    identity = "teamid" if entity == "team" else "playerid"
    columns = list(rating_columns)
    day = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    keys = [out[identity], day]
    for column in columns:
        out[column] = out.groupby(keys, sort=False)[column].transform(
            lambda values: values.iloc[0]
        )
    out = out.drop(columns=[f"opp_{column}" for column in columns], errors="ignore")
    return add_opponent_columns(out, entity=entity, source_columns=columns)
