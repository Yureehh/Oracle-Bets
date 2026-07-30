"""
League taxonomy helpers.

Provides a centralized mapping from league -> region/tier/strength_pool and
utilities to attach those columns to dataframes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.paths import LEAGUE_TAXONOMY

if TYPE_CHECKING:
    from oracle_bets_core.pd import pd

UNKNOWN_LEAGUE_SAMPLE_SIZE = 10


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict[str, Any]:
    loaded = json_loader(LEAGUE_TAXONOMY)
    if not isinstance(loaded, dict):
        raise TypeError("league taxonomy configuration must be an object")
    data = cast("dict[str, Any]", loaded)
    data.setdefault("defaults", {})
    data.setdefault("leagues", {})
    return data


def get_league_taxonomy(league: str | None) -> dict[str, Any]:
    taxonomy = _load_taxonomy()
    defaults = taxonomy.get("defaults", {})
    leagues = taxonomy.get("leagues", {})
    if league is None:
        return {
            "region": defaults.get("region", "Unknown"),
            "tier": defaults.get("tier", "minor"),
            "strength_pool": defaults.get("strength_pool", "minor"),
        }
    entry = leagues.get(str(league), {})
    return {
        "region": entry.get("region", defaults.get("region", "Unknown")),
        "tier": entry.get("tier", defaults.get("tier", "minor")),
        "strength_pool": entry.get(
            "strength_pool", defaults.get("strength_pool", "minor")
        ),
    }


def add_league_taxonomy_columns(
    df: pd.DataFrame, league_col: str = "league", *, strict: bool = True
) -> pd.DataFrame:
    if league_col not in df.columns:
        raise ValueError(f"Missing '{league_col}' column for league taxonomy mapping.")

    taxonomy = _load_taxonomy()
    known = set(taxonomy.get("leagues", {}))
    leagues = set(df[league_col].dropna().astype(str))
    unknown = sorted(leagues - known)
    if strict and unknown:
        msg = (
            "League taxonomy missing entries for: "
            f"{unknown[:UNKNOWN_LEAGUE_SAMPLE_SIZE]}"
            + (" ..." if len(unknown) > UNKNOWN_LEAGUE_SAMPLE_SIZE else "")
        )
        raise ValueError(msg)

    out = df.copy()
    mapped = out[league_col].map(get_league_taxonomy)
    out["league_region"] = mapped.map(lambda x: x["region"])
    out["league_tier"] = mapped.map(lambda x: x["tier"])
    out["strength_pool"] = mapped.map(lambda x: x["strength_pool"])
    return out
