"""
League taxonomy helpers.

Provides a centralized mapping from league -> region/tier/strength prior and
utilities to attach those columns to dataframes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from utils.io_utils import json_loader
from utils.paths import LEAGUE_TAXONOMY


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict[str, Any]:
    data: dict[str, Any] = json_loader(LEAGUE_TAXONOMY)
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
            "strength_prior": defaults.get("strength_prior", 0),
        }
    entry = leagues.get(str(league), {})
    return {
        "region": entry.get("region", defaults.get("region", "Unknown")),
        "tier": entry.get("tier", defaults.get("tier", "minor")),
        "strength_prior": entry.get(
            "strength_prior", defaults.get("strength_prior", 0)
        ),
    }


def get_league_strength_prior(league: str | None) -> float:
    return float(get_league_taxonomy(league)["strength_prior"])


def add_league_taxonomy_columns(
    df: pd.DataFrame, league_col: str = "league"
) -> pd.DataFrame:
    if league_col not in df.columns:
        raise ValueError(f"Missing '{league_col}' column for league taxonomy mapping.")

    out = df.copy()
    mapped = out[league_col].map(get_league_taxonomy)
    out["league_region"] = mapped.map(lambda x: x["region"])
    out["league_tier"] = mapped.map(lambda x: x["tier"])
    out["league_strength_prior"] = mapped.map(lambda x: x["strength_prior"]).astype(
        float
    )
    return out
