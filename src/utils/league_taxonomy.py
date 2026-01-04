"""
League taxonomy helpers.

Provides a centralized mapping from league -> region/tier/strength prior and
utilities to attach those columns to dataframes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from utils.io_utils import FileLoadError, json_loader
from utils.paths import LEAGUE_STRENGTH_PRIORS, LEAGUE_TAXONOMY

if TYPE_CHECKING:
    import pandas as pd


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict[str, Any]:
    data: dict[str, Any] = json_loader(LEAGUE_TAXONOMY)
    data.setdefault("defaults", {})
    data.setdefault("leagues", {})
    return data


@lru_cache(maxsize=1)
def _load_strength_priors() -> dict[str, float]:
    try:
        priors: dict[str, float] = json_loader(LEAGUE_STRENGTH_PRIORS)
    except FileLoadError:
        return {}
    return {str(k): float(v) for k, v in priors.items()}


def refresh_league_taxonomy_cache() -> None:
    """Clear cached league taxonomy + calibrated priors (useful after regen)."""
    _load_taxonomy.cache_clear()
    _load_strength_priors.cache_clear()


def get_prior_settings() -> dict[str, Any]:
    """Return calibration settings stored alongside league taxonomy."""
    try:
        taxonomy = _load_taxonomy()
    except FileLoadError:
        return {}
    settings = taxonomy.get("prior_settings", {})
    return settings if isinstance(settings, dict) else {}


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


def get_config_strength_prior(league: str | None) -> float:
    """Strength prior from static taxonomy config only (no calibration)."""
    return float(get_league_taxonomy(league)["strength_prior"])


def get_league_strength_prior(league: str | None) -> float:
    """Strength prior from calibrated priors when available, else taxonomy defaults."""
    if league is None:
        return float(get_league_taxonomy(None)["strength_prior"])
    priors = _load_strength_priors()
    if str(league) in priors:
        return float(priors[str(league)])
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
