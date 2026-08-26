"""League selection helpers for LoL ingestion."""

from __future__ import annotations

from typing import Any, cast

from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.paths import CONSIDERED_LEAGUES


def load_league_selection_config() -> dict[str, Any]:
    """Load the league selection configuration."""
    loaded = json_loader(CONSIDERED_LEAGUES)
    if not isinstance(loaded, dict):
        raise TypeError("league selection configuration must be an object")
    return cast("dict[str, Any]", loaded)


def selected_leagues(profile: str | None = None) -> list[str]:
    """Return leagues from the requested or active profile."""
    config = load_league_selection_config()
    try:
        active_profile = profile or config["active_profile"]
        return list(config["profiles"][active_profile])
    except KeyError as exc:
        msg = "League selection config must define active_profile and profiles."
        raise KeyError(msg) from exc


def training_leagues() -> list[str]:
    """Return the full historical research universe used by ingestion/training."""
    from oracle_bets_core.config import load_product_config

    return selected_leagues(load_product_config().leagues.training_profile)


def actionable_leagues() -> list[str]:
    """Return leagues allowed in owner-facing market reviews."""
    from oracle_bets_core.config import load_product_config

    product = load_product_config()
    excluded = set(product.leagues.actionable_exclusions)
    prediction = selected_leagues(product.leagues.prediction_profile)
    return [league for league in prediction if league not in excluded]
