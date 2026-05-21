"""League selection helpers for LoL ingestion."""

from __future__ import annotations

from typing import Any

from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.paths import CONSIDERED_LEAGUES


def load_league_selection_config() -> dict[str, Any]:
    """Load the league selection configuration."""
    return json_loader(CONSIDERED_LEAGUES)


def selected_leagues() -> list[str]:
    """Return leagues from the active profile."""
    config = load_league_selection_config()
    try:
        active_profile = config["active_profile"]
        return list(config["profiles"][active_profile])
    except KeyError as exc:
        msg = "League selection config must define active_profile and profiles."
        raise KeyError(msg) from exc
