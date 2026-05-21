"""League selection helpers for LoL ingestion and ratings."""

from __future__ import annotations

from typing import Any

from oracle_bets_core.io_utils import json_loader
from oracle_bets_core.paths import CONSIDERED_LEAGUES


def load_league_selection_config() -> dict[str, Any]:
    """Load the league selection configuration."""
    return json_loader(CONSIDERED_LEAGUES)


def selected_leagues() -> list[str]:
    """Return leagues from the active profile, falling back to legacy config."""
    config = load_league_selection_config()
    profiles = config.get("profiles", {})
    active_profile = config.get("active_profile")

    if active_profile:
        try:
            return list(profiles[active_profile])
        except KeyError as exc:
            msg = f"Active league profile not found: {active_profile}"
            raise KeyError(msg) from exc

    return list(config["considered_leagues"])


def cross_league_competitions() -> set[str]:
    """Return cross-league competitions used by rating models."""
    return set(load_league_selection_config()["cross_league_competitions"])


def major_leagues() -> list[str]:
    """Return leagues treated as major for rating transfer rules."""
    return list(load_league_selection_config()["major_leagues"])
