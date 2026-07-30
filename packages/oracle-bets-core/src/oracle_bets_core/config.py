"""Strict loader for the small set of product rules used at runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from oracle_bets_core.paths import PRODUCT_CONFIG

FIXTURE_WINDOW_HOURS = 36
NEW_MAJOR_MAPS_TRIGGER = 20
NEW_VALID_MAPS_TRIGGER = 50
OPERATIONAL_LEAGUE_PROFILE = "tier1_plus_erls"
SCHEMA_VERSION = 2


class ProductConfigError(ValueError):
    """Raised when product configuration contradicts its schema or safety rules."""


def _expect_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ProductConfigError(f"{context} must be an object.")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ProductConfigError(
            f"Unknown {context} fields: {', '.join(sorted(unknown))}."
        )
    if missing:
        raise ProductConfigError(
            f"Missing {context} fields: {', '.join(sorted(missing))}."
        )


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProductConfigError(f"{field_name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class MarketConfig:
    read_only: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MarketConfig:
        _expect_keys(value, {"read_only"}, "market")
        if value["read_only"] is not True:
            raise ProductConfigError("market comparison must remain read-only.")
        return cls(read_only=True)


@dataclass(frozen=True)
class LeagueConfig:
    profile: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LeagueConfig:
        _expect_keys(value, {"profile"}, "leagues")
        if value["profile"] != OPERATIONAL_LEAGUE_PROFILE:
            raise ProductConfigError(
                f"leagues.profile must remain {OPERATIONAL_LEAGUE_PROFILE}."
            )
        return cls(profile=value["profile"])


@dataclass(frozen=True)
class TrainingConfig:
    new_valid_maps_trigger: int
    new_major_maps_trigger: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingConfig:
        _expect_keys(
            value,
            {"new_valid_maps_trigger", "new_major_maps_trigger"},
            "training",
        )
        valid = _positive_int(
            value["new_valid_maps_trigger"], "training.new_valid_maps_trigger"
        )
        major = _positive_int(
            value["new_major_maps_trigger"], "training.new_major_maps_trigger"
        )
        if valid != NEW_VALID_MAPS_TRIGGER or major != NEW_MAJOR_MAPS_TRIGGER:
            raise ProductConfigError(
                "training triggers must remain 50 valid and 20 major-league maps."
            )
        return cls(valid, major)


@dataclass(frozen=True)
class PromotionConfig:
    automatic: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PromotionConfig:
        _expect_keys(value, {"automatic"}, "promotion")
        if value["automatic"] is not False:
            raise ProductConfigError("automatic model promotion must remain disabled.")
        return cls(automatic=False)


@dataclass(frozen=True)
class ProductConfig:
    schema_version: int
    timezone: str
    fixture_window_hours: int
    market: MarketConfig
    leagues: LeagueConfig
    training: TrainingConfig
    promotion: PromotionConfig

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProductConfig:
        expected = {
            "schema_version",
            "timezone",
            "fixture_window_hours",
            "market",
            "leagues",
            "training",
            "promotion",
        }
        _expect_keys(value, expected, "product config")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ProductConfigError(
                f"product config schema_version must be {SCHEMA_VERSION}."
            )
        try:
            ZoneInfo(value["timezone"])
        except (KeyError, TypeError, ZoneInfoNotFoundError) as exc:
            raise ProductConfigError(
                "product timezone must name an installed IANA timezone."
            ) from exc
        if value["timezone"] != "Europe/Rome":
            raise ProductConfigError("product timezone must remain Europe/Rome.")
        fixture_window = _positive_int(
            value["fixture_window_hours"], "fixture_window_hours"
        )
        if fixture_window != FIXTURE_WINDOW_HOURS:
            raise ProductConfigError("fixture_window_hours must remain 36.")
        return cls(
            schema_version=SCHEMA_VERSION,
            timezone=value["timezone"],
            fixture_window_hours=fixture_window,
            market=MarketConfig.from_dict(value["market"]),
            leagues=LeagueConfig.from_dict(value["leagues"]),
            training=TrainingConfig.from_dict(value["training"]),
            promotion=PromotionConfig.from_dict(value["promotion"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_product_config(path: str | Path = PRODUCT_CONFIG) -> ProductConfig:
    """Load and validate the versioned operating contract."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductConfigError(
            f"Could not load product config: {config_path}"
        ) from exc
    return ProductConfig.from_dict(raw)
