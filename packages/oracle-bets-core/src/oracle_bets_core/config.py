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
ACTIONABLE_LEAGUE_EXCLUSIONS = ("CBLOL", "LCP")
PREDICTION_LEAGUE_PROFILE = "tier1_plus_erls"
RESEARCH_LEAGUE_PROFILE = "research_all_supported"
SCHEMA_VERSION = 4
QUOTE_TTL_SECONDS = 120


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
    quote_ttl_seconds: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MarketConfig:
        _expect_keys(value, {"read_only", "quote_ttl_seconds"}, "market")
        if value["read_only"] is not True:
            raise ProductConfigError("market comparison must remain read-only.")
        ttl = _positive_int(value["quote_ttl_seconds"], "market.quote_ttl_seconds")
        if ttl != QUOTE_TTL_SECONDS:
            raise ProductConfigError("market.quote_ttl_seconds must remain 120.")
        return cls(read_only=True, quote_ttl_seconds=ttl)


@dataclass(frozen=True)
class StrategyConfig:
    policy_version: str
    recommendation_target: str
    minimum_model_favorite: float
    unit_bankroll_fraction: float
    paper_positive_ev_path: str
    paper_negative_ev_exploration_path: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StrategyConfig:
        expected = {
            "policy_version",
            "recommendation_target",
            "minimum_model_favorite",
            "unit_bankroll_fraction",
            "paper_positive_ev_path",
            "paper_negative_ev_exploration_path",
        }
        _expect_keys(value, expected, "strategy")
        required = {
            "policy_version": "lol-market-policy-v1",
            "recommendation_target": "series_winner",
            "minimum_model_favorite": 0.51,
            "unit_bankroll_fraction": 0.01,
            "paper_positive_ev_path": "full_kelly",
            "paper_negative_ev_exploration_path": "flat_1u",
        }
        if any(
            value.get(key) != expected_value for key, expected_value in required.items()
        ):
            raise ProductConfigError("strategy values must match the versioned policy.")
        return cls(**required)


@dataclass(frozen=True)
class LeagueConfig:
    training_profile: str
    prediction_profile: str
    actionable_exclusions: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LeagueConfig:
        _expect_keys(
            value,
            {"training_profile", "prediction_profile", "actionable_exclusions"},
            "leagues",
        )
        if value["training_profile"] != RESEARCH_LEAGUE_PROFILE:
            raise ProductConfigError(
                f"leagues.training_profile must remain {RESEARCH_LEAGUE_PROFILE}."
            )
        if value["prediction_profile"] != PREDICTION_LEAGUE_PROFILE:
            raise ProductConfigError(
                f"leagues.prediction_profile must remain {PREDICTION_LEAGUE_PROFILE}."
            )
        exclusions = value["actionable_exclusions"]
        if not isinstance(exclusions, list) or tuple(sorted(exclusions)) != (
            ACTIONABLE_LEAGUE_EXCLUSIONS
        ):
            raise ProductConfigError(
                "leagues.actionable_exclusions must contain exactly CBLOL and LCP."
            )
        return cls(
            training_profile=value["training_profile"],
            prediction_profile=value["prediction_profile"],
            actionable_exclusions=tuple(sorted(exclusions)),
        )


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
    routine_automatic: bool
    optuna_automatic: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PromotionConfig:
        _expect_keys(value, {"routine_automatic", "optuna_automatic"}, "promotion")
        if value["routine_automatic"] is not True:
            raise ProductConfigError(
                "routine automatic model promotion must remain enabled."
            )
        if value["optuna_automatic"] is not False:
            raise ProductConfigError(
                "Optuna candidates must never promote automatically."
            )
        return cls(routine_automatic=True, optuna_automatic=False)


@dataclass(frozen=True)
class ProductConfig:
    schema_version: int
    timezone: str
    fixture_window_hours: int
    market: MarketConfig
    leagues: LeagueConfig
    training: TrainingConfig
    promotion: PromotionConfig
    strategy: StrategyConfig

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
            "strategy",
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
            strategy=StrategyConfig.from_dict(value["strategy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["leagues"]["actionable_exclusions"] = list(
            self.leagues.actionable_exclusions
        )
        return payload


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
