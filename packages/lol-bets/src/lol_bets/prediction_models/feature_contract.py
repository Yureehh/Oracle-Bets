"""Versioned availability and role contract for LoL model columns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import Any


class FeatureContractError(ValueError):
    """Raised when a model feature is unknown or unavailable at decision time."""


class Availability(IntEnum):
    PREMATCH = 1
    POSTGAME = 2


class FeatureRole(StrEnum):
    INPUT = "input"
    IDENTIFIER = "identifier"
    TARGET = "target"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    role: FeatureRole
    availability: Availability
    missing_policy: str


@dataclass(frozen=True)
class FeatureRule:
    suffix: str
    family: str
    role: FeatureRole
    availability: Availability
    missing_policy: str


class FeatureRegistry:
    def __init__(self) -> None:
        self._exact: dict[str, FeatureSpec] = {}
        self._suffix_rules: list[FeatureRule] = []

    def register_exact(
        self,
        name: str,
        *,
        family: str,
        role: FeatureRole,
        availability: Availability,
        missing_policy: str,
    ) -> None:
        spec = FeatureSpec(
            name=name,
            family=family,
            role=FeatureRole(role),
            availability=Availability(availability),
            missing_policy=missing_policy,
        )
        existing = self._exact.get(name)
        if existing is not None and existing != spec:
            msg = f"Feature {name} already has a different contract."
            raise FeatureContractError(msg)
        self._exact[name] = spec

    def register_suffix(
        self,
        suffix: str,
        *,
        family: str,
        role: FeatureRole,
        availability: Availability,
        missing_policy: str,
    ) -> None:
        self._suffix_rules.append(
            FeatureRule(
                suffix=suffix,
                family=family,
                role=FeatureRole(role),
                availability=Availability(availability),
                missing_policy=missing_policy,
            )
        )

    def resolve(self, name: str) -> FeatureSpec:
        if name in self._exact:
            return self._exact[name]
        matches = [rule for rule in self._suffix_rules if name.endswith(rule.suffix)]
        if len(matches) != 1:
            msg = f"Feature {name!r} is unregistered or matches ambiguous rules."
            raise FeatureContractError(msg)
        rule = matches[0]
        return FeatureSpec(
            name=name,
            family=rule.family,
            role=rule.role,
            availability=rule.availability,
            missing_policy=rule.missing_policy,
        )

    def assert_model_inputs_available(
        self,
        names: list[str] | tuple[str, ...],
        *,
        at: Availability,
    ) -> None:
        violations: list[str] = []
        for name in names:
            spec = self.resolve(name)
            if spec.role != FeatureRole.INPUT:
                violations.append(f"{name} ({spec.role.value})")
            elif spec.availability > at:
                violations.append(
                    f"{name} (available {spec.availability.name.casefold()})"
                )
        if violations:
            msg = "Unavailable model inputs: " + ", ".join(violations)
            raise FeatureContractError(msg)

    def validate_materialization(
        self,
        names: list[str] | tuple[str, ...],
    ) -> None:
        prohibited = [
            name for name in names if self.resolve(name).role == FeatureRole.PROHIBITED
        ]
        if prohibited:
            msg = "Prohibited configured features: " + ", ".join(sorted(prohibited))
            raise FeatureContractError(msg)

    def manifest(self, names: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                **asdict(self.resolve(name)),
                "role": self.resolve(name).role.value,
                "availability": self.resolve(name).availability.name.casefold(),
            }
            for name in sorted(set(names))
        ]

    def fingerprint(self, names: list[str] | tuple[str, ...]) -> str:
        payload = json.dumps(
            self.manifest(names),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


IDENTIFIERS = {
    "date",
    "gameid",
    "teamid",
    "teamname",
    "playerid",
    "playername",
    "side",
    "position",
}
TARGETS = {"result", "gamelength", "total_kills", "total_towers"}
PROHIBITED = {"first_pick", "side_win_likelihood"}
PREMATCH_EXACT_INPUTS = {
    "league",
    "patch",
    "playoffs",
    "game",
    "season",
    "league_region",
    "league_tier",
    "strength_pool",
    "game_in_series",
    "is_bo1",
    "is_bo3",
    "is_bo5",
    "is_deciding_game",
    "days_since_last_game",
    "is_after_break",
    "is_first_season_game",
    "roster_continuity",
    "roster_uncertainty",
    "rating_uncertainty",
    "h2h_games_before",
    "h2h_wins_before",
    "h2h_win_rate_before",
    "league_elo_win_likelihood",
    "strength_pool_win_likelihood",
    "season_win_likelihood",
    "elo_win_likelihood",
    "glicko2_win_likelihood",
    "pl_win_likelihood",
    "trueskill_win_likelihood",
    "season_avg_gamelength",
    "team_season_avg_gamelength",
}


def default_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for name in IDENTIFIERS:
        registry.register_exact(
            name,
            family="identity",
            role=FeatureRole.IDENTIFIER,
            availability=Availability.PREMATCH,
            missing_policy="block_or_unknown",
        )
    for name in TARGETS:
        registry.register_exact(
            name,
            family="target",
            role=FeatureRole.TARGET,
            availability=Availability.POSTGAME,
            missing_policy="exclude",
        )
    for name in PROHIBITED:
        registry.register_exact(
            name,
            family="unproven_context",
            role=FeatureRole.PROHIBITED,
            availability=Availability.POSTGAME,
            missing_policy="exclude",
        )
    for name in PREMATCH_EXACT_INPUTS:
        registry.register_exact(
            name,
            family="prematch_context",
            role=FeatureRole.INPUT,
            availability=Availability.PREMATCH,
            missing_policy="unknown",
        )
    for suffix, family in (
        ("_before", "historical_aggregate"),
        ("_after", "latest_historical_state"),
    ):
        registry.register_suffix(
            suffix,
            family=family,
            role=FeatureRole.INPUT,
            availability=Availability.PREMATCH,
            missing_policy="unknown",
        )
    return registry
