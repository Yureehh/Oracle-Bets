"""Validated domain records shared across sports and delivery clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from math import isfinite
from typing import Any


class ContractValidationError(ValueError):
    """Raised when a domain record contains an impossible or unsafe value."""


class DecisionMode(StrEnum):
    """Evidence cohorts that must never be blended silently."""

    PREMATCH = "prematch"


class WarningCode(StrEnum):
    """Non-blocking evidence-quality warnings."""

    ROSTER_UNCERTAIN = "roster_uncertain"
    MARKET_DEPTH_LOW = "market_depth_low"
    MARKET_RULES_UNCERTAIN = "market_rules_uncertain"
    MARKET_FEES_UNKNOWN = "market_fees_unknown"
    MARKET_SLIPPAGE_HIGH = "market_slippage_high"
    CORRELATED_EXPOSURE = "correlated_exposure"
    DAILY_CAP_DISABLED = "daily_cap_disabled"
    INTERNAL_SETTLEMENT = "internal_settlement"


class RejectionReason(StrEnum):
    """Stable no-bet and blocked-decision reasons."""

    IDENTITY_AMBIGUOUS = "identity_ambiguous"
    FIXTURE_UNSUPPORTED = "fixture_unsupported"
    LEAGUE_NOT_ACTIONABLE = "league_not_actionable"
    ROSTER_UNKNOWN = "roster_unknown"
    ROSTER_UNSTABLE = "roster_unstable"
    MARKET_NOT_FOUND = "market_not_found"
    MARKET_MISMATCH = "market_mismatch"
    ODDS_BELOW_MINIMUM = "odds_below_minimum"
    INSUFFICIENT_CONSERVATIVE_EDGE = "insufficient_conservative_edge"
    PRICE_CHANGED = "price_changed"
    MODEL_UNHEALTHY = "model_unhealthy"
    INSUFFICIENT_BANKROLL = "insufficient_bankroll"
    RISK_CAP_EXHAUSTED = "risk_cap_exhausted"


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string."
        raise ContractValidationError(msg)
    return value


def _require_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        msg = f"{field_name} must be timezone-aware UTC."
        raise ContractValidationError(msg)
    if value.utcoffset() != UTC.utcoffset(value):
        msg = f"{field_name} must be stored in UTC."
        raise ContractValidationError(msg)
    return value


def _parse_utc(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (AttributeError, TypeError, ValueError) as exc:
        msg = f"{field_name} must be an ISO-8601 UTC timestamp."
        raise ContractValidationError(msg) from exc
    return _require_utc(parsed, field_name)


def _utc_json(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _decimal(value: Decimal | str | float, field_name: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        msg = f"{field_name} must be a finite decimal."
        raise ContractValidationError(msg) from exc
    if not parsed.is_finite():
        msg = f"{field_name} must be a finite decimal."
        raise ContractValidationError(msg)
    return parsed


@dataclass(frozen=True)
class ProbabilityEstimate:
    """A calibrated point probability and its conservative uncertainty range."""

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        values = (self.point, self.lower, self.upper)
        if not all(
            isinstance(value, (int, float)) and isfinite(value) for value in values
        ):
            msg = "Probability values must be finite numbers."
            raise ContractValidationError(msg)
        if not all(0.0 <= value <= 1.0 for value in values):
            msg = "Probability values must be in [0, 1]."
            raise ContractValidationError(msg)
        if not self.lower <= self.point <= self.upper:
            msg = "Probability range must satisfy lower <= point <= upper."
            raise ContractValidationError(msg)

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProbabilityEstimate:
        try:
            return cls(
                point=value["point"],
                lower=value["lower"],
                upper=value["upper"],
            )
        except (KeyError, TypeError) as exc:
            msg = "Probability estimate requires point, lower, and upper."
            raise ContractValidationError(msg) from exc


@dataclass(frozen=True)
class DecimalOdds:
    """Finite decimal odds greater than one."""

    value: Decimal | str | float

    def __post_init__(self) -> None:
        parsed = _decimal(self.value, "Decimal odds")
        if parsed <= 1:
            msg = "Decimal odds must be greater than 1."
            raise ContractValidationError(msg)
        object.__setattr__(self, "value", parsed)

    def to_json_value(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class StakeUnits:
    """A non-negative bankroll-unit amount represented exactly."""

    value: Decimal | str | float

    def __post_init__(self) -> None:
        parsed = _decimal(self.value, "Stake units")
        if parsed < 0:
            msg = "Stake units must be non-negative."
            raise ContractValidationError(msg)
        object.__setattr__(self, "value", parsed)

    def to_json_value(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class FixtureRef:
    """Canonical future fixture facts shared by predictors and market adapters."""

    fixture_id: str
    sport: str
    competition_id: str
    team_a_id: str
    team_b_id: str
    start_time: datetime
    best_of: int | None

    def __post_init__(self) -> None:
        for field_name in (
            "fixture_id",
            "sport",
            "competition_id",
            "team_a_id",
            "team_b_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.team_a_id == self.team_b_id:
            msg = "Fixture teams must be distinct."
            raise ContractValidationError(msg)
        _require_utc(self.start_time, "start_time")
        if self.best_of not in {None, 1, 2, 3, 5}:
            msg = "best_of must be one of 1, 2, 3, 5, or unknown."
            raise ContractValidationError(msg)


@dataclass(frozen=True)
class PredictionRecord:
    """Immutable probability record produced by a versioned model."""

    prediction_id: str
    fixture_id: str
    selection_id: str
    model_version: str
    created_at: datetime
    mode: DecisionMode
    probability: ProbabilityEstimate
    warnings: tuple[WarningCode, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "prediction_id",
            "fixture_id",
            "selection_id",
            "model_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_utc(self.created_at, "created_at")
        if not isinstance(self.mode, DecisionMode):
            msg = "mode must be a DecisionMode."
            raise ContractValidationError(msg)
        if not isinstance(self.probability, ProbabilityEstimate):
            msg = "probability must be a ProbabilityEstimate."
            raise ContractValidationError(msg)
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(warning, WarningCode) for warning in self.warnings
        ):
            msg = "warnings must be a tuple of WarningCode values."
            raise ContractValidationError(msg)
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            msg = "schema_version must be a positive integer."
            raise ContractValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "fixture_id": self.fixture_id,
            "selection_id": self.selection_id,
            "model_version": self.model_version,
            "created_at": _utc_json(self.created_at),
            "mode": self.mode.value,
            "probability": self.probability.to_dict(),
            "warnings": [warning.value for warning in self.warnings],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredictionRecord:
        try:
            return cls(
                prediction_id=value["prediction_id"],
                fixture_id=value["fixture_id"],
                selection_id=value["selection_id"],
                model_version=value["model_version"],
                created_at=_parse_utc(value["created_at"], "created_at"),
                mode=DecisionMode(value["mode"]),
                probability=ProbabilityEstimate.from_dict(value["probability"]),
                warnings=tuple(WarningCode(item) for item in value.get("warnings", [])),
                schema_version=value.get("schema_version", 1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ContractValidationError):
                raise
            msg = "Invalid prediction record payload."
            raise ContractValidationError(msg) from exc
