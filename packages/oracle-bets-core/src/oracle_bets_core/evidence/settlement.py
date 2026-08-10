"""Paper settlement reconciliation and fixed pre-start closing-line value."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import cast


class SettlementError(ValueError):
    """Raised when a paper position cannot be settled unambiguously."""


class SettlementResult(StrEnum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class SettlementSource(StrEnum):
    PROVIDER = "provider"
    VERIFIED_INTERNAL = "verified_internal"
    OWNER_VERIFIED = "owner_verified"


@dataclass(frozen=True)
class PositionTerms:
    """Immutable entry terms required for accounting."""

    position_id: str
    stake_units: Decimal | str | float
    decimal_odds: Decimal | str | float

    def __post_init__(self) -> None:
        if not self.position_id.strip():
            raise SettlementError("position_id cannot be empty")
        stake = _decimal(self.stake_units, field="stake_units")
        odds = _decimal(self.decimal_odds, field="decimal_odds")
        if stake <= 0:
            raise SettlementError("stake_units must be positive")
        if odds <= 1:
            raise SettlementError("decimal_odds must exceed one")
        object.__setattr__(self, "stake_units", stake)
        object.__setattr__(self, "decimal_odds", odds)


@dataclass(frozen=True)
class SettlementRecord:
    """Reconciled append-only accounting fact."""

    settlement_id: str
    position_id: str
    settled_at: datetime
    result: SettlementResult
    source: SettlementSource
    source_reference: str
    pnl_units: Decimal
    warnings: tuple[str, ...]


def reconcile_settlement(
    terms: PositionTerms,
    *,
    settled_at: datetime,
    provider_result: SettlementResult | None,
    provider_reference: str | None = None,
    internal_result: SettlementResult | None = None,
    internal_reference: str | None = None,
    internal_verified: bool = False,
    owner_result: SettlementResult | None = None,
    owner_reference: str | None = None,
) -> SettlementRecord:
    """Build one settlement and stop when verified sources conflict."""
    _require_utc(settled_at, field="settled_at")
    supplied = [
        result
        for result in (provider_result, internal_result, owner_result)
        if result is not None
    ]
    if len(set(supplied)) > 1:
        raise SettlementError("verified settlement results conflict")

    warnings: list[str] = []
    if provider_result is not None:
        result = provider_result
        source = SettlementSource.PROVIDER
        reference = provider_reference
    elif internal_result is not None and internal_verified:
        result = internal_result
        source = SettlementSource.VERIFIED_INTERNAL
        reference = internal_reference
        warnings.append("internal_settlement")
    elif owner_result is not None:
        result = owner_result
        source = SettlementSource.OWNER_VERIFIED
        reference = owner_reference
    else:
        raise SettlementError(
            "settlement requires a provider, verified internal, or owner result"
        )
    if not reference or not reference.strip():
        raise SettlementError("settlement source reference cannot be empty")

    pnl = settlement_pnl(terms, result)
    identity = (
        f"{terms.position_id}|{settled_at.isoformat()}|{result.value}|"
        f"{source.value}|{reference}"
    )
    return SettlementRecord(
        settlement_id=(
            f"settlement-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        ),
        position_id=terms.position_id,
        settled_at=settled_at,
        result=result,
        source=source,
        source_reference=reference,
        pnl_units=pnl,
        warnings=tuple(warnings),
    )


def settlement_pnl(
    terms: PositionTerms,
    result: SettlementResult,
) -> Decimal:
    """Return profit/loss excluding returned stake."""
    stake = cast("Decimal", terms.stake_units)
    odds = cast("Decimal", terms.decimal_odds)
    if result is SettlementResult.WIN:
        return stake * (odds - 1)
    if result is SettlementResult.LOSS:
        return -stake
    return Decimal(0)


@dataclass(frozen=True)
class MarketCloseSnapshot:
    """Exact executable pre-start fill retained for closing-line analysis."""

    snapshot_id: str
    observed_at: datetime
    decimal_odds: Decimal | str | float
    available_stake_units: Decimal | str | float
    source: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or not self.source.strip():
            raise ValueError("closing snapshot identifiers cannot be empty")
        _require_utc(self.observed_at, field="observed_at")
        odds = _decimal(self.decimal_odds, field="decimal_odds")
        available = _decimal(
            self.available_stake_units,
            field="available_stake_units",
        )
        if odds <= 1:
            raise ValueError("closing decimal_odds must exceed one")
        if available < 0:
            raise ValueError("closing available stake cannot be negative")
        object.__setattr__(self, "decimal_odds", odds)
        object.__setattr__(self, "available_stake_units", available)


def select_fixed_close(
    snapshots: tuple[MarketCloseSnapshot, ...] | list[MarketCloseSnapshot],
    *,
    event_start: datetime,
    intended_stake_units: Decimal | str | float,
) -> MarketCloseSnapshot | None:
    """Select the last pre-start snapshot able to fill the intended stake."""
    _require_utc(event_start, field="event_start")
    intended = _decimal(intended_stake_units, field="intended_stake_units")
    if intended <= 0:
        raise ValueError("intended_stake_units must be positive")
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.observed_at < event_start
        and cast("Decimal", snapshot.available_stake_units) >= intended
    ]
    return max(eligible, key=lambda item: item.observed_at) if eligible else None


@dataclass(frozen=True)
class ClosingLineValue:
    """Entry-versus-close comparison with explicit unavailable state."""

    available: bool
    snapshot_id: str | None
    probability_clv: float | None
    odds_ratio_clv: float | None
    reason: str | None
    warnings: tuple[str, ...]


def calculate_clv(
    *,
    entry_decimal_odds: Decimal | str | float,
    close: MarketCloseSnapshot | None,
) -> ClosingLineValue:
    """Positive values mean the entry price beat the executable close."""
    entry = _decimal(entry_decimal_odds, field="entry_decimal_odds")
    if entry <= 1:
        raise ValueError("entry_decimal_odds must exceed one")
    if close is None:
        return ClosingLineValue(
            available=False,
            snapshot_id=None,
            probability_clv=None,
            odds_ratio_clv=None,
            reason="closing_snapshot_unavailable",
            warnings=(),
        )
    close_odds = cast("Decimal", close.decimal_odds)
    probability_clv = float(Decimal(1) / close_odds - Decimal(1) / entry)
    odds_ratio_clv = float(entry / close_odds - 1)
    return ClosingLineValue(
        available=True,
        snapshot_id=close.snapshot_id,
        probability_clv=probability_clv,
        odds_ratio_clv=odds_ratio_clv,
        reason=None,
        warnings=close.warnings,
    )


def _require_utc(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")


def _decimal(value: Decimal | str | float, *, field: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return parsed
