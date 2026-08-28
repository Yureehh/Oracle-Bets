"""Shared settlement accounting for the owner-managed bet ledger."""

from __future__ import annotations

from dataclasses import dataclass
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


def _decimal(value: Decimal | str | float, *, field: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return parsed
