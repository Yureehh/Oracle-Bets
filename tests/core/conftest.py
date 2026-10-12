"""Explicit synthetic capital for pre-existing ledger examples."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from oracle_bets_core.operations.capital import PaperCapital, VenueCapital


@pytest.fixture
def ledger_capital(monkeypatch):
    from oracle_bets_core.operations import bets

    def configure(currency="EUR"):
        venue = VenueCapital(
            Decimal(1000),
            currency,
            Decimal(1),
            Decimal(0),
            "synthetic test conversion",
            datetime(2026, 1, 1, tzinfo=UTC),
            True,
        )
        profile = PaperCapital(
            "test-ledger",
            "test-ledger-hash",
            {"polymarket": venue, "thunderpick": venue},
        )
        monkeypatch.setattr(bets, "load_paper_capital", lambda: profile)
        return profile

    configure()
    return configure


@pytest.fixture(autouse=True)
def explicit_legacy_ledger_capital(request):
    if request.path.name in {"test_bet_evidence.py", "test_quote_confirmation.py"}:
        request.getfixturevalue("ledger_capital")
