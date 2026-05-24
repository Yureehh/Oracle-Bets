import pytest
from oracle_bets_dashboard import database


@pytest.fixture
def ledger_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "ledger.db")
    database.init_db()
    return database


def _bet_payload(**overrides):
    payload = {
        "event_date": "2026-05-24",
        "sport": "League of Legends",
        "league": "LPL",
        "event_name": "Team WE vs LNG Esports",
        "market_type": "kills",
        "selection": "Over",
        "odds_decimal": 1.85,
        "stake": 25.0,
        "line": 26.5,
        "over_under": "Over",
        "bookmaker": "polymarket",
        "bet_slip_ref": "ticket-1",
    }
    payload.update(overrides)
    return payload


def test_ledger_insert_settle_and_duplicate_ref(ledger_db):
    bet_id = ledger_db.insert_bet(_bet_payload())

    bet = ledger_db.get_bet(bet_id)
    assert bet["implied_probability"] == pytest.approx(1 / 1.85)
    assert bet["potential_payout"] == pytest.approx(46.25)

    settled = ledger_db.settle_bet(bet_id, "win")
    assert settled["status"] == "settled"
    assert settled["pnl"] == pytest.approx(21.25)

    with pytest.raises(ValueError, match="Duplicate bet"):
        ledger_db.insert_bet(_bet_payload())


def test_ledger_rejects_invalid_probability(ledger_db):
    with pytest.raises(ValueError, match="model_probability"):
        ledger_db.insert_bet(
            _bet_payload(bet_slip_ref="ticket-2", model_probability=1.2)
        )
