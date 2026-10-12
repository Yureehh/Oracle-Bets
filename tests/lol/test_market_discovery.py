from datetime import UTC, datetime
from types import SimpleNamespace

from lol_bets.operations.market_discovery import (
    discover_markets,
    record_market_observations,
)
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.pd import pd

EXPECTED_OBSERVATIONS = 2


def schedule():
    return pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "best_of": 3,
                "start_utc": "2026-10-12T12:00:00Z",
                "market_query": "LCK T1 Gen.G",
            }
        ]
    )


def test_listing_absent_distinct_from_provider_failure():
    empty = SimpleNamespace(search_markets=lambda *_args, **_kwargs: [])

    def fail(*_args, **_kwargs):
        raise RuntimeError("unavailable")

    failed = SimpleNamespace(search_markets=fail)
    assert discover_markets(schedule(), adapter=empty)[0]["status"] == "not_seen"
    assert (
        discover_markets(schedule(), adapter=failed)[0]["status"]
        == "provider_unavailable"
    )


def test_observation_timestamp_is_append_only_and_not_opening_time(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "run",
            "run_type": "daily_lol",
            "started_at": datetime(2026, 10, 11, tzinfo=UTC),
            "status": "completed",
            "idempotency_key": "run",
            "payload_json": {},
        },
    )
    first = datetime(2026, 10, 11, 9, tzinfo=UTC)
    second = datetime(2026, 10, 11, 12, tzinfo=UTC)
    rows = [
        {
            "provider": "polymarket",
            "fixture_key": "pandascore:1",
            "status": "needs_exact_review",
            "candidates": [{"market_id": "m", "matched": True, "reasons": []}],
        }
    ]
    record_market_observations(
        store, run_id="run", observations=rows, observed_at=first
    )
    record_market_observations(
        store, run_id="run", observations=rows, observed_at=second
    )
    assert rows[0]["candidates"][0]["first_observed_at"] == first.isoformat()
    assert store.count(EvidenceTable.RUN_EVENTS) == EXPECTED_OBSERVATIONS
    assert rows[0]["opening_time_known"] is False


def test_invalid_fixture_time_never_searches_provider():
    def unexpected(*_args, **_kwargs):
        raise AssertionError("invalid fixture must not reach provider")

    for value in ("2026-10-12T12:00:00", None):
        frame = schedule()
        frame["start_utc"] = value
        result = discover_markets(
            frame, adapter=SimpleNamespace(search_markets=unexpected)
        )
        assert result[0]["status"] == "fixture_unavailable"


def test_persistence_preserves_actual_observation_time(tmp_path):
    import json

    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    at = datetime(2026, 10, 11, 9, tzinfo=UTC)
    store.append(
        EvidenceTable.RUNS,
        {
            "id": "run",
            "run_type": "daily_lol",
            "started_at": at,
            "status": "completed",
            "idempotency_key": "run",
            "payload_json": {},
        },
    )
    rows = [
        {
            "provider": "polymarket",
            "fixture_key": "f",
            "status": "needs_exact_review",
            "observed_at": at.isoformat(),
            "candidates": [{"market_id": "m"}],
        }
    ]
    record_market_observations(
        store,
        run_id="run",
        observations=rows,
        observed_at=datetime(2026, 10, 11, 12, tzinfo=UTC),
    )
    payload = json.loads(store.list(EvidenceTable.RUN_EVENTS)[0]["payload_json"])
    assert payload["observed_at"] == at.isoformat()
    assert payload["candidates"][0]["first_observed_at"] == at.isoformat()
