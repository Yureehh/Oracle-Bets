from __future__ import annotations

import json
from datetime import UTC, datetime

from lol_bets.operations import manual_market
from oracle_bets_core.markets import PolymarketEvent, PolymarketGammaAdapter

NOW = datetime(2026, 8, 21, 8, tzinfo=UTC)
EXPECTED_REPORT_FILES = 2
URL = "https://polymarket.com/esports/league-of-legends/lpl/lol-we-tes-2026-08-22"


def _event() -> PolymarketEvent:
    adapter = PolymarketGammaAdapter()
    market = adapter._market_from_payload(
        {
            "id": "market-1",
            "question": "LoL: Team WE vs Top Esports (BO3) - LPL",
            "slug": "lol-we-tes-2026-08-22",
            "outcomes": '["Team WE", "Top Esports"]',
            "outcomePrices": '["0.40", "0.60"]',
            "clobTokenIds": '["we-token", "tes-token"]',
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "eventStartTime": "2026-08-22T20:00:00Z",
            "sportsMarketType": "moneyline",
            "groupItemTitle": "Match Winner",
            "resolutionSource": "https://gol.gg/esports/home",
        },
        event={
            "id": "event-1",
            "slug": "lol-we-tes-2026-08-22",
            "title": "LoL: Team WE vs Top Esports (BO3) - LPL",
        },
    )
    return PolymarketEvent(
        event_id="event-1",
        title="LoL: Team WE vs Top Esports (BO3) - LPL",
        slug="lol-we-tes-2026-08-22",
        url=URL,
        markets=(market,),
    )


class _Gamma:
    def __init__(self):
        self.calls = []

    def event(self, url):
        self.calls.append(url)
        return _event()


def test_manual_review_uses_exact_event_and_writes_one_report_pair(
    tmp_path, monkeypatch
):
    gamma = _Gamma()
    monkeypatch.setattr(manual_market, "SCHEDULE", tmp_path / "missing.parquet")
    monkeypatch.setattr(manual_market, "daily_position_exposure", lambda *_a, **_k: 0)

    def fake_build(schedule, **kwargs):
        kwargs["snapshot_sink"].extend(
            [
                {"market": "series_winner", "selection": "Team WE"},
                {"market": "series_winner", "selection": "Top Esports"},
            ]
        )
        kwargs["market_action_sink"].append(
            {"state": "no_edge", "selection": "Top Esports"}
        )
        assert tuple(kwargs["typed_markets_override"]) == ("polymarket:event-1",)
        return ["prediction output"], [
            {"status": "predicted", "match_key": schedule.iloc[0]["match_key"]}
        ]

    monkeypatch.setattr(manual_market, "_build_prediction_messages", fake_build)

    result = manual_market.review_polymarket_events(
        [URL, URL], gamma=gamma, report_dir=tmp_path / "reports", now=NOW
    )

    assert result.ok
    assert result.fixtures == result.predictions == 1
    assert gamma.calls == [URL]
    assert len(list((tmp_path / "reports").iterdir())) == EXPECTED_REPORT_FILES
    payload = json.loads(result.report_paths[0].read_text())
    assert payload["fixtures"][0]["league"] == "LPL"
    assert payload["fixtures"][0]["team_a"] == "Team WE"
    assert payload["messages"] == ["prediction output"]
    assert payload["publish_requested"] is False


def test_manual_review_publishes_only_when_explicitly_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(manual_market, "SCHEDULE", tmp_path / "missing.parquet")
    monkeypatch.setattr(manual_market, "daily_position_exposure", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        manual_market,
        "_build_prediction_messages",
        lambda _schedule, **kwargs: (
            kwargs["snapshot_sink"].extend(
                [{"market": "series_winner", "selection": "Team WE"}]
            )
            or ["prediction output"],
            [{"status": "predicted"}],
        ),
    )
    calls = []

    def fake_record(**kwargs):
        calls.append(kwargs)
        return "run-1"

    monkeypatch.setattr(manual_market, "record_daily_evidence", fake_record)

    result = manual_market.review_polymarket_events(
        [URL],
        publish=True,
        gamma=_Gamma(),
        report_dir=tmp_path / "reports",
        now=NOW,
    )

    assert result.evidence_run_id == "run-1"
    assert calls[0]["run_type"] == "manual_lol_market_review"
    assert calls[0]["run_key"].startswith("manual-lol-market-")
