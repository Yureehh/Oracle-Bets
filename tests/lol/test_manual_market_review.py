from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from lol_bets.operations import manual_market
from oracle_bets_core.markets import (
    MarketDataError,
    PolymarketEvent,
    PolymarketGammaAdapter,
)
from oracle_bets_core.pd import pd

NOW = datetime(2026, 8, 21, 8, tzinfo=UTC)
EXPECTED_REPORT_FILES = 2
REPORT_SCHEMA_VERSION = 4
EXPECTED_COMPARISONS = 2
BEST_ODDS = 1.9
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

    def fake_build(schedule, **kwargs):
        kwargs["snapshot_sink"].extend(
            [
                {"market": "series_winner", "selection": "Team WE"},
                {"market": "series_winner", "selection": "Top Esports"},
            ]
        )
        return ["prediction output"], [
            {"status": "predicted", "match_key": schedule.iloc[0]["match_key"]}
        ]

    monkeypatch.setattr(manual_market, "_build_prediction_messages", fake_build)
    monkeypatch.setattr(
        manual_market, "record_daily_evidence", lambda **_kwargs: "run-1"
    )

    result = manual_market.review_polymarket_events(
        [URL, URL], gamma=gamma, report_dir=tmp_path / "reports", now=NOW
    )

    assert result.ok
    assert result.fixtures == result.predictions == 1
    assert gamma.calls == [URL]
    assert len(list((tmp_path / "reports").iterdir())) == EXPECTED_REPORT_FILES
    payload = json.loads(result.report_paths[0].read_text())
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    assert payload["workflow"] == "owner_market_review_v4"
    assert payload["fixture"]["league"] == "LPL"
    assert payload["fixture"]["team_a"] == "Team WE"
    assert payload["evidence_run_id"] == "run-1"
    markdown = result.report_paths[1].read_text()
    assert "## Provider coverage" in markdown
    assert "## Contract inventory" not in markdown


def test_manual_review_persists_evidence_without_queueing_discord(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(manual_market, "SCHEDULE", tmp_path / "missing.parquet")
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
        gamma=_Gamma(),
        report_dir=tmp_path / "reports",
        now=NOW,
    )

    assert result.evidence_run_id == "run-1"
    assert calls[0]["run_type"] == "manual_lol_market_review"
    assert calls[0]["run_key"].startswith("manual-lol-market-")
    assert result.evidence_run_id == "run-1"


def test_manual_discord_summary_includes_series_map_props_and_quote_state():
    schedule = pd.DataFrame([{"team_a": "Team WE", "team_b": "Top Esports"}])
    snapshots = [
        {"market": "series_winner", "selection": "Team WE", "model_value": 0.4},
        {
            "market": "series_winner",
            "selection": "Top Esports",
            "model_value": 0.6,
        },
        {"market": "map_winner", "selection": "Team WE", "model_value": 0.45},
        {"market": "map_winner", "selection": "Top Esports", "model_value": 0.55},
        {"market": "gamelength_mean", "selection": None, "model_value": 31.2},
        {"market": "total_kills_mean", "selection": None, "model_value": 27.5},
    ]
    actions = [
        {
            "fixture_key": "",
            "target": "series_winner",
            "selection": "Top Esports",
            "best_provider": "polymarket",
            "best_decimal_odds": 1.8,
            "model_probability": 0.6,
            "point_ev": 0.08,
        }
    ]

    summary = manual_market.discord_review_summary(schedule, snapshots, actions)

    assert "Series" in summary
    assert "Top Esports 60.0%" in summary
    assert "Map 1 research" in summary
    assert "Team WE 45.0%" in summary
    assert "**Length:** 31.2m" in summary
    assert "**Kills:** 27.5" in summary
    assert "Top Esports" in summary
    assert "Systematic controls" not in summary
    assert "Personal paper" not in summary
    assert "oracle-ref" not in summary


def test_lrn_is_intentionally_ignored_without_provider_or_model_call(
    tmp_path, monkeypatch
):
    gamma = _Gamma()
    monkeypatch.setattr(manual_market, "SCHEDULE", tmp_path / "missing.parquet")
    url = "https://polymarket.com/esports/league-of-legends/lrn/lol-fue-z5-2026-08-24"

    result = manual_market.review_polymarket_events(
        [url], gamma=gamma, report_dir=tmp_path / "reports", now=NOW
    )

    assert result.ok
    assert result.fixtures == result.predictions == 0
    assert gamma.calls == []
    assert result.failures == ()
    assert result.ignored_links[0]["reason"] == "league_not_supported_or_trained"
    assert "no prediction was generated" in result.report_paths[1].read_text().lower()


def test_market_url_parser_allows_two_known_providers_and_rejects_hostile_urls():
    thunderpick = "https://thunderpick.io/en/esports/lol/team-we-vs-top-esports"
    assert manual_market.normalize_market_urls([f"{URL},\n{thunderpick}"]) == (
        URL,
        thunderpick,
    )
    with pytest.raises(MarketDataError, match="HTTPS Polymarket or Thunderpick"):
        manual_market.normalize_market_urls(["https://example.com/steal"])
    with pytest.raises(MarketDataError, match="one or two"):
        manual_market.normalize_market_urls(
            [
                URL,
                "https://polymarket.com/event/second",
                "https://thunderpick.io/en/esports/lol/third",
            ]
        )
    with pytest.raises(MarketDataError, match="at most one Polymarket"):
        manual_market.normalize_market_urls(
            [URL, "https://polymarket.com/event/second"]
        )


def test_provider_comparison_groups_only_identical_semantics_and_picks_best_odds():
    rows = [
        {
            "fixture_key": "fixture-1",
            "provider": "polymarket",
            "target": "series_winner",
            "selection": "Top Esports",
            "line": None,
            "game_number": None,
            "probability": 0.6,
            "decimal_odds": 1.8,
            "point_edge": 0.08,
            "hard_blocks": [],
            "market_id": "poly-series",
        },
        {
            "fixture_key": "fixture-1",
            "provider": "thunderpick",
            "target": "series_winner",
            "selection": "Top Esports",
            "line": None,
            "game_number": None,
            "probability": 0.6,
            "decimal_odds": 1.9,
            "point_edge": 0.14,
            "hard_blocks": [],
            "market_id": "tp-series",
        },
        {
            "fixture_key": "fixture-1",
            "provider": "thunderpick",
            "target": "series_total_maps",
            "selection": "Over",
            "line": 2.5,
            "game_number": None,
            "probability": 0.55,
            "decimal_odds": 1.95,
            "point_edge": 0.0725,
            "hard_blocks": [],
            "market_id": "tp-total",
        },
    ]

    comparisons = manual_market.provider_comparisons(rows)

    assert len(comparisons) == EXPECTED_COMPARISONS
    winner = next(row for row in comparisons if row["target"] == "series_winner")
    assert winner["best_provider"] == "thunderpick"
    assert winner["best_decimal_odds"] == BEST_ODDS
    assert len(winner["providers"]) == EXPECTED_COMPARISONS


def test_thunderpick_is_schedule_resolved_and_never_fetched(tmp_path, monkeypatch):
    schedule_path = tmp_path / "schedule.parquet"
    pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "provider": "pandascore",
                "provider_match_id": "1",
                "league": "LPL",
                "team_a": "Team WE",
                "team_b": "Top Esports",
                "start_utc": "2026-08-22T20:00:00Z",
                "best_of": 3,
                "status": "not_started",
            }
        ]
    ).to_parquet(schedule_path)
    monkeypatch.setattr(manual_market, "SCHEDULE", schedule_path)

    def fake_build(schedule, **kwargs):
        key = schedule.iloc[0]["match_key"]
        kwargs["snapshot_sink"].extend(
            [
                {
                    "source_match_key": key,
                    "market": "series_winner",
                    "selection": "Team WE",
                    "model_value": 0.6,
                },
                {
                    "source_match_key": key,
                    "market": "series_winner",
                    "selection": "Top Esports",
                    "model_value": 0.4,
                },
            ]
        )
        return ["prediction"], [{"status": "predicted"}]

    captured = []
    monkeypatch.setattr(manual_market, "_build_prediction_messages", fake_build)
    monkeypatch.setattr(
        manual_market,
        "record_daily_evidence",
        lambda **kwargs: captured.extend(kwargs["market_actions"]) or "run-1",
    )
    url = "https://thunderpick.io/en/esports/lol/team-we-vs-top-esports"

    result = manual_market.review_polymarket_events(
        [url],
        manual_lines=[
            {
                "target": "series_winner",
                "selection": "Team WE",
                "decimal_odds": 2.0,
            }
        ],
        gamma=_Gamma(),
        report_dir=tmp_path / "reports",
        now=NOW,
    )

    assert result.ok
    assert result.comparisons == 1
    assert captured[0]["provider"] == "thunderpick"
    assert captured[0]["point_edge"] == pytest.approx(0.2)
    assert "Thunderpick" in result.report_paths[1].read_text()


def test_mismatched_thunderpick_fixture_stops_the_review(tmp_path, monkeypatch):
    schedule_path = tmp_path / "schedule.parquet"
    pd.DataFrame(
        [
            {
                "match_key": "pandascore:other",
                "provider": "pandascore",
                "provider_match_id": "other",
                "league": "LPL",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": "2026-08-22T20:00:00Z",
                "best_of": 3,
                "status": "not_started",
            }
        ]
    ).to_parquet(schedule_path)
    monkeypatch.setattr(manual_market, "SCHEDULE", schedule_path)
    monkeypatch.setattr(
        manual_market,
        "_build_prediction_messages",
        lambda *_args, **_kwargs: pytest.fail("mismatched fixture was predicted"),
    )

    result = manual_market.review_polymarket_events(
        [URL, "https://thunderpick.io/en/esports/lol/t1-vs-gen-g"],
        manual_lines=[
            {
                "target": "series_winner",
                "selection": "T1",
                "decimal_odds": 2.0,
            }
        ],
        gamma=_Gamma(),
        report_dir=tmp_path / "reports",
        now=NOW,
    )

    assert result.fixtures == result.predictions == result.comparisons == 0
    assert result.failures[0]["reason"] == "fixture_mismatch"


@pytest.mark.parametrize(
    "value",
    [
        "series_winner | Team WE | nan",
        "series_winner | Team WE | inf",
        "series_total_maps | Over | 1.9 | nan",
    ],
)
def test_thunderpick_manual_lines_reject_non_finite_numbers(value):
    with pytest.raises(MarketDataError, match="finite"):
        manual_market.parse_manual_lines_text(value)
