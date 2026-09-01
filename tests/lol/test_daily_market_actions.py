from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from lol_bets.daily import build_prediction_snapshot_rows
from lol_bets.operations import market_actions
from lol_bets.operations.market_actions import evaluate_daily_market_actions
from oracle_bets_core.markets import OrderBook, PolymarketGammaAdapter
from oracle_bets_core.pd import pd

START = datetime(2026, 8, 2, 12, tzinfo=UTC)
FIXTURE_START = START + timedelta(hours=36)
EXPECTED_OUTCOMES = 2


def _market(adapter, market_id, market_type, title, outcomes, tokens, *, line=None):
    raw = {
        "id": market_id,
        "question": f"T1 vs Gen.G: {title}",
        "slug": market_id,
        "outcomes": outcomes,
        "outcomePrices": '["0.50", "0.50"]',
        "clobTokenIds": tokens,
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "eventStartTime": FIXTURE_START.isoformat(),
        "bestOf": 3,
        "sportsMarketType": market_type,
        "groupItemTitle": title,
        "resolutionSource": "https://liquipedia.net/leagueoflegends/Main_Page",
        "liquidity": "1000",
        "line": line,
    }
    return adapter._market_from_payload(
        raw,
        event={"id": "event-1", "title": "LCK T1 vs Gen.G"},
    )


class _Books:
    def __init__(self):
        self.calls = 0

    def get_order_book(self, token_id):
        self.calls += 1
        return OrderBook.from_payload(
            {
                "market": "condition-1",
                "asset_id": token_id,
                "timestamp": str(int(START.timestamp() * 1000)),
                "hash": f"book-{self.calls}",
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.50", "size": "100"}],
                "min_order_size": "5",
                "tick_size": "0.01",
                "neg_risk": False,
            },
            expected_token_id=token_id,
        )


def test_exact_event_markets_use_one_snapshot_and_rank_correlated_contracts():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": FIXTURE_START,
                "best_of": 3,
            }
        ]
    )
    snapshots = build_prediction_snapshot_rows(
        schedule.iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.70,
        team_b_win=0.30,
        team_a_lower=0.65,
        team_a_upper=0.75,
        team_b_lower=0.25,
        team_b_upper=0.35,
        probability_source="calibrated",
        uncertainty_method="held_out",
        uncertainty_confidence=0.90,
        uncertainty_sample_count=100,
        drivers=["Team rating strength pushed the model toward T1"],
        rating_baseline_team_a=0.60,
        full_model_team_a=0.72,
        prop_values={},
        map_prediction={
            "team1_win_probability": 0.65,
            "team2_win_probability": 0.35,
            "team1_probability_lower": 0.60,
            "team2_probability_lower": 0.30,
            "team1_probability_upper": 0.70,
            "team2_probability_upper": 0.40,
            "uncertainty_method": "held_out",
        },
        lineup_ready=True,
        roster_ready=True,
    )
    adapter = PolymarketGammaAdapter()
    markets = [
        _market(
            adapter,
            "game-1",
            "child_moneyline",
            "Game 1 Winner",
            '["T1","Gen.G"]',
            '["g1-t1","g1-geng"]',
        ),
        _market(
            adapter,
            "match",
            "moneyline",
            "Match Winner",
            '["T1","Gen.G"]',
            '["m-t1","m-geng"]',
        ),
        _market(
            adapter,
            "totals",
            "totals",
            "Total Maps 2.5",
            '["Over","Under"]',
            '["over","under"]',
            line="2.5",
        ),
    ]
    books = _Books()
    result = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=snapshots,
        markets={"pandascore:1": markets},
        clob_client=books,
        model_healthy=True,
        run_key="daily-lol-2026-08-02",
        clock=lambda: START,
    )

    quoted = [row for row in result.actions if row.get("decimal_odds")]
    assert books.calls == len({row["token_id"] for row in quoted})
    assert result.reviews
    assert all(row["state"] == "exploration" for row in quoted)
    assert all(row["bet_type"] == "exploration" for row in quoted)
    assert all(Decimal(str(row["decimal_odds"])) > 1 for row in quoted)
    assert all(row["requested_shares"] == "5" for row in quoted)
    assert all(row["hypothetical_cost"] == "2.50" for row in quoted)
    assert all(
        observation["minimum_order_size"] == "5"
        for row in quoted
        for observation in row["observations"]
    )
    assert all(row["correlation_rank"] is None for row in quoted)

    rerun_snapshots = [
        dict(row, run_ts=START + timedelta(hours=1)) for row in snapshots
    ]
    rerun = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=rerun_snapshots,
        markets={"pandascore:1": markets},
        clob_client=_Books(),
        model_healthy=True,
        run_key="daily-lol-2026-08-02",
        clock=lambda: START,
    )
    assert {row["comparison_id"] for row in rerun.actions} == {
        row["comparison_id"] for row in result.actions
    }


def test_scalar_prop_is_priced_and_recorded_but_never_systematic(monkeypatch):
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": FIXTURE_START,
                "best_of": 3,
            }
        ]
    )
    snapshots = build_prediction_snapshot_rows(
        schedule.iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.7,
        team_b_win=0.3,
        team_a_lower=0.65,
        team_a_upper=0.75,
        team_b_lower=0.25,
        team_b_upper=0.35,
        probability_source="calibrated",
        uncertainty_method="held_out",
        uncertainty_confidence=0.9,
        uncertainty_sample_count=100,
        drivers=[],
        prop_values={"total_kills": 25.0},
        map_prediction=None,
        lineup_ready=True,
        roster_ready=True,
    )

    class Calibrator:
        def price(self, **_kwargs):
            return SimpleNamespace(over_probability=0.6, under_probability=0.4)

    monkeypatch.setattr(
        market_actions, "_prop_calibrator", lambda _target: Calibrator()
    )
    adapter = PolymarketGammaAdapter()
    market = _market(
        adapter,
        "kills",
        "kill_over_under_game",
        "Game 1 Total Kills 24.5",
        '["Over","Under"]',
        '["kills-over","kills-under"]',
        line="24.5",
    )
    generic_totals_market = _market(
        adapter,
        "kills-generic-totals",
        "totals",
        "Game 1 Total Kills 24.5",
        '["Over","Under"]',
        '["kills-generic-over","kills-generic-under"]',
        line="24.5",
    )

    result = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=snapshots,
        markets={"pandascore:1": [market, generic_totals_market]},
        clob_client=_Books(),
        model_healthy=True,
        clock=lambda: START,
    )

    assert {row["probability"] for row in result.actions} == {0.4, 0.6}
    assert all(row["decimal_odds"] for row in result.actions)
    assert all(row["state"] == "exploration" for row in result.actions)
    assert all("display_only_prop" in row["warnings"] for row in result.actions)


def test_prop_calibrator_resolves_the_serving_champion(monkeypatch, tmp_path):
    resolved = tmp_path / "champion-prop.pkl"
    loaded = object()
    calls = []
    monkeypatch.setattr(
        market_actions,
        "resolve_serving_artifact",
        lambda path, **_kwargs: calls.append(path) or resolved,
    )
    monkeypatch.setattr(
        market_actions,
        "load_model",
        lambda path: loaded if path == resolved else None,
    )
    market_actions._prop_calibrator.cache_clear()

    assert market_actions._prop_calibrator("total_kills_mean") is loaded
    assert calls == [market_actions.TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR]
    market_actions._prop_calibrator.cache_clear()


def test_handicap_is_derived_while_odd_even_kills_has_no_model_target():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": FIXTURE_START,
                "best_of": 3,
            }
        ]
    )
    snapshots = build_prediction_snapshot_rows(
        schedule.iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.7,
        team_b_win=0.3,
        probability_source="calibrated",
        prop_values={},
        map_prediction={
            "team1_win_probability": 0.65,
            "team2_win_probability": 0.35,
        },
        lineup_ready=True,
        roster_ready=True,
    )
    adapter = PolymarketGammaAdapter()
    markets = [
        _market(
            adapter,
            "handicap",
            "map_handicap",
            "Game Handicap: T1 (-1.5) vs Gen.G (+1.5)",
            '["T1","Gen.G"]',
            '["handicap-t1","handicap-geng"]',
            line="-1.5",
        ),
        _market(
            adapter,
            "odd-even-kills",
            "lol_odd_even_total_kills",
            "Odd/Even Total Kills",
            '["Odd","Even"]',
            '["odd","even"]',
        ),
    ]

    result = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=snapshots,
        markets={"pandascore:1": markets},
        clob_client=_Books(),
        model_healthy=True,
        clock=lambda: START,
    )

    handicap = [row for row in result.actions if row["market_id"] == "handicap"]
    unsupported = [
        row for row in result.actions if row["market_id"] == "odd-even-kills"
    ]
    assert {row["target"] for row in handicap} == {"series_handicap"}
    assert all(row["probability"] is not None for row in handicap)
    assert len(unsupported) == 1
    assert all(row["reason"] == "no_model_target" for row in unsupported)
    assert all(row["decimal_odds"] is None for row in unsupported)
    unsupported_review = next(
        row for row in result.reviews if row["market_id"] == "odd-even-kills"
    )
    assert unsupported_review["target"] == "unknown"
    assert "no_model_target" in unsupported_review["warnings"]


def test_manual_integer_total_is_not_priced_without_push_accounting():
    fixture = pd.Series(
        {
            "match_key": "pandascore:1",
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "start_utc": FIXTURE_START,
            "best_of": 3,
        }
    )
    rows = [
        {
            "source_match_key": "pandascore:1",
            "market": "map_winner",
            "selection": "T1",
            "model_value": 0.6,
            "roster_ready": True,
        }
    ]

    action = market_actions.price_manual_lines(
        fixture=fixture,
        snapshot_rows=rows,
        lines=[
            {
                "target": "series_total_maps",
                "selection": "Over",
                "line": 3.0,
                "decimal_odds": 2.0,
            }
        ],
        reviewed_at=START,
    )[0]

    assert action["probability"] is None
    assert "total_push_probability_not_supported" in action["warnings"]
    assert any(
        reason.startswith("semantic_contract:") for reason in action["reason_codes"]
    )


@pytest.mark.parametrize(
    ("target", "selection", "line", "expected_point", "expected_lower"),
    [
        ("series_total_maps", "Over", 2.5, 0.48, 0.42),
        ("series_handicap", "T1", -1.5, 0.36, 0.3025),
    ],
)
def test_manual_derived_markets_preserve_map_uncertainty(
    target, selection, line, expected_point, expected_lower
):
    fixture = pd.Series(
        {
            "match_key": "pandascore:1",
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "start_utc": FIXTURE_START,
            "best_of": 3,
        }
    )
    rows = [
        {
            "source_match_key": "pandascore:1",
            "market": "map_winner",
            "selection": "T1",
            "model_value": 0.6,
            "probability_lower": 0.55,
        },
        {
            "source_match_key": "pandascore:1",
            "market": "map_winner",
            "selection": "Gen.G",
            "model_value": 0.4,
            "probability_lower": 0.3,
        },
    ]

    action = market_actions.price_manual_lines(
        fixture=fixture,
        snapshot_rows=rows,
        lines=[
            {
                "target": target,
                "selection": selection,
                "line": line,
                "decimal_odds": 2.0,
            }
        ],
        reviewed_at=START,
    )[0]

    assert action["probability"] == pytest.approx(expected_point)
    assert action["probability_lower"] == pytest.approx(expected_lower)
    assert action["conservative_edge"] < action["point_edge"]


def test_ready_low_confidence_roster_is_warned():
    schedule = pd.DataFrame(
        [
            {
                "match_key": "pandascore:1",
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": FIXTURE_START,
                "best_of": 3,
            }
        ]
    )
    snapshots = build_prediction_snapshot_rows(
        schedule.iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.6,
        team_b_win=0.4,
        team_a_lower=0.55,
        team_a_upper=0.65,
        team_b_lower=0.35,
        team_b_upper=0.45,
        probability_source="calibrated",
        uncertainty_method="held_out",
        uncertainty_confidence=0.9,
        uncertainty_sample_count=100,
        drivers=[],
        rating_baseline_team_a=0.55,
        full_model_team_a=0.6,
        prop_values={},
        map_prediction=None,
        lineup_ready=True,
        roster_ready=True,
    )
    for row in snapshots:
        row["team_a_roster_confidence"] = "low"
        row["team_b_roster_confidence"] = "high"
    adapter = PolymarketGammaAdapter()
    market = _market(
        adapter,
        "match-low-roster",
        "moneyline",
        "Match Winner",
        '["T1","Gen.G"]',
        '["m-t1","m-geng"]',
    )

    result = evaluate_daily_market_actions(
        schedule=schedule,
        snapshot_rows=snapshots,
        markets=[market],
        clob_client=_Books(),
        model_healthy=True,
        clock=lambda: START,
    )

    assert all(
        "roster_confidence_low_or_unknown" in row["warnings"] for row in result.actions
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_programmatic_manual_lines_reject_non_finite_odds(value):
    fixture = pd.Series(
        {
            "match_key": "pandascore:1",
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "start_utc": FIXTURE_START,
            "best_of": 3,
        }
    )

    with pytest.raises(ValueError, match="finite odds"):
        market_actions.price_manual_lines(
            fixture=fixture,
            snapshot_rows=[],
            lines=[
                {
                    "target": "series_winner",
                    "selection": "T1",
                    "decimal_odds": value,
                }
            ],
            reviewed_at=START,
        )
