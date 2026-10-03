"""Tests for daily workflow failure handling and explicit-review inference."""

import datetime as dt

import pytest
from lol_bets.daily import (
    DailyStepResult,
    DailyWorkflowConfig,
    _build_prediction_snapshots,
    build_prediction_snapshot_rows,
    format_step_summary,
    run_daily_lol_workflow,
)
from lol_bets.inference.team import InsufficientRosterHistoryError
from oracle_bets_core.pd import pd

EXPECTED_WIN_PROBABILITY = 0.6
EXPECTED_TOTAL_KILLS = 27.5
EXPECTED_MAP_WIN_PROBABILITY = 0.55


@pytest.fixture(autouse=True)
def _isolate_daily_reports(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(daily_module, "EVIDENCE_DB", tmp_path / "evidence.db")


def _future_schedule() -> pd.DataFrame:
    start = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
    return pd.DataFrame(
        [
            {
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": start.isoformat(),
                "best_of": 3,
                "market_query": "LCK T1 Gen.G",
                "discord_label": "LCK | T1 vs Gen.G | BO3",
            }
        ]
    )


class _FakePredictor:
    outcome_calibrator = object()

    def predict_match(self, team1, *_args, **_kwargs):
        if team1.name == "T1":
            return {"team1_win_probability": 0.6, "team2_win_probability": 0.4}
        return {"team1_win_probability": 0.4, "team2_win_probability": 0.6}

    def predict_gamelength(self, *_args, **_kwargs):
        return 31.2

    def predict_total_kills(self, *_args, **_kwargs):
        return 27.5

    def predict_total_towers(self, *_args, **_kwargs):
        return 13.1


def test_insufficient_roster_history_is_reported_separately(monkeypatch):
    import lol_bets.daily as daily_module

    def fail_for_history(*_args, **_kwargs):
        raise InsufficientRosterHistoryError("missing top")

    monkeypatch.setattr(
        daily_module, "build_match_prediction_snapshot", fail_for_history
    )
    details = _build_prediction_snapshots(
        _future_schedule(),
        cfg=DailyWorkflowConfig(dry_run=True),
        predictor_factory=_FakePredictor,
    )

    assert details[0]["status"] == "insufficient_history"


def test_excluded_league_is_not_predicted_or_reported(monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(
        daily_module,
        "build_match_prediction_snapshot",
        lambda *_args, **_kwargs: None,
    )
    schedule = _future_schedule()
    schedule["league"] = "LCP"

    details = _build_prediction_snapshots(
        schedule,
        cfg=DailyWorkflowConfig(dry_run=True),
    )

    assert details == []


# ── schedule fetch failure handling ─────────────────────────────────────── #


def test_schedule_fetch_failure_yields_failed_step_not_crash(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    # Ensure no stored-schedule fallback exists for this test.
    monkeypatch.setattr(daily_module, "SCHEDULE", tmp_path / "missing.parquet")

    def broken_fetcher(**_kwargs):
        msg = "PandaScore down"
        raise RuntimeError(msg)

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True),
        schedule_fetcher=broken_fetcher,
    )

    assert not result.ok
    schedule_steps = [step for step in result.steps if step.name == "schedule"]
    assert len(schedule_steps) == 1
    assert not schedule_steps[0].ok
    assert "PandaScore down" in schedule_steps[0].detail
    assert result.schedule.empty


def test_reconcile_and_ingest_precede_schedule_fetch(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "SCHEDULE", tmp_path / "missing.parquet")

    class ReadySource:
        current_year_max_match_at = dt.datetime(2026, 8, 9, tzinfo=dt.UTC)

        def raise_if_unready(self):
            pass

        def to_dict(self):
            return {"ready": True}

    monkeypatch.setattr(daily_module, "_daily_source_readiness", ReadySource)
    monkeypatch.setattr(
        daily_module,
        "refresh_oracle_source",
        lambda: type("RefreshResult", (), {"files": (1, 2, 3)})(),
    )
    called = {"ingest": False}

    def broken_fetcher(**_kwargs):
        msg = "boom"
        raise RuntimeError(msg)

    def data_generator_factory():
        called["ingest"] = True
        msg = "stop after proving ingest precedes schedule"
        raise AssertionError(msg)

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=False),
        schedule_fetcher=broken_fetcher,
        data_generator_factory=data_generator_factory,
    )

    assert not result.ok
    assert called["ingest"]


# ── step summary honesty ────────────────────────────────────────────────── #


def test_step_summary_marks_skipped_and_failed_steps():
    steps = [
        DailyStepResult("ingest", ok=False, detail="Google Drive timeout"),
        DailyStepResult("validate-data", ok=True, detail="skipped after failure"),
        DailyStepResult("train", ok=True, detail="skipped after failure"),
    ]

    summary = format_step_summary(steps)

    assert "FAILED: ingest" in summary
    assert "SKIPPED: validate-data" in summary
    assert "SKIPPED: train" in summary
    assert "predictions suppressed" in summary


def test_step_summary_all_ok_has_no_failure_banner():
    steps = [DailyStepResult("schedule", ok=True, detail="3 matches in next 2 days")]

    summary = format_step_summary(steps)

    assert "OK: schedule" in summary
    assert "predictions suppressed" not in summary


# ── prediction snapshots ────────────────────────────────────────────────── #


def _snapshot_rows() -> list[dict]:
    row = _future_schedule().iloc[0]
    return build_prediction_snapshot_rows(
        row,
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.6,
        team_b_win=0.4,
        probability_source="calibrated",
        prop_values={"total_kills": 27.5},
    )


def test_snapshot_rows_cover_both_selections_and_props():
    rows = build_prediction_snapshot_rows(
        _future_schedule().iloc[0],
        team_a_name="T1",
        team_b_name="Gen.G",
        match_type="bo3",
        team_a_win=0.6,
        team_b_win=0.4,
        probability_source="calibrated",
        prop_values={"total_kills": 27.5},
        map_prediction={
            "team1_win_probability": 0.55,
            "team2_win_probability": 0.45,
            "team1_probability_lower": 0.50,
            "team1_probability_upper": 0.60,
            "team2_probability_lower": 0.40,
            "team2_probability_upper": 0.50,
            "uncertainty_method": "held_out",
        },
    )

    winner_rows = [r for r in rows if r["market"] == "series_winner"]
    assert {r["selection"] for r in winner_rows} == {"T1", "Gen.G"}
    t1_row = next(r for r in winner_rows if r["selection"] == "T1")
    assert t1_row["model_value"] == EXPECTED_WIN_PROBABILITY
    map_rows = [r for r in rows if r["market"] == "map_winner"]
    assert {r["selection"] for r in map_rows} == {"T1", "Gen.G"}
    assert (
        next(r for r in map_rows if r["selection"] == "T1")["model_value"]
        == EXPECTED_MAP_WIN_PROBABILITY
    )
    prop_rows = [r for r in rows if r["market"] == "total_kills_mean"]
    assert len(prop_rows) == 1
    assert prop_rows[0]["model_value"] == EXPECTED_TOTAL_KILLS


def test_daily_uses_fresh_clocks_after_maintenance_and_schedule_fetch(monkeypatch):
    import lol_bets.daily as daily_module

    started = dt.datetime(2026, 9, 20, 0, tzinfo=dt.UTC)
    refreshed = started + dt.timedelta(hours=12)
    fetched = refreshed + dt.timedelta(minutes=2)
    current = started
    captured = {}

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return current.astimezone(tz)

    class ReadySource:
        def to_dict(self):
            return {"ready": True}

    def maintain(*_args, **_kwargs):
        nonlocal current
        current = refreshed
        return []

    def fetch(**kwargs):
        nonlocal current
        captured["fetch_at"] = kwargs["start_datetime"]
        current = fetched
        return pd.DataFrame(
            [
                {
                    "match_key": "already-started",
                    "league": "LCK",
                    "team_a": "T1",
                    "team_b": "Gen.G",
                    "start_utc": started + dt.timedelta(hours=6),
                    "best_of": 3,
                },
            ]
        )

    class Refresher:
        def refresh(self, schedule):
            captured["lineup_at"] = current
            schedule["lineup_refresh_error"] = ""
            return schedule

    def record(**kwargs):
        captured["evidence"] = kwargs
        return "run-test"

    monkeypatch.setattr(daily_module.dt, "datetime", Clock)
    monkeypatch.setattr(daily_module, "_run_mutating_steps", maintain)
    monkeypatch.setattr(daily_module, "_daily_source_readiness", ReadySource)
    monkeypatch.setattr(daily_module, "record_daily_evidence", record)
    monkeypatch.setattr(daily_module, "_latest_model_drift_review", lambda: None)
    result = run_daily_lol_workflow(
        DailyWorkflowConfig(delivery_mode="off"),
        schedule_fetcher=fetch,
        lineup_refresher_factory=Refresher,
    )

    assert captured["fetch_at"] == refreshed
    assert captured["lineup_at"] == fetched
    assert captured["evidence"]["scheduled_for"] == started
    assert captured["evidence"]["observed_at"] == fetched
    assert result.schedule.empty
