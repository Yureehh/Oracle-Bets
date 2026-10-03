import datetime as dt
import json
from types import SimpleNamespace

import pytest
from lol_bets.daily import (
    DailyWorkflowConfig,
    _resolve_fixture_roster,
    _run_mutating_steps,
    cadence_reminders,
    expected_roster_from_schedule,
    filter_daily_schedule,
    match_type_from_best_of,
    run_daily_lol_workflow,
    split_reportable_schedule,
)
from oracle_bets_core.cli import build_parser
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.interfaces import ArtifactHealth
from oracle_bets_core.operations import StepOutcome, WorkflowOutcome
from oracle_bets_core.pd import pd
from oracle_bets_discord.predictions.lol import format_schedule_messages

MESSAGE_SPLIT_LIMIT = 420
EXPECTED_MIN_SPLIT_MESSAGES = 2
EXPECTED_HORIZON_HOURS = 36
DAILY_REPORT_SCHEMA_VERSION = 6
EXPECTED_STABLE_SERIES = 3


@pytest.fixture(autouse=True)
def _isolate_daily_reports(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(daily_module, "EVIDENCE_DB", tmp_path / "evidence.db")


def _schedule_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "start_utc": "2026-07-01T13:00:00Z",
                "best_of": 3,
                "market_query": "LCK T1 Gen.G",
                "discord_label": "LCK | T1 vs Gen.G | BO3",
            },
            {
                "league": "LEC",
                "team_a": "G2 Esports",
                "team_b": "Fnatic",
                "start_utc": "2026-07-02T18:00:00Z",
                "best_of": 5,
                "market_query": "LEC G2 Esports Fnatic",
                "discord_label": "LEC | G2 Esports vs Fnatic | BO5",
            },
            {
                "league": "LPL",
                "team_a": "Bilibili Gaming",
                "team_b": "Top Esports",
                "start_utc": "2026-07-04T10:00:00Z",
                "best_of": 3,
                "market_query": "LPL Bilibili Gaming Top Esports",
                "discord_label": "LPL | Bilibili Gaming vs Top Esports | BO3",
            },
        ]
    )


def test_filter_daily_schedule_keeps_only_the_next_36_hours():
    out = filter_daily_schedule(
        _schedule_frame(),
        now=dt.datetime(2026, 7, 1, 23, 45, tzinfo=dt.UTC),
        horizon_hours=36,
    )

    assert list(out["league"]) == ["LEC"]


def test_schedule_messages_group_by_day_and_split_under_limit():
    messages = format_schedule_messages(
        _schedule_frame().head(2), message_limit=MESSAGE_SPLIT_LIMIT
    )

    assert len(messages) >= EXPECTED_MIN_SPLIT_MESSAGES
    assert all(len(message) < MESSAGE_SPLIT_LIMIT for message in messages)
    assert any("2026-07-01" in message and "LCK" in message for message in messages)
    assert any("2026-07-02" in message and "LEC" in message for message in messages)


def test_reportable_schedule_excludes_empty_teams_and_equal_esports():
    schedule = pd.DataFrame(
        [
            {"league": "Equal eSports Cup", "team_a": "A", "team_b": "B"},
            {"league": "LCK", "team_a": "", "team_b": "T1"},
            {"league": "LCK", "team_a": "TBD", "team_b": "T1"},
            {
                "league": "LCK",
                "team_a": "T1",
                "team_b": "Gen.G",
                "status": "cancelled",
            },
            {"league": "LCK", "team_a": "T1", "team_b": "Gen.G"},
        ]
    )

    visible, excluded = split_reportable_schedule(schedule)

    assert visible[["team_a", "team_b"]].values.tolist() == [["T1", "Gen.G"]]
    assert [item["reason"] for item in excluded] == [
        "excluded league",
        "teams not yet determined",
        "teams not yet determined",
        "fixture is not currently playable",
    ]


def test_match_type_from_best_of_is_conservative():
    assert match_type_from_best_of(3) == "bo3"
    assert match_type_from_best_of("5") == "bo5"
    with pytest.raises(ValueError, match="1, 3, or 5"):
        match_type_from_best_of(2)
    with pytest.raises(ValueError, match="1, 3, or 5"):
        match_type_from_best_of(None)


def test_expected_lineup_is_used_only_when_all_roles_are_unambiguous():
    complete = pd.Series(
        {
            "team_a_lineup_json": (
                '[{"name":"Top","role":"top"},{"name":"Jungle","role":"jun"},'
                '{"name":"Mid","role":"mid"},{"name":"Carry","role":"adc"},'
                '{"name":"Support","role":"support"}]'
            )
        }
    )
    roster, ready = expected_roster_from_schedule(complete, team="a")

    assert ready
    assert roster == {
        "top": "Top",
        "jng": "Jungle",
        "mid": "Mid",
        "bot": "Carry",
        "sup": "Support",
    }

    _, missing_ready = expected_roster_from_schedule(
        pd.Series({"team_a_lineup_json": '[{"name":"Top","role":"top"}]'}),
        team="a",
    )
    assert not missing_ready


def test_empty_provider_lineup_uses_last_ten_map_history(monkeypatch):
    import lol_bets.daily as daily_module

    roles = ("top", "jng", "mid", "bot", "sup")
    rows = [
        {
            "date": dt.datetime(2026, 7, 1 + series, game, tzinfo=dt.UTC),
            "gameid": f"s{series}-g{game}",
            "game": game,
            "teamid": "team-t1",
            "teamname": "T1",
            "playername": f"player-{role}",
            "position": role,
        }
        for series in range(EXPECTED_STABLE_SERIES)
        for game in (1, 2)
        for role in roles
    ]
    monkeypatch.setattr(daily_module, "_roster_history", lambda: pd.DataFrame(rows))
    row = pd.Series(
        {
            "team_a": "T1",
            "team_a_lineup_json": "[]",
            "start_utc": dt.datetime(2026, 7, 5, tzinfo=dt.UTC),
        }
    )

    resolved = _resolve_fixture_roster(row, team="a")

    assert resolved.ready
    assert resolved.source == "historical_last_ten_maps"
    assert resolved.roster == {role: f"player-{role}" for role in roles}
    assert resolved.evidence is not None
    assert len(resolved.evidence.source_map_ids) == EXPECTED_STABLE_SERIES * 2


def test_partial_provider_lineup_never_uses_historical_fallback(monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(
        daily_module,
        "_roster_history",
        lambda: (_ for _ in ()).throw(AssertionError("fallback must not be read")),
    )
    row = pd.Series(
        {
            "team_a": "T1",
            "team_a_lineup_json": '[{"name":"Zeus","role":"top"}]',
            "start_utc": dt.datetime(2026, 7, 5, tzinfo=dt.UTC),
        }
    )

    resolved = _resolve_fixture_roster(row, team="a")

    assert not resolved.ready
    assert resolved.source == "provider_lineup_incomplete"
    assert resolved.evidence is None


def test_lineup_refresh_failure_blocks_historical_fallback(monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(
        daily_module,
        "_roster_history",
        lambda: (_ for _ in ()).throw(AssertionError("fallback must not be read")),
    )
    row = pd.Series(
        {
            "team_a": "T1",
            "team_a_lineup_json": "[]",
            "lineup_refresh_error": "HTTPError: refresh_unavailable",
            "start_utc": dt.datetime(2026, 7, 5, tzinfo=dt.UTC),
        }
    )

    resolved = _resolve_fixture_roster(row, team="a")

    assert not resolved.ready
    assert resolved.source == "provider_refresh_failed:HTTPError: refresh_unavailable"
    assert resolved.evidence is None


class _FakePredictor:
    outcome_calibrator = None

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


class _FakeMarketSearch:
    def search_markets(self, _query: str, **_kwargs):
        return []


def test_daily_dry_run_builds_output_without_persistent_writes(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)
    called = {"ingest": False}

    def schedule_fetcher(**kwargs):
        assert kwargs["save_path"] is None
        schedule = _schedule_frame().head(1).copy()
        schedule["start_utc"] = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        return schedule

    def data_generator_factory():
        called["ingest"] = True
        raise AssertionError("dry run must not ingest")

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True, delivery_mode="off"),
        schedule_fetcher=schedule_fetcher,
        data_generator_factory=data_generator_factory,
    )

    assert result.ok
    assert not any(called.values())
    assert any("Upcoming" in message for message in result.messages)
    assert result.report_paths is not None
    json_path, markdown_path = result.report_paths
    assert json_path.is_file()
    assert markdown_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["config"]["delivery_mode"] == "off"
    assert payload["evidence_run_id"] is None
    assert payload["schema_version"] == DAILY_REPORT_SCHEMA_VERSION
    assert payload["cadence_reminders"] == result.cadence_reminders
    assert payload["open_positions"] == result.open_positions
    assert "source_freshness" in payload
    assert "predictions" not in payload
    assert "drift_review" in payload
    daily_module.write_daily_report(
        cfg=DailyWorkflowConfig(dry_run=True),
        steps=result.steps,
        schedule=result.schedule,
        excluded_fixtures=result.excluded_fixtures,
        messages=result.messages,
        cadence_reminders=result.cadence_reminders,
        open_positions=result.open_positions,
        report_paths=result.report_paths,
    )
    assert list((tmp_path / "daily").glob("*.json")) == [json_path]
    assert list((tmp_path / "daily").glob("*.md")) == [markdown_path]


@pytest.mark.parametrize(
    ("instant", "trigger"),
    [
        (dt.datetime(2026, 8, 3, 10, tzinfo=dt.UTC), "monday"),
        (dt.datetime(2026, 8, 6, 10, tzinfo=dt.UTC), "thursday"),
        (dt.datetime(2026, 9, 1, 10, tzinfo=dt.UTC), "first_of_month"),
    ],
)
def test_daily_cadence_reminders(instant, trigger):
    reminders = cadence_reminders(instant)

    assert len(reminders) == 1
    assert trigger in reminders[0]["triggers"]
    assert reminders[0]["advisory_only"] is True


def test_overlapping_cadence_is_one_combined_reminder():
    reminders = cadence_reminders(dt.datetime(2026, 6, 1, 10, tzinfo=dt.UTC))

    assert len(reminders) == 1
    assert reminders[0]["triggers"] == ["monday", "first_of_month"]
    assert len(reminders[0]["commands"]) == len(set(reminders[0]["commands"]))


def test_cadence_uses_rome_date_at_utc_boundary():
    reminders = cadence_reminders(dt.datetime(2026, 8, 2, 22, 30, tzinfo=dt.UTC))

    assert reminders[0]["local_date"] == "2026-08-03"
    assert reminders[0]["triggers"] == ["monday"]


def test_ordinary_day_has_no_cadence_reminder():
    assert cadence_reminders(dt.datetime(2026, 8, 5, 10, tzinfo=dt.UTC)) == []


def test_daily_defaults_to_operational_leagues_and_hides_academy(
    tmp_path,
    monkeypatch,
):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)
    captured = {}

    def schedule_fetcher(**kwargs):
        captured["leagues"] = kwargs["leagues"]
        start = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        return pd.DataFrame(
            [
                {
                    "league": league,
                    "team_a": "T1",
                    "team_b": "Gen.G",
                    "start_utc": start,
                    "best_of": 3,
                    "status": "not_started",
                }
                for league in (
                    "LCK",
                    "LIT",
                    "LCP",
                    "CBLOL",
                    "LCK Challengers League",
                    "North American Challengers League",
                )
            ]
        )

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True),
        schedule_fetcher=schedule_fetcher,
    )

    assert {"LCK", "LIT"}.issubset(set(captured["leagues"].split(",")))
    assert "LCKC" not in captured["leagues"].split(",")
    assert "NACL" not in captured["leagues"].split(",")
    assert result.schedule["league"].tolist() == ["LCK", "LIT"]


def test_daily_explicit_league_filter_overrides_operational_profile(
    tmp_path,
    monkeypatch,
):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)

    def schedule_fetcher(**_kwargs):
        start = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        return pd.DataFrame(
            [
                {
                    "league": league,
                    "team_a": "T1",
                    "team_b": "Gen.G",
                    "start_utc": start,
                    "best_of": 3,
                    "status": "not_started",
                }
                for league in ("LCK", "LIT")
            ]
        )

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(
            dry_run=True,
            leagues="LCK",
        ),
        schedule_fetcher=schedule_fetcher,
    )

    assert result.schedule["league"].tolist() == ["LCK"]


def test_daily_refreshes_match_detail_before_filtering(monkeypatch):
    import lol_bets.daily as daily_module

    called = {"refresh": False}

    def schedule_fetcher(**_kwargs):
        schedule = _schedule_frame().head(1).copy()
        schedule["start_utc"] = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        schedule["status"] = "cancelled"
        return schedule

    class _Refresher:
        def refresh(self, schedule):
            called["refresh"] = True
            refreshed = schedule.copy()
            refreshed["lineup_refresh_error"] = ""
            refreshed["fixture_version"] = "fixture-refreshed"
            return refreshed

    monkeypatch.delenv("PANDASCORE_API_KEY", raising=False)
    result = daily_module.run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True),
        schedule_fetcher=schedule_fetcher,
        lineup_refresher_factory=_Refresher,
    )

    assert called["refresh"]
    lineup_step = next(step for step in result.steps if step.name == "lineups")
    assert lineup_step.ok
    assert "1 fixture details refreshed" in lineup_step.detail
    assert result.schedule.empty


def test_daily_lol_cli_has_no_training_options():
    args = build_parser().parse_args(["daily", "lol", "--dry-run"])

    assert args.domain == "daily"
    assert args.action == "lol"
    assert args.dry_run is True
    assert not hasattr(args, "skip_retrain")
    assert args.horizon_hours == EXPECTED_HORIZON_HOURS
    assert not hasattr(args, "feature_set")


def test_daily_core_steps_use_the_resumable_evidence_journal(
    tmp_path,
    monkeypatch,
):
    import lol_bets.daily as daily_module

    captured = {}

    def fake_workflow(run_key, steps, *, journal, dry_run):
        captured["run_key"] = run_key
        captured.setdefault("names", []).extend(step.name for step in steps)
        captured["journal"] = journal
        assert dry_run is False
        return WorkflowOutcome(
            run_key=run_key,
            steps=tuple(StepOutcome(step.name, "completed", "ok", 1) for step in steps),
            dry_run=False,
        )

    monkeypatch.setattr(daily_module, "run_workflow", fake_workflow)
    store = EvidenceStore(tmp_path / "daily-journal.db")
    scheduled_for = dt.datetime(2026, 7, 26, tzinfo=dt.UTC)

    result = _run_mutating_steps(
        data_generator_factory=lambda: object(),
        module_factory=lambda: object(),
        store=store,
        scheduled_for=scheduled_for,
        effective_config={"horizon_hours": 36},
    )

    assert captured["run_key"].endswith(":core")
    assert captured["names"] == [
        "source-refresh",
        "source-check",
        "ingest",
        "identity-graph",
        "validate-data",
        "build-series",
        "health",
    ]
    assert captured["journal"].store.path == store.path
    assert all(step.ok for step in result)


def test_daily_source_failure_blocks_ingest_but_still_checks_health(tmp_path):
    called = {"ingest": False}

    def fail_source():
        raise RuntimeError("current_year_file_stale")

    class RefreshResult:
        files = (1, 2, 3)

    def data_generator_factory():
        called["ingest"] = True
        raise AssertionError("stale source must block ingestion")

    def check_health():
        called["health"] = True
        return ArtifactHealth("lol")

    result = _run_mutating_steps(
        data_generator_factory=data_generator_factory,
        module_factory=lambda: SimpleNamespace(
            artifact_health=check_health, training_artifact_health=check_health
        ),
        store=EvidenceStore(tmp_path / "daily-journal.db"),
        scheduled_for=dt.datetime(2026, 8, 10, tzinfo=dt.UTC),
        effective_config={"horizon_hours": 36},
        source_refresh_fn=RefreshResult,
        source_check_fn=fail_source,
    )

    assert not called["ingest"]
    assert called["health"]
    assert result[0].name == "source-refresh"
    assert result[0].ok
    assert result[1].name == "source-check"
    assert not result[1].ok
    assert result[1].detail.endswith("current_year_file_stale")
    assert all(step.detail == "skipped after failure" for step in result[2:-1])
    assert result[-1].name == "health"
    assert result[-1].ok
    assert result[-1].detail == "inference=ok, training=ok"


def test_repeated_daily_maintenance_does_not_reuse_unverified_artifacts(tmp_path):
    calls = []

    class RefreshResult:
        files = ()

    def refresh():
        calls.append("source")
        return RefreshResult()

    def stop_after_source():
        raise RuntimeError("stop before production artifacts")

    store = EvidenceStore(tmp_path / "daily-journal.db")
    for _ in range(2):
        _run_mutating_steps(
            data_generator_factory=lambda: object(),
            module_factory=lambda: SimpleNamespace(
                artifact_health=lambda: ArtifactHealth("lol"),
                training_artifact_health=lambda: ArtifactHealth("lol"),
            ),
            store=store,
            scheduled_for=dt.datetime(2026, 9, 25, tzinfo=dt.UTC),
            effective_config={"horizon_hours": 36},
            source_refresh_fn=refresh,
            source_check_fn=stop_after_source,
        )

    assert calls == ["source", "source"]
