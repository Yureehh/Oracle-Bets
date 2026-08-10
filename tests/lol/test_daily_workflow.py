import datetime as dt
import json

import pytest
from lol_bets.daily import (
    DailyStepResult,
    DailyWorkflowConfig,
    _deliver_daily_webhook,
    _resolve_fixture_roster,
    _run_mutating_steps,
    cadence_reminders,
    expected_roster_from_schedule,
    filter_daily_schedule,
    format_market_candidates,
    match_type_from_best_of,
    run_daily_lol_workflow,
    select_market_candidates,
    send_discord_webhook_messages,
    split_reportable_schedule,
)
from oracle_bets_core.cli import build_parser
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.markets import MarketQuote
from oracle_bets_core.operations import StepOutcome, WorkflowOutcome
from oracle_bets_core.pd import pd
from oracle_bets_discord.predictions.lol import format_schedule_messages

MESSAGE_SPLIT_LIMIT = 420
EXPECTED_MIN_SPLIT_MESSAGES = 2
EXPECTED_HORIZON_HOURS = 36
DAILY_REPORT_SCHEMA_VERSION = 4
EXPECTED_STABLE_SERIES = 3


@pytest.fixture(autouse=True)
def _isolate_daily_reports(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)


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
        "fixture is not currently playable",
    ]


def test_match_type_from_best_of_is_conservative():
    assert match_type_from_best_of(3) == "bo3"
    assert match_type_from_best_of("5") == "bo5"
    assert match_type_from_best_of(7) == "bo1"
    assert match_type_from_best_of(None) == "bo1"


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


def test_empty_provider_lineup_uses_exact_three_series_history(monkeypatch):
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
    assert resolved.source == "historical_three_series"
    assert resolved.roster == {role: f"player-{role}" for role in roles}
    assert resolved.evidence is not None
    assert len(resolved.evidence.series_ids) == EXPECTED_STABLE_SERIES


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


def test_market_candidate_selection_requires_matchup_context():
    quotes = [
        MarketQuote(
            source="polymarket",
            market_id="1",
            question="Will T1 beat Gen.G in League of Legends?",
            outcome="T1",
            implied_probability=0.44,
            liquidity=1000,
        ),
        MarketQuote(
            source="polymarket",
            market_id="2",
            question="Will G2 beat Fnatic in League of Legends?",
            outcome="G2",
            implied_probability=0.51,
            liquidity=2000,
        ),
    ]

    out = select_market_candidates(quotes, team_a="T1", team_b="Gen.G")

    assert [quote.market_id for quote in out] == ["1"]


def test_market_candidate_formatting_shows_model_edge():
    output = format_market_candidates(
        team_a="T1",
        team_b="Gen.G",
        team_a_probability=0.55,
        team_b_probability=0.45,
        quotes=[
            MarketQuote(
                source="polymarket",
                market_id="1",
                question="Will T1 beat Gen.G in League of Legends?",
                outcome="T1",
                implied_probability=0.44,
                liquidity=1000,
                url="https://polymarket.com/event/t1-geng",
            )
        ],
    )

    assert "Polymarket candidates" in output
    assert "T1" in output
    assert "+25.0%" in output
    assert "https://polymarket.com/event/t1-geng" in output


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
    def search(self, _query: str, **_kwargs):
        return [
            MarketQuote(
                source="polymarket",
                market_id="1",
                question="Will T1 beat Gen.G in League of Legends?",
                outcome="T1",
                implied_probability=0.50,
            )
        ]


def test_daily_dry_run_builds_output_without_persistent_writes(tmp_path, monkeypatch):
    import lol_bets.daily as daily_module

    monkeypatch.setattr(daily_module, "REPORTS_DIR", tmp_path)
    called = {"ingest": False, "train": False, "send": False}

    def schedule_fetcher(**kwargs):
        assert kwargs["save_path"] is None
        schedule = _schedule_frame().head(1).copy()
        schedule["start_utc"] = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        return schedule

    def data_generator_factory():
        called["ingest"] = True
        raise AssertionError("dry run must not ingest")

    def train_fn(**_kwargs):
        called["train"] = True
        raise AssertionError("dry run must not train")

    def webhook_sender(_url, _messages):
        called["send"] = True
        raise AssertionError("dry run must not send")

    result = run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True, webhook_url="https://discord.test"),
        schedule_fetcher=schedule_fetcher,
        data_generator_factory=data_generator_factory,
        train_fn=train_fn,
        predictor_factory=_FakePredictor,
        market_search_factory=_FakeMarketSearch,
        webhook_sender=webhook_sender,
    )

    assert result.ok
    assert not any(called.values())
    assert any("T1 vs Gen.G" in message for message in result.messages)
    assert any("Polymarket candidates" in message for message in result.messages)
    assert result.report_paths is not None
    json_path, markdown_path = result.report_paths
    assert json_path.is_file()
    assert markdown_path.is_file()
    payload = json.loads(json_path.read_text())
    assert "webhook_url" not in payload["config"]
    assert payload["evidence_run_id"] is None
    assert payload["schema_version"] == DAILY_REPORT_SCHEMA_VERSION
    assert payload["market_reviews"] == result.market_reviews
    assert payload["market_actions"] == result.market_actions
    assert payload["cadence_reminders"] == result.cadence_reminders
    assert payload["open_positions"] == result.open_positions
    assert "source_freshness" in payload
    assert "roster_evidence" in payload
    assert "drift_review" in payload
    daily_module.write_daily_report(
        cfg=DailyWorkflowConfig(dry_run=True),
        steps=result.steps,
        schedule=result.schedule,
        excluded_fixtures=result.excluded_fixtures,
        prediction_details=result.prediction_details,
        messages=result.messages,
        market_reviews=result.market_reviews,
        market_actions=result.market_actions,
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
        DailyWorkflowConfig(dry_run=True, skip_market_search=True),
        schedule_fetcher=schedule_fetcher,
        predictor_factory=_FakePredictor,
        market_search_factory=_FakeMarketSearch,
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
            skip_market_search=True,
            leagues="LCK",
        ),
        schedule_fetcher=schedule_fetcher,
        predictor_factory=_FakePredictor,
        market_search_factory=_FakeMarketSearch,
    )

    assert result.schedule["league"].tolist() == ["LCK"]


def test_discord_webhook_disables_all_mentions():
    captured = {}

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

    class _Session:
        @staticmethod
        def post(url, *, json, timeout):
            captured.update(url=url, json=json, timeout=timeout)
            return _Response()

    send_discord_webhook_messages(
        "https://discord.test",
        ["provider text @everyone"],
        session=_Session(),
    )

    assert captured["json"]["allowed_mentions"] == {"parse": []}


def test_discord_webhook_failure_redacts_secret_url(caplog):
    webhook_url = "https://discord.com/api/webhooks/id/secret-token"

    def fail(url, _messages):
        raise RuntimeError(f"failed request to {url}")

    result = _deliver_daily_webhook(fail, webhook_url, ["report"])

    assert result == DailyStepResult("discord", False, "delivery failed (RuntimeError)")
    assert webhook_url not in caplog.text
    assert "secret-token" not in result.detail


def test_daily_refreshes_match_detail_before_filtering(monkeypatch):
    import lol_bets.daily as daily_module

    called = {"refresh": False}

    def schedule_fetcher(**_kwargs):
        schedule = _schedule_frame().head(1).copy()
        schedule["start_utc"] = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
        schedule["status"] = "cancelled"
        return schedule

    class _Refresher:
        def refresh(self, schedule, *, observed_at):
            called["refresh"] = observed_at.tzinfo is not None
            refreshed = schedule.copy()
            refreshed["lineup_refresh_error"] = ""
            refreshed["fixture_version"] = "fixture-refreshed"
            return refreshed

    monkeypatch.delenv("PANDASCORE_API_KEY", raising=False)
    result = daily_module.run_daily_lol_workflow(
        DailyWorkflowConfig(dry_run=True, skip_market_search=True),
        schedule_fetcher=schedule_fetcher,
        lineup_refresher_factory=_Refresher,
        predictor_factory=_FakePredictor,
        market_search_factory=_FakeMarketSearch,
    )

    assert called["refresh"]
    lineup_step = next(step for step in result.steps if step.name == "lineups")
    assert lineup_step.ok
    assert "1 fixture details refreshed" in lineup_step.detail
    assert result.schedule.empty


def test_daily_lol_cli_defaults_to_no_retune():
    args = build_parser().parse_args(["daily", "lol", "--dry-run"])

    assert args.domain == "daily"
    assert args.action == "lol"
    assert args.dry_run is True
    assert args.skip_retrain is False
    assert args.horizon_hours == EXPECTED_HORIZON_HOURS
    assert args.feature_set == "compact"


def test_daily_core_steps_use_the_resumable_evidence_journal(
    tmp_path,
    monkeypatch,
):
    import lol_bets.daily as daily_module

    captured = {}

    def fake_workflow(run_key, steps, *, journal, dry_run):
        captured["run_key"] = run_key
        captured["names"] = [step.name for step in steps]
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
        DailyWorkflowConfig(skip_retrain=True),
        data_generator_factory=lambda: object(),
        train_fn=lambda **_kwargs: None,
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
        "train",
        "health",
    ]
    assert captured["journal"].store.path == store.path
    assert all(step.ok for step in result)


def test_daily_source_failure_blocks_ingest_and_training(tmp_path):
    called = {"ingest": False, "train": False}

    def fail_source():
        raise RuntimeError("current_year_file_stale")

    class RefreshResult:
        files = (1, 2, 3)

    def data_generator_factory():
        called["ingest"] = True
        raise AssertionError("stale source must block ingestion")

    def train_fn(**_kwargs):
        called["train"] = True
        raise AssertionError("stale source must block training")

    result = _run_mutating_steps(
        DailyWorkflowConfig(),
        data_generator_factory=data_generator_factory,
        train_fn=train_fn,
        module_factory=lambda: object(),
        store=EvidenceStore(tmp_path / "daily-journal.db"),
        scheduled_for=dt.datetime(2026, 8, 10, tzinfo=dt.UTC),
        effective_config={"horizon_hours": 36},
        source_refresh_fn=RefreshResult,
        source_check_fn=fail_source,
    )

    assert not called["ingest"]
    assert not called["train"]
    assert result[0].name == "source-refresh"
    assert result[0].ok
    assert result[1].name == "source-check"
    assert not result[1].ok
    assert result[1].detail.endswith("current_year_file_stale")
    assert all(step.detail == "skipped after failure" for step in result[2:])
