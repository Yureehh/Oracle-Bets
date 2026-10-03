"""Daily League of Legends prediction workflow orchestration."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol
from zoneinfo import ZoneInfo

from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.io_utils import atomic_write_text
from oracle_bets_core.league_selection import actionable_leagues
from oracle_bets_core.logger import logger
from oracle_bets_core.operations import (
    EvidenceWorkflowJournal,
    WorkflowStep,
    daily_run_key,
    run_workflow,
)
from oracle_bets_core.paths import (
    EVIDENCE_DB,
    INTERIM_PLAYER_DATA,
    MODEL_REGISTRY_DIR,
    PROCESSED_DIR,
    RAW_CURRENT_POINTER,
    RAW_DATA,
    REPORTS_DIR,
    SCHEDULE,
)
from oracle_bets_core.pd import pd
from oracle_bets_discord.delivery import (
    DiscordDeliveryMode,
    resolve_delivery_mode,
)
from oracle_bets_discord.predictions.lol import (
    format_schedule_messages,
    get_empty_roster,
)

from lol_bets.data_generation.ingestion.history import current_history_data_path
from lol_bets.data_generation.ingestion.schedule import (
    PandaScoreLineupRefresher,
    PandaScoreSchedule,
    fetch_and_store_schedule,
    resolved_team_name,
)
from lol_bets.data_generation.ingestion.source import (
    inspect_oracle_source,
    refresh_oracle_source,
)
from lol_bets.data_generation.series import build_series_artifacts
from lol_bets.inference.roster import (
    EXPECTED_STARTERS,
    HistoricalRosterEvidence,
    RosterGateDecision,
    RosterGateEvidence,
    completed_series_with_roster,
    evaluate_roster_gate,
    infer_historical_roster,
)
from lol_bets.inference.snapshots import load_feature_snapshot
from lol_bets.inference.team import InsufficientRosterHistoryError, Team
from lol_bets.inference.team_resolver import (
    TeamResolutionError,
    resolve_team_name,
)
from lol_bets.module import LoLBetsModule
from lol_bets.operations.evidence import record_daily_evidence
from lol_bets.operations.identity import sync_history_identity_graph
from lol_bets.pipeline import DataGenerator
from lol_bets.training import validate_training_tables

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from oracle_bets_core.interfaces import ArtifactHealth


class Predictor(Protocol):
    outcome_calibrator: Any
    series_winner_calibrator: Any

    def predict_match(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool = True,
        match_type: str | None = None,
    ) -> dict[str, Any]: ...

    def predict_map(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool = False,
        match_type: str | None = None,
    ) -> dict[str, Any]: ...

    def predict_gamelength(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float: ...

    def predict_total_kills(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float: ...

    def predict_total_towers(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> float: ...


@dataclass(frozen=True)
class DailyWorkflowConfig:
    """Runtime options for the daily LoL workflow."""

    horizon_hours: int = 36
    leagues: str | None = None
    delivery_mode: str | None = None
    dry_run: bool = False


def _resolve_daily_config(
    config: DailyWorkflowConfig | None,
) -> DailyWorkflowConfig:
    cfg = config or DailyWorkflowConfig()
    return replace(
        cfg,
        leagues=cfg.leagues or ",".join(actionable_leagues()),
        delivery_mode=resolve_delivery_mode(cfg.delivery_mode).value,
    )


EXCLUDED_DAILY_LEAGUES = frozenset({"Equal eSports Cup"})
MONDAY = 0
THURSDAY = 3
RESTRICTED_FIXTURE_STATUSES = frozenset(
    {"abandoned", "canceled", "cancelled", "postponed", "rescheduled"}
)
_PANDASCORE_ROLES = {
    "top": "top",
    "jun": "jng",
    "jungle": "jng",
    "jungler": "jng",
    "jng": "jng",
    "mid": "mid",
    "middle": "mid",
    "adc": "bot",
    "bot": "bot",
    "support": "sup",
    "sup": "sup",
}


@dataclass(frozen=True)
class DailyStepResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ResolvedFixtureRoster:
    roster: dict[str, str | None]
    ready: bool
    source: str
    evidence: HistoricalRosterEvidence | None = None


@dataclass(frozen=True)
class DailyWorkflowResult:
    schedule: pd.DataFrame
    messages: list[str]
    steps: list[DailyStepResult] = field(default_factory=list)
    excluded_fixtures: list[dict[str, Any]] = field(default_factory=list)
    cadence_reminders: list[dict[str, Any]] = field(default_factory=list)
    open_positions: dict[str, Any] = field(default_factory=dict)
    report_paths: tuple[Path, Path] | None = None

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)


def utc_day_window(
    *,
    now: dt.datetime | None = None,
    window_days: int = 2,
) -> tuple[dt.datetime, dt.datetime]:
    """Return the UTC midnight window used for daily reports."""
    current = now or dt.datetime.now(dt.UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.UTC)
    start = current.astimezone(dt.UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start, start + dt.timedelta(days=window_days)


def _daily_time_window(
    horizon_hours: int,
) -> tuple[dt.datetime, dt.datetime, int]:
    if horizon_hours <= 0:
        raise ValueError("fixture horizon must be positive")
    run_now = dt.datetime.now(dt.UTC)
    scheduled_for, _ = utc_day_window(now=run_now, window_days=1)
    fetch_window_days = (horizon_hours + 23) // 24
    return run_now, scheduled_for, fetch_window_days


def cadence_reminders(
    now: dt.datetime,
    *,
    timezone: str = "Europe/Rome",
) -> list[dict[str, Any]]:
    """Return one combined advisory reminder for today's Rome-local cadence."""
    if now.tzinfo is None:
        raise ValueError("reminder clock must include a timezone")
    local_date = now.astimezone(ZoneInfo(timezone)).date()

    triggers: list[str] = []
    commands: list[str] = []
    if local_date.weekday() == MONDAY:
        triggers.append("monday")
        commands.extend(
            [
                "uv run oracle-bets bet list --state open",
                "uv run oracle-bets health system",
                "uv run oracle-bets bet performance --mode paper",
            ]
        )
    if local_date.weekday() == THURSDAY:
        triggers.append("thursday")
        commands.extend(
            [
                "uv run oracle-bets bet list --state open",
                "uv run oracle-bets bet performance --mode paper",
                "uv run oracle-bets lol market-check",
            ]
        )
    if local_date.day == 1:
        current_month = local_date.replace(day=1)
        previous_month_end = current_month - dt.timedelta(days=1)
        previous_month = previous_month_end.replace(day=1)
        triggers.append("first_of_month")
        commands.extend(
            [
                (
                    "uv run oracle-bets audit monthly --period "
                    f"{previous_month.strftime('%Y-%m')}"
                ),
                "uv run oracle-bets bet performance --mode paper",
            ]
        )
    if not triggers:
        return []
    return [
        {
            "local_date": local_date.isoformat(),
            "timezone": timezone,
            "triggers": triggers,
            "commands": list(dict.fromkeys(commands)),
            "advisory_only": True,
        }
    ]


def format_cadence_reminders(reminders: Sequence[dict[str, Any]]) -> str:
    if not reminders:
        return ""
    reminder = reminders[0]
    labels = ", ".join(str(item) for item in reminder["triggers"])
    lines = [f"**Operating reminder — {labels}**"]
    lines.extend(f"- `{command}`" for command in reminder["commands"])
    return "\n".join(lines)


def _open_position_summary(store: EvidenceStore) -> dict[str, Any]:
    """Read open tracked bets without ever mutating or settling them."""
    from oracle_bets_core.operations.bets import count_open_bets

    try:
        count = count_open_bets(store)
    except Exception as exc:
        logger.warning("Open bet count unavailable: %s", exc)
        return {"available": False, "count": None}
    return {"available": True, "count": count}


def _format_open_position_summary(summary: dict[str, Any]) -> str:
    if not summary.get("available"):
        return "**Open bets:** unavailable; inspect with `bet list`."
    return (
        f"**Open bets: {summary['count']}** — settle only through the "
        "owner bot controls or `uv run oracle-bets bet settle --help`."
    )


def filter_daily_schedule(
    schedule_df: pd.DataFrame,
    *,
    now: dt.datetime | None = None,
    horizon_hours: int = 36,
) -> pd.DataFrame:
    """Keep fixtures in the exact rolling UTC horizon."""
    if horizon_hours <= 0:
        raise ValueError("fixture horizon must be positive")
    if schedule_df.empty:
        return schedule_df.copy()
    start = now or dt.datetime.now(dt.UTC)
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.UTC)
    start = start.astimezone(dt.UTC)
    end = start + dt.timedelta(hours=horizon_hours)
    out = schedule_df.copy()
    out["start_utc"] = pd.to_datetime(out["start_utc"], errors="coerce", utc=True)
    mask = (out["start_utc"] >= start) & (out["start_utc"] < end)
    return (
        out.loc[mask]
        .sort_values(["start_utc", "league", "team_a"])
        .reset_index(drop=True)
    )


def split_reportable_schedule(
    schedule_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Separate printable fixtures from intentionally excluded schedule rows."""
    if schedule_df.empty:
        return schedule_df.copy(), []
    visible_rows: list[Any] = []
    excluded: list[dict[str, Any]] = []
    for index, row in schedule_df.iterrows():
        team_a = resolved_team_name(row.get("team_a"))
        team_b = resolved_team_name(row.get("team_b"))
        league = str(row.get("league") or "").strip()
        reason = None
        if league in EXCLUDED_DAILY_LEAGUES:
            reason = "excluded league"
        elif str(row.get("status") or "").strip().casefold() in (
            RESTRICTED_FIXTURE_STATUSES
        ):
            reason = "fixture is not currently playable"
        elif not team_a or not team_b:
            reason = "teams not yet determined"
        if reason is None:
            visible_rows.append(index)
        else:
            excluded.append(
                {
                    "match_key": row.get("match_key"),
                    "league": league,
                    "team_a": team_a or None,
                    "team_b": team_b or None,
                    "start_utc": row.get("start_utc"),
                    "reason": reason,
                }
            )
    return schedule_df.loc[visible_rows].reset_index(drop=True), excluded


def _actionable_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty or "league" not in schedule:
        return schedule
    return schedule[schedule["league"].isin(actionable_leagues())]


def expected_roster_from_schedule(
    row: pd.Series,
    *,
    team: str,
) -> tuple[dict[str, str | None], bool]:
    """Parse one provider lineup; incomplete or ambiguous input stays unknown."""
    if team not in {"a", "b"}:
        raise ValueError("team must be 'a' or 'b'")
    raw = row.get(f"team_{team}_lineup_json")
    try:
        players = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return get_empty_roster(), False
    if not isinstance(players, list):
        return get_empty_roster(), False
    roster = get_empty_roster()
    ambiguous = False
    for player in players:
        if not isinstance(player, dict):
            continue
        role = _PANDASCORE_ROLES.get(str(player.get("role") or "").casefold())
        name = str(player.get("name") or "").strip()
        if not role or not name:
            continue
        if roster[role] not in {None, name}:
            ambiguous = True
        roster[role] = name
    complete = not ambiguous and all(roster.values())
    return roster, complete


def _resolve_fixture_roster(
    row: pd.Series,
    *,
    team: str,
) -> ResolvedFixtureRoster:
    """Prefer a complete provider five; use history only when it is absent."""
    raw_refresh_error = row.get("lineup_refresh_error")
    refresh_error = (
        ""
        if raw_refresh_error is None or pd.isna(raw_refresh_error)
        else str(raw_refresh_error).strip()
    )
    if refresh_error:
        return ResolvedFixtureRoster(
            get_empty_roster(), False, f"provider_refresh_failed:{refresh_error}"
        )
    provider_roster, provider_ready = expected_roster_from_schedule(row, team=team)
    if provider_ready:
        return ResolvedFixtureRoster(
            provider_roster,
            True,
            str(row.get("lineup_source") or "pandascore_match_detail"),
        )
    if not _provider_lineup_absent(row.get(f"team_{team}_lineup_json")):
        return ResolvedFixtureRoster(
            get_empty_roster(), False, "provider_lineup_incomplete"
        )

    team_name = str(row.get(f"team_{team}") or "").strip()
    fixture_start = row.get("start_utc")
    history = _roster_history()
    known_names = history["teamname"].dropna().astype(str).unique().tolist()
    resolution = resolve_team_name(team_name, known_names)
    if not resolution.ok or resolution.resolved_name is None:
        return ResolvedFixtureRoster(
            get_empty_roster(), False, "historical_roster_unavailable"
        )
    canonical_name = resolution.resolved_name
    dates = pd.to_datetime(history["date"], errors="coerce")
    cutoff = pd.Timestamp(fixture_start)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    matching = history.loc[
        history["teamname"].astype(str).str.casefold().eq(canonical_name.casefold())
        & dates.lt(cutoff)
    ].copy()
    matching["__date"] = dates.loc[matching.index]
    team_ids = (
        matching.sort_values("__date", ascending=False)["teamid"].dropna().astype(str)
    )
    team_id = next((value for value in team_ids if value.strip()), "")
    evidence = infer_historical_roster(
        history,
        team_id=team_id,
        team_name=canonical_name,
        before=fixture_start,
    )
    if evidence is None:
        return ResolvedFixtureRoster(
            get_empty_roster(), False, "historical_roster_unavailable"
        )
    return ResolvedFixtureRoster(
        dict(evidence.roster),
        True,
        "historical_last_ten_maps",
        evidence,
    )


def _provider_lineup_absent(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, list) and not parsed


@lru_cache(maxsize=2)
def _roster_history_version(mtime_ns: int, size: int) -> pd.DataFrame:
    _ = mtime_ns, size
    return pd.read_parquet(
        INTERIM_PLAYER_DATA,
        columns=[
            "date",
            "gameid",
            "game",
            "teamid",
            "teamname",
            "playername",
            "position",
        ],
    )


def _roster_history() -> pd.DataFrame:
    stat = INTERIM_PLAYER_DATA.stat()
    return _roster_history_version(stat.st_mtime_ns, stat.st_size)


def _evaluate_team_roster_gate(
    team: Team,
    *,
    lineup_ready: bool,
    fixture_start: Any,
    emergency_substitute: bool = False,
) -> RosterGateDecision:
    expected_names = (
        tuple(team.player_stats["playername"].dropna().astype(str))
        if lineup_ready and team.player_stats is not None
        else ()
    )
    established = tuple(team.established_roster.values())
    team_id_value = team.team_stats.get("teamid") if team.team_stats is not None else ""
    team_id = str(team_id_value) if pd.notna(team_id_value) else team.name
    completed = 0
    expected_key = tuple(sorted(name.casefold() for name in expected_names))
    established_key = tuple(sorted(name.casefold() for name in established))
    if len(expected_names) == EXPECTED_STARTERS and expected_key != established_key:
        completed = completed_series_with_roster(
            _roster_history(),
            team_id=team_id,
            team_name=team.name,
            expected_player_names=expected_names,
            before=fixture_start,
        )
    return evaluate_roster_gate(
        RosterGateEvidence(
            team_id=team_id,
            expected_player_ids=expected_names,
            established_player_ids=established or None,
            completed_series_with_expected_roster=completed,
            emergency_substitute=emergency_substitute,
        )
    )


def match_type_from_best_of(best_of: Any) -> str:
    try:
        value = int(best_of)
    except (TypeError, ValueError) as error:
        raise ValueError("best_of must be one of 1, 3, or 5") from error
    if value in {1, 3, 5}:
        return f"bo{value}"
    raise ValueError("best_of must be one of 1, 3, or 5")


def build_match_prediction_snapshot(  # noqa: PLR0915
    row: pd.Series,
    *,
    predictor: Predictor,
    snapshot_sink: list[dict[str, Any]] | None = None,
    resolution_sink: list[dict[str, str]] | None = None,
) -> None:
    """Run prematch inference and append its structured snapshot rows."""
    decision_at = dt.datetime.now(dt.UTC)
    feature_snapshot = load_feature_snapshot(
        PROCESSED_DIR / "serving", decision_at=decision_at
    )
    team_a_name = str(row.get("team_a") or "").strip()
    team_b_name = str(row.get("team_b") or "").strip()
    match_type = match_type_from_best_of(row.get("best_of"))
    row = row.copy()
    team_a_resolution = _resolve_fixture_roster(row, team="a")
    team_b_resolution = _resolve_fixture_roster(row, team="b")
    team_a_roster = team_a_resolution.roster
    team_b_roster = team_b_resolution.roster
    team_a_lineup_ready = team_a_resolution.ready
    team_b_lineup_ready = team_b_resolution.ready
    sources = {team_a_resolution.source, team_b_resolution.source}
    row["lineup_source"] = (
        next(iter(sources)) if len(sources) == 1 else "mixed_lineup_sources"
    )
    row["team_a_roster_evidence"] = (
        team_a_resolution.evidence.to_dict() if team_a_resolution.evidence else None
    )
    row["team_b_roster_evidence"] = (
        team_b_resolution.evidence.to_dict() if team_b_resolution.evidence else None
    )
    lineup_ready = team_a_lineup_ready and team_b_lineup_ready
    team_a = Team(
        name=team_a_name,
        side="Blue",
        first_pick=None,
        as_of_date=row.get("start_utc"),
        decision_at=decision_at,
        feature_snapshot=feature_snapshot,
        roster=team_a_roster,
    )
    team_b = Team(
        name=team_b_name,
        side="Red",
        first_pick=None,
        as_of_date=row.get("start_utc"),
        decision_at=decision_at,
        feature_snapshot=feature_snapshot,
        roster=team_b_roster,
    )
    if resolution_sink is not None:
        resolution_sink.extend(
            {"provider_name": supplied, "canonical_name": team.name}
            for supplied, team in ((team_a_name, team_a), (team_b_name, team_b))
            if supplied != team.name
        )
    team_a_roster_gate = _evaluate_team_roster_gate(
        team_a,
        lineup_ready=team_a_lineup_ready,
        fixture_start=row.get("start_utc"),
        emergency_substitute=bool(row.get("team_a_emergency_substitute", False)),
    )
    team_b_roster_gate = _evaluate_team_roster_gate(
        team_b,
        lineup_ready=team_b_lineup_ready,
        fixture_start=row.get("start_utc"),
        emergency_substitute=bool(row.get("team_b_emergency_substitute", False)),
    )
    roster_ready = team_a_roster_gate.actionable and team_b_roster_gate.actionable
    roster_confidences = [
        resolution.evidence.confidence
        if resolution.evidence
        else ("high" if resolution.ready else "unknown")
        for resolution in (team_a_resolution, team_b_resolution)
    ]
    row["team_a_roster_confidence"] = roster_confidences[0]
    row["team_b_roster_confidence"] = roster_confidences[1]
    prediction = predictor.predict_match(
        team_a,
        team_b,
        account_for_side=False,
        match_type=match_type,
    )
    team_a_win = prediction["team1_win_probability"]
    team_b_win = prediction["team2_win_probability"]
    team_a_lower = float(prediction.get("team1_probability_lower", team_a_win))
    team_a_upper = float(prediction.get("team1_probability_upper", team_a_win))
    team_b_lower = float(prediction.get("team2_probability_lower", team_b_win))
    team_b_upper = float(prediction.get("team2_probability_upper", team_b_win))
    uncertainty_method = prediction.get("uncertainty_method")
    uncertainty_confidence = prediction.get("uncertainty_confidence")
    uncertainty_sample_count = prediction.get("uncertainty_sample_count")
    rating_baseline_a = prediction.get("rating_baseline_probability")
    full_model_a = prediction.get("full_model_probability")
    drivers = [str(value) for value in prediction.get("drivers", [])]
    attribution_stable = prediction.get("attribution_stable") is True
    calibrator_method = getattr(
        getattr(predictor, "series_winner_calibrator", None), "method", None
    )
    probability_source = (
        f"calibrated model ({calibrator_method})"
        if calibrator_method and calibrator_method != "raw"
        else "raw model (selected by calibration)"
        if calibrator_method == "raw"
        else "raw model"
    )

    prop_values: dict[str, float] = {}
    for _label, method_name, source_name in (
        ("Length", "predict_gamelength", "gamelength"),
        ("Kills", "predict_total_kills", "total_kills"),
        ("Towers", "predict_total_towers", "total_towers"),
    ):
        try:
            value = getattr(predictor, method_name)(
                team_a, team_b, account_for_side=False
            )
        except Exception as exc:
            logger.debug("Shadow prop %s unavailable: %s", source_name, exc)
            continue
        prop_values[source_name] = float(value)

    map_prediction = None
    predict_map = getattr(predictor, "predict_map", None)
    if callable(predict_map):
        try:
            map_prediction = predict_map(
                team_a,
                team_b,
                account_for_side=False,
                match_type=match_type,
            )
        except Exception as exc:
            logger.debug("Shadow Map 1 prediction unavailable: %s", exc)

    if snapshot_sink is not None:
        snapshot_sink.extend(
            build_prediction_snapshot_rows(
                row,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                match_type=match_type,
                decision_at=decision_at,
                feature_snapshot_id=feature_snapshot.snapshot_id,
                serving_model_id=getattr(predictor, "serving_model_id", None),
                raw_model_team_a=prediction.get("raw_model_probability"),
                team_a_win=team_a_win,
                team_b_win=team_b_win,
                team_a_lower=team_a_lower,
                team_a_upper=team_a_upper,
                team_b_lower=team_b_lower,
                team_b_upper=team_b_upper,
                probability_source=probability_source,
                uncertainty_method=uncertainty_method,
                uncertainty_confidence=uncertainty_confidence,
                uncertainty_sample_count=uncertainty_sample_count,
                drivers=drivers,
                attribution_stable=attribution_stable,
                driver_attribution=prediction.get("driver_attribution", []),
                rating_baseline_team_a=(
                    float(rating_baseline_a) if rating_baseline_a is not None else None
                ),
                full_model_team_a=(
                    float(full_model_a) if full_model_a is not None else None
                ),
                blend_weight=(
                    float(prediction["blend_weight"])
                    if prediction.get("blend_weight") is not None
                    else None
                ),
                team_a_roster_gate=team_a_roster_gate,
                team_b_roster_gate=team_b_roster_gate,
                prop_values=prop_values,
                map_prediction=map_prediction,
                lineup_ready=lineup_ready,
                roster_ready=roster_ready,
            )
        )


def build_prediction_snapshot_rows(
    row: pd.Series,
    *,
    team_a_name: str,
    team_b_name: str,
    match_type: str,
    team_a_win: float,
    team_b_win: float,
    probability_source: str,
    prop_values: dict[str, float],
    decision_at: dt.datetime | None = None,
    feature_snapshot_id: str | None = None,
    serving_model_id: str | None = None,
    raw_model_team_a: float | None = None,
    map_prediction: dict[str, Any] | None = None,
    team_a_lower: float | None = None,
    team_a_upper: float | None = None,
    team_b_lower: float | None = None,
    team_b_upper: float | None = None,
    uncertainty_method: str | None = None,
    uncertainty_confidence: float | None = None,
    uncertainty_sample_count: int | None = None,
    drivers: Sequence[str] = (),
    attribution_stable: bool | None = None,
    driver_attribution: Sequence[dict[str, Any]] = (),
    rating_baseline_team_a: float | None = None,
    full_model_team_a: float | None = None,
    blend_weight: float | None = None,
    team_a_roster_gate: RosterGateDecision | None = None,
    team_b_roster_gate: RosterGateDecision | None = None,
    lineup_ready: bool = False,
    roster_ready: bool = False,
) -> list[dict[str, Any]]:
    """One snapshot row per (match, market, selection) for later CLV/backtests."""
    run_ts = dt.datetime.now(dt.UTC)
    paired_distribution = {
        "method": "direct_series_winner_v2",
        "best_of": int(match_type.removeprefix("bo")),
        "series": {
            team_a_name: {
                "point": team_a_win,
                "lower": team_a_lower,
                "upper": team_a_upper,
            },
            team_b_name: {
                "point": team_b_win,
                "lower": team_b_lower,
                "upper": team_b_upper,
            },
        },
    }
    base = {
        "run_ts": run_ts,
        "decision_at": decision_at.isoformat() if decision_at else None,
        "feature_snapshot_id": feature_snapshot_id,
        "serving_model_id": serving_model_id,
        "run_date": run_ts.date().isoformat(),
        "team_a": team_a_name,
        "team_b": team_b_name,
        "league": row.get("league"),
        "start_utc": row.get("start_utc"),
        "source_match_key": row.get("match_key"),
        "fixture_version": row.get("fixture_version"),
        "fixture_status": row.get("status"),
        "lineup_source": row.get("lineup_source"),
        "lineup_observed_at": row.get("lineup_observed_at"),
        "lineup_refresh_error": row.get("lineup_refresh_error"),
        "team_a_roster_evidence": row.get("team_a_roster_evidence"),
        "team_b_roster_evidence": row.get("team_b_roster_evidence"),
        "team_a_roster_confidence": row.get("team_a_roster_confidence"),
        "team_b_roster_confidence": row.get("team_b_roster_confidence"),
        "match_type": match_type,
        "probability_source": probability_source,
        "uncertainty_method": uncertainty_method,
        "uncertainty_confidence": uncertainty_confidence,
        "uncertainty_sample_count": uncertainty_sample_count,
        "drivers": list(drivers),
        "paired_distribution": paired_distribution,
        "team_a_roster_state": (
            team_a_roster_gate.state if team_a_roster_gate is not None else None
        ),
        "team_b_roster_state": (
            team_b_roster_gate.state if team_b_roster_gate is not None else None
        ),
        "team_a_roster_version": (
            team_a_roster_gate.roster_version
            if team_a_roster_gate is not None
            else None
        ),
        "team_b_roster_version": (
            team_b_roster_gate.roster_version
            if team_b_roster_gate is not None
            else None
        ),
        "team_a_completed_roster_series": (
            team_a_roster_gate.completed_series
            if team_a_roster_gate is not None
            else None
        ),
        "team_b_completed_roster_series": (
            team_b_roster_gate.completed_series
            if team_b_roster_gate is not None
            else None
        ),
        "lineup_ready": lineup_ready,
        "roster_ready": roster_ready,
    }
    rows: list[dict[str, Any]] = []
    for selection, probability in (
        (team_a_name, team_a_win),
        (team_b_name, team_b_win),
    ):
        if selection == team_a_name:
            probability_lower = (
                float(team_a_lower) if team_a_lower is not None else float(probability)
            )
            probability_upper = (
                float(team_a_upper) if team_a_upper is not None else float(probability)
            )
        else:
            probability_lower = (
                float(team_b_lower) if team_b_lower is not None else float(probability)
            )
            probability_upper = (
                float(team_b_upper) if team_b_upper is not None else float(probability)
            )
        rows.append(
            {
                **base,
                "market": "series_winner",
                "selection": selection,
                "model_value": float(probability),
                "raw_model_probability": (
                    raw_model_team_a
                    if selection == team_a_name
                    else 1.0 - raw_model_team_a
                    if raw_model_team_a is not None
                    else None
                ),
                "probability_lower": probability_lower,
                "probability_upper": probability_upper,
                "rating_baseline_probability": (
                    rating_baseline_team_a
                    if selection == team_a_name
                    else (
                        1.0 - rating_baseline_team_a
                        if rating_baseline_team_a is not None
                        else None
                    )
                ),
                "full_model_probability": (
                    full_model_team_a
                    if selection == team_a_name
                    else (
                        1.0 - full_model_team_a
                        if full_model_team_a is not None
                        else None
                    )
                ),
                "blend_weight": blend_weight,
                "strategy_version": "independent-winner-v2",
                "model_target": "series_winner",
                "attribution_stable": (
                    bool(drivers) if attribution_stable is None else attribution_stable
                ),
                "driver_attribution": list(driver_attribution),
            }
        )
    for market, value in prop_values.items():
        rows.append(
            {
                **base,
                "market": f"{market}_mean",
                "selection": None,
                "model_value": float(value),
            }
        )
    if map_prediction is not None:
        raw_map = map_prediction.get("raw_model_probability")
        for selection, probability, lower, upper in (
            (
                team_a_name,
                map_prediction["team1_win_probability"],
                map_prediction.get("team1_probability_lower"),
                map_prediction.get("team1_probability_upper"),
            ),
            (
                team_b_name,
                map_prediction["team2_win_probability"],
                map_prediction.get("team2_probability_lower"),
                map_prediction.get("team2_probability_upper"),
            ),
        ):
            rows.append(
                {
                    **base,
                    "market": "map_winner",
                    "selection": selection,
                    "model_value": float(probability),
                    "raw_model_probability": (
                        raw_map
                        if selection == team_a_name
                        else 1.0 - raw_map
                        if raw_map is not None
                        else None
                    ),
                    "probability_lower": float(lower or probability),
                    "probability_upper": float(upper or probability),
                    "probability_source": "independent_map_winner_model",
                    "uncertainty_method": map_prediction.get("uncertainty_method"),
                    "uncertainty_confidence": map_prediction.get(
                        "uncertainty_confidence"
                    ),
                    "uncertainty_sample_count": map_prediction.get(
                        "uncertainty_sample_count"
                    ),
                    "strategy_version": "map-winner-shadow-v1",
                    "model_target": "map_winner",
                }
            )
    return rows


def _health_detail(health: ArtifactHealth) -> str:
    failed = [check for check in health.checks if not check.ok]
    if not failed:
        return "ok"
    return "; ".join(f"{check.name}: {check.reason or 'missing'}" for check in failed)


def _step(name: str, fn: Callable[[], str]) -> DailyStepResult:
    try:
        detail = fn()
    except Exception as exc:
        logger.exception("Daily workflow step failed: %s", name)
        return DailyStepResult(name=name, ok=False, detail=str(exc))
    logger.info("Daily workflow step ok: %s - %s", name, detail)
    return DailyStepResult(name=name, ok=True, detail=detail)


def format_step_summary(steps: Sequence[DailyStepResult]) -> str:
    any_failed = any(not result.ok for result in steps)
    title = "**Oracle Bets Daily Workflow**"
    if any_failed:
        title += " — **FAILED** (predictions suppressed)"
    lines = [title]
    for result in steps:
        if not result.ok:
            status = "FAILED"
        elif result.detail.startswith("skipped"):
            status = "SKIPPED"
        else:
            status = "OK"
        lines.append(f"- {status}: {result.name} - {result.detail}")
    return "\n".join(lines)


def _run_mutating_steps(
    *,
    data_generator_factory: Callable[[], DataGenerator],
    module_factory: Callable[[], LoLBetsModule],
    store: EvidenceStore,
    scheduled_for: dt.datetime,
    effective_config: dict[str, Any],
    source_refresh_fn: Callable[[], Any] | None = None,
    source_check_fn: Callable[[], Any] | None = None,
    series_builder_fn: Callable[[], dict[str, Any]] = build_series_artifacts,
) -> list[DailyStepResult]:
    def _source_refresh() -> str:
        result = (source_refresh_fn or refresh_oracle_source)()
        return f"{len(result.files)} public files refreshed"

    def _source_check() -> str:
        report = (source_check_fn or _daily_source_readiness)()
        report.raise_if_unready()
        maximum = getattr(report, "current_year_max_match_at", None)
        return f"fresh through {maximum}" if maximum else "source ready"

    def _sync_identities() -> str:
        result = sync_history_identity_graph(
            store,
            pd.read_parquet(
                current_history_data_path(
                    RAW_DATA,
                    pointer_path=RAW_CURRENT_POINTER,
                )
            ),
            observed_at=dt.datetime.now(dt.UTC),
        )
        return (
            f"{result.added_identities} identities and "
            f"{result.added_links} provider links added"
        )

    def _validate() -> str:
        from oracle_bets_core.io_utils import load_training_data
        from oracle_bets_core.paths import TRAINING_PLAYER_DATA, TRAINING_TEAM_DATA

        team_df, player_df = load_training_data(
            TRAINING_TEAM_DATA, TRAINING_PLAYER_DATA, logger
        )
        validate_training_tables(team_df, player_df)
        return f"{team_df['gameid'].nunique()} games"

    def _build_series() -> str:
        report = series_builder_fn()
        return (
            f"{report['accepted_series']} series accepted; "
            f"{report['rejected_groups']} groups quarantined"
        )

    def _health() -> str:
        module = module_factory()
        inference = module.artifact_health()
        training = module.training_artifact_health()
        if not inference.ok:
            inference.raise_if_unhealthy()
        if not training.ok:
            training.raise_if_unhealthy()
        return f"inference={_health_detail(inference)}, training={_health_detail(training)}"

    store.initialize_schema()
    started_at = dt.datetime.now(dt.UTC)
    # A completed daily step is not evidence that source, transforms, or outputs
    # remain unchanged. Recompute on each invocation until a validated cache exists.
    workflow_key = (
        daily_run_key(
            "lol",
            scheduled_for=scheduled_for,
            config=effective_config,
        )
        + f":{started_at.isoformat()}:core"
    )
    journal = EvidenceWorkflowJournal(
        store,
        run_type="daily_lol_core",
        started_at=started_at,
    )
    outcome = run_workflow(
        workflow_key,
        (
            WorkflowStep(
                "source-refresh",
                _source_refresh,
                writes=True,
                retryable=True,
            ),
            WorkflowStep("source-check", _source_check, writes=False),
            WorkflowStep(
                "ingest",
                lambda: data_generator_factory().run() or "artifacts refreshed",
                writes=True,
                retryable=True,
            ),
            WorkflowStep("identity-graph", _sync_identities, writes=True),
            WorkflowStep("validate-data", _validate, writes=False),
            WorkflowStep("build-series", _build_series, writes=True),
        ),
        journal=journal,
        dry_run=False,
    )
    # Read the current champion even when source refresh or validation failed.
    # Health is independent of the ordered writes, but retains their journal.
    health = run_workflow(
        workflow_key,
        (WorkflowStep("health", _health, writes=False),),
        journal=journal,
        dry_run=False,
    )
    return [
        DailyStepResult(
            name=step.name,
            ok=step.status in {"completed", "skipped_completed", "skipped_dry_run"},
            detail=(
                "skipped after failure"
                if step.status == "skipped_after_failure"
                else (
                    f"skipped completed: {step.detail}"
                    if step.status == "skipped_completed"
                    else step.detail
                )
            ),
        )
        for step in (*outcome.steps, *health.steps)
    ]


def _build_prediction_snapshots(
    schedule: pd.DataFrame,
    *,
    cfg: DailyWorkflowConfig,
    predictor_factory: Callable[[], Predictor] | None = None,
    snapshot_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schedule = _actionable_schedule(schedule)
    if schedule.empty:
        return []
    predictor = predictor_factory() if predictor_factory is not None else None
    if predictor is None:
        # Dry run included: exercising the real prediction path is the point of
        # the smoke test. Fall back to a stub only when artifacts can't load.
        try:
            from lol_bets.inference.match_predictor import MatchPredictor

            predictor = MatchPredictor()
        except Exception as exc:
            if cfg.dry_run:
                return [{"status": "artifacts_unavailable", "reason": str(exc)}]
            raise

    details: list[dict[str, Any]] = []
    actionable = set(actionable_leagues())
    for _, row in schedule.iterrows():
        try:
            resolutions: list[dict[str, str]] = []
            build_match_prediction_snapshot(
                row,
                predictor=predictor,
                snapshot_sink=snapshot_sink,
                resolution_sink=resolutions,
            )
            league = str(row.get("league") or "")
            details.append(
                {
                    "status": "predicted",
                    "league": league,
                    "actionable_league": league in actionable,
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "resolved_aliases": resolutions,
                }
            )
        except TeamResolutionError as exc:
            details.append(
                {
                    "status": "unsupported_team",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "team": exc.resolved.query,
                    "suggestions": list(exc.resolved.suggestions),
                }
            )
        except InsufficientRosterHistoryError as exc:
            details.append(
                {
                    "status": "insufficient_history",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "reason": str(exc),
                }
            )
        except Exception as exc:
            details.append(
                {
                    "status": "prediction_unavailable",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "reason": str(exc),
                }
            )
    return details


def write_daily_report(
    *,
    cfg: DailyWorkflowConfig,
    steps: Sequence[DailyStepResult],
    schedule: pd.DataFrame,
    excluded_fixtures: Sequence[dict[str, Any]],
    messages: Sequence[str],
    cadence_reminders: Sequence[dict[str, Any]] = (),
    open_positions: dict[str, Any] | None = None,
    source_freshness: dict[str, Any] | None = None,
    drift_review: dict[str, Any] | None = None,
    report_paths: tuple[Path, Path] | None = None,
) -> tuple[Path, Path]:
    """Persist machine-readable and human-readable reviews for every daily run."""
    report_dir = REPORTS_DIR / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    if report_paths is None:
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        json_path = report_dir / f"{timestamp}.json"
        markdown_path = report_dir / f"{timestamp}.md"
    else:
        json_path, markdown_path = report_paths
    payload = {
        "schema_version": 6,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "evidence_run_id": _evidence_run_id(steps),
        "config": dict(vars(cfg)),
        "steps": [vars(step) for step in steps],
        "schedule": schedule.to_dict(orient="records"),
        "excluded_fixtures": list(excluded_fixtures),
        "messages": list(messages),
        "cadence_reminders": list(cadence_reminders),
        "open_positions": open_positions or {"available": False, "count": None},
        "source_freshness": source_freshness,
        "drift_review": drift_review,
    }
    atomic_write_text(
        json_path,
        json.dumps(payload, indent=2, default=str) + "\n",
    )
    excluded_lines = [
        f"- {item.get('league')}: {item.get('team_a') or 'TBD'} vs "
        f"{item.get('team_b') or 'TBD'} ({item.get('reason')})"
        for item in excluded_fixtures
    ]
    markdown = "\n\n".join(messages)
    if excluded_lines:
        markdown += "\n\n## Excluded fixtures\n" + "\n".join(excluded_lines)
    atomic_write_text(markdown_path, markdown + "\n")
    return json_path, markdown_path


def _evidence_run_id(steps: Sequence[DailyStepResult]) -> str | None:
    prefix = "canonical run recorded: "
    return next(
        (
            step.detail.removeprefix(prefix)
            for step in steps
            if step.name == "evidence" and step.ok and step.detail.startswith(prefix)
        ),
        None,
    )


def _daily_source_readiness():
    return inspect_oracle_source()


def _latest_model_drift_review() -> dict[str, Any] | None:
    try:
        champion = json.loads(
            (MODEL_REGISTRY_DIR / "champion.json").read_text(encoding="utf-8")
        )["model_id"]
        review = json.loads(
            (MODEL_REGISTRY_DIR / "reviews" / f"{champion}.json").read_text(
                encoding="utf-8"
            )
        )
        drift = review.get("evidence", {}).get("drift_review")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return drift if isinstance(drift, dict) else None


def _refresh_expected_lineups(
    schedule: pd.DataFrame,
    *,
    schedule_available: bool,
    refresher_factory: Callable[[], PandaScoreLineupRefresher] | None,
) -> tuple[pd.DataFrame, DailyStepResult]:
    if not schedule_available or schedule.empty:
        return schedule, DailyStepResult(
            "lineups", True, "skipped because schedule is unavailable"
        )
    try:
        if refresher_factory is not None:
            refresher = refresher_factory()
        elif os.getenv("PANDASCORE_API_KEY"):
            refresher = PandaScoreLineupRefresher()
        else:
            return schedule, DailyStepResult(
                "lineups",
                True,
                "skipped: PandaScore API key unavailable",
            )
        refreshed = refresher.refresh(schedule)
        errors = int(
            refreshed["lineup_refresh_error"].fillna("").astype(str).ne("").sum()
        )
        return refreshed, DailyStepResult(
            "lineups",
            True,
            f"{len(refreshed) - errors} fixture details refreshed; "
            f"{errors} retained with warnings",
        )
    except Exception as exc:
        logger.exception("Expected-lineup refresh failed.")
        failed = schedule.copy()
        failed["lineup_refresh_error"] = f"{type(exc).__name__}: refresh_unavailable"
        return failed, DailyStepResult(
            "lineups",
            True,
            f"unavailable ({exc}); embedded fixture lineups retained",
        )


def _fetch_daily_schedule(
    *,
    cfg: DailyWorkflowConfig,
    run_now: dt.datetime,
    fetch_window_days: int,
    schedule_fetcher: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    try:
        schedule = schedule_fetcher(
            start_datetime=run_now,
            window_days=fetch_window_days,
            leagues=cfg.leagues,
            save_path=None if cfg.dry_run else SCHEDULE,
        )
        detail = "fetched"
    except Exception as exc:
        try:
            schedule = pd.read_parquet(SCHEDULE)
        except Exception:
            raise exc from None
        age = dt.datetime.now(dt.UTC).timestamp() - SCHEDULE.stat().st_mtime
        if age > 2 * 60 * 60:
            raise RuntimeError(
                "PandaScore fetch failed and the stored schedule is older than two "
                "hours; refusing stale fixtures."
            ) from exc
        detail = f"fetch failed ({exc}); using stored schedule"
    if cfg.leagues is None:
        raise ValueError("daily league filter was not resolved")
    filtered = PandaScoreSchedule.filter_by_league(schedule, cfg.leagues)
    return filtered.reset_index(drop=True), detail


def run_daily_lol_workflow(
    config: DailyWorkflowConfig | None = None,
    *,
    schedule_fetcher: Callable[..., pd.DataFrame] = fetch_and_store_schedule,
    data_generator_factory: Callable[[], DataGenerator] = DataGenerator,
    module_factory: Callable[[], LoLBetsModule] = LoLBetsModule,
    lineup_refresher_factory: (Callable[[], PandaScoreLineupRefresher] | None) = None,
    series_builder_fn: Callable[[], dict[str, Any]] = build_series_artifacts,
) -> DailyWorkflowResult:
    """Refresh data and schedule, check the champion, and report without training."""
    cfg = _resolve_daily_config(config)
    delivery_mode = DiscordDeliveryMode(str(cfg.delivery_mode))
    steps: list[DailyStepResult] = []
    run_now, scheduled_for, fetch_window_days = _daily_time_window(cfg.horizon_hours)
    source_readiness = _daily_source_readiness()
    effective_config = dict(vars(cfg))

    if not cfg.dry_run:
        steps.extend(
            _run_mutating_steps(
                data_generator_factory=data_generator_factory,
                module_factory=module_factory,
                store=EvidenceStore(EVIDENCE_DB),
                scheduled_for=scheduled_for,
                effective_config=effective_config,
                series_builder_fn=series_builder_fn,
            )
        )
        source_readiness = _daily_source_readiness()

    # The schedule fetch is an external dependency and must degrade like every
    # other step: a PandaScore outage should produce a FAILED step and a
    # report, never a stack trace. When the fetch fails but a stored schedule
    # exists, fall back to it (marked in the step detail).
    run_now = dt.datetime.now(dt.UTC)
    fetch_state: dict[str, pd.DataFrame] = {}

    def _fetch_schedule() -> str:
        schedule, detail = _fetch_daily_schedule(
            cfg=cfg,
            run_now=run_now,
            fetch_window_days=fetch_window_days,
            schedule_fetcher=schedule_fetcher,
        )
        fetch_state["schedule"] = schedule
        return detail

    schedule_step = _step("schedule", _fetch_schedule)
    schedule = fetch_state.get("schedule", pd.DataFrame())
    schedule, lineup_step = _refresh_expected_lineups(
        schedule,
        schedule_available=schedule_step.ok,
        refresher_factory=lineup_refresher_factory,
    )
    daily_schedule = _actionable_schedule(
        filter_daily_schedule(
            schedule,
            now=dt.datetime.now(dt.UTC),
            horizon_hours=cfg.horizon_hours,
        )
    ).reset_index(drop=True)
    reportable_schedule, excluded_fixtures = split_reportable_schedule(daily_schedule)
    if schedule_step.ok:
        suffix = (
            "" if schedule_step.detail == "fetched" else f" ({schedule_step.detail})"
        )
        schedule_step = DailyStepResult(
            name="schedule",
            ok=True,
            detail=(
                f"{len(reportable_schedule)} reportable matches in next "
                f"{cfg.horizon_hours} hours"
                f" ({len(excluded_fixtures)} excluded){suffix}"
            ),
        )
    steps.extend([schedule_step, lineup_step])
    messages = [
        format_step_summary(steps),
        *format_schedule_messages(_actionable_schedule(reportable_schedule)),
    ]

    if not cfg.dry_run and schedule_step.ok:
        try:
            run_id = record_daily_evidence(
                store=EvidenceStore(EVIDENCE_DB),
                scheduled_for=scheduled_for,
                observed_at=dt.datetime.now(dt.UTC),
                effective_config=effective_config,
                schedule=daily_schedule,
                snapshot_rows=(),
                steps=steps,
            )
            steps.append(
                DailyStepResult("evidence", True, f"canonical run recorded: {run_id}")
            )
        except Exception as exc:
            logger.exception("Canonical evidence write failed.")
            steps.append(DailyStepResult("evidence", False, str(exc)))

    messages[0] = format_step_summary(steps)

    reminders = cadence_reminders(run_now)
    reminder_message = format_cadence_reminders(reminders)
    if reminder_message:
        messages.append(reminder_message)
    open_positions = _open_position_summary(EvidenceStore(EVIDENCE_DB))
    source_freshness = source_readiness.to_dict()
    drift_review = _latest_model_drift_review()
    messages.append(_format_open_position_summary(open_positions))
    if not cfg.dry_run and delivery_mode is DiscordDeliveryMode.GATEWAY:
        steps.append(
            DailyStepResult(
                "discord",
                True,
                "Gateway owner console enabled; open `/oracle` for interactive review",
            )
        )
        messages[0] = format_step_summary(steps)
    report_paths = write_daily_report(
        cfg=cfg,
        steps=steps,
        schedule=reportable_schedule,
        excluded_fixtures=excluded_fixtures,
        messages=messages,
        cadence_reminders=reminders,
        open_positions=open_positions,
        source_freshness=source_freshness,
        drift_review=drift_review,
    )
    return DailyWorkflowResult(
        schedule=reportable_schedule,
        messages=messages,
        steps=steps,
        excluded_fixtures=excluded_fixtures,
        cadence_reminders=reminders,
        open_positions=open_positions,
        report_paths=report_paths,
    )
