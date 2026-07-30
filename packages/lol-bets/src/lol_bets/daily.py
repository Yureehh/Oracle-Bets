"""Daily League of Legends prediction workflow orchestration."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

import requests
from oracle_bets_core.betting import expected_edge
from oracle_bets_core.config import load_product_config
from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.league_selection import selected_leagues
from oracle_bets_core.logger import logger
from oracle_bets_core.markets import MarketAdapter, MarketQuote, PolymarketGammaAdapter
from oracle_bets_core.operations import (
    EvidenceWorkflowJournal,
    WorkflowStep,
    daily_run_key,
    run_workflow,
)
from oracle_bets_core.paths import (
    EVIDENCE_DB,
    INSIGHTS_DIR,
    MODEL_REGISTRY_DIR,
    RAW_DATA,
    REPORTS_DIR,
    SCHEDULE,
)
from oracle_bets_core.pd import pd
from oracle_bets_discord.formatting import MESSAGE_LIMIT
from oracle_bets_discord.predictions.lol import (
    confidence_label,
    context_line,
    format_schedule_messages,
    format_warnings,
    format_winner_market_output,
    get_empty_roster,
    outcome_probability_source,
    prop_probability_source,
)

from lol_bets.data_generation.ingestion.schedule import (
    PandaScoreLineupRefresher,
    fetch_and_store_schedule,
)
from lol_bets.inference.roster import (
    EXPECTED_STARTERS,
    RosterGateDecision,
    RosterGateEvidence,
    completed_series_with_roster,
    evaluate_roster_gate,
)
from lol_bets.inference.series import derive_series_distribution
from lol_bets.inference.team import InsufficientRosterHistoryError, Team
from lol_bets.inference.team_resolver import TeamResolutionError
from lol_bets.module import LoLBetsModule
from lol_bets.operations.evidence import record_daily_evidence
from lol_bets.operations.identity import sync_history_identity_graph
from lol_bets.operations.models import (
    ModelRegistry,
    evaluate_training_triggers_from_history,
    orchestrate_candidate_training,
    register_current_candidate,
)
from lol_bets.pipeline import DataGenerator
from lol_bets.training import train_models, validate_training_tables

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from oracle_bets_core.interfaces import ArtifactHealth


class Predictor(Protocol):
    outcome_calibrator: Any

    def predict_match(
        self,
        team1: Team,
        team2: Team,
        account_for_side: bool = True,
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
    webhook_url: str | None = None
    dry_run: bool = False
    skip_retrain: bool = False
    skip_market_search: bool = False
    targets: str = "all"
    feature_set: str = "compact"
    max_features: int = 120


EXCLUDED_DAILY_LEAGUES = frozenset({"Equal eSports Cup"})
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
class DailyWorkflowResult:
    schedule: pd.DataFrame
    messages: list[str]
    steps: list[DailyStepResult] = field(default_factory=list)
    excluded_fixtures: list[dict[str, Any]] = field(default_factory=list)
    prediction_details: list[dict[str, Any]] = field(default_factory=list)
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
        team_a = str(row.get("team_a") or "").strip()
        team_b = str(row.get("team_b") or "").strip()
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


@lru_cache(maxsize=1)
def _roster_history() -> pd.DataFrame:
    return pd.read_parquet(
        RAW_DATA,
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
    except (TypeError, ValueError):
        return "bo1"
    if value in {1, 2, 3, 5}:
        return f"bo{value}"
    return "bo1"


def send_discord_webhook_messages(
    webhook_url: str,
    messages: Sequence[str],
    *,
    session: requests.Session | None = None,
) -> None:
    """Post messages to a Discord webhook URL."""
    http = session or requests.Session()
    for message in messages:
        response = http.post(
            webhook_url,
            json={
                "content": message[: MESSAGE_LIMIT - 1],
                "allowed_mentions": {"parse": []},
            },
            timeout=15,
        )
        response.raise_for_status()


def _normalize_market_text(value: str) -> set[str]:
    return {
        part
        for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()
        if part
    }


# Tokens that are common English words or generic org words. Short team names
# like "T1", "WE", or "G2" otherwise false-positive on unrelated questions.
_MATCH_STOPWORDS = frozenset(
    {"we", "will", "the", "team", "of", "in", "vs", "beat", "win", "and", "to", "a"}
)


def _team_matches_text(team_name: str, text: str) -> bool:
    """All informative tokens of the team name must appear in the text."""
    team_tokens = _normalize_market_text(team_name)
    if not team_tokens:
        return False
    informative = {t for t in team_tokens if t not in _MATCH_STOPWORDS}
    # A name made entirely of stopword-like tokens ("Team WE") must match on
    # its full token set instead.
    required = informative or team_tokens
    return required <= _normalize_market_text(text)


def _quote_score(quote: MarketQuote, team_a: str, team_b: str) -> int:
    question_tokens = _normalize_market_text(quote.question)
    score = 0
    if _team_matches_text(team_a, quote.question):
        score += 1
    if _team_matches_text(team_b, quote.question):
        score += 1
    if _team_matches_text(team_a, quote.outcome) or _team_matches_text(
        team_b, quote.outcome
    ):
        score += 1
    if {"lol", "league", "legends", "esports"} & question_tokens:
        score += 1
    return score


def select_market_candidates(
    quotes: Sequence[MarketQuote],
    *,
    team_a: str,
    team_b: str,
    min_score: int = 3,
    limit: int = 6,
) -> list[MarketQuote]:
    """Conservatively keep quotes that mention both the matchup and a selection."""
    scored = [
        (quote, _quote_score(quote, team_a, team_b))
        for quote in quotes
        if quote.implied_probability is not None
    ]
    kept = [quote for quote, score in scored if score >= min_score]
    kept.sort(
        key=lambda quote: (quote.liquidity or 0.0, quote.volume or 0.0), reverse=True
    )
    return kept[:limit]


def format_market_candidates(
    *,
    team_a: str,
    team_b: str,
    team_a_probability: float,
    team_b_probability: float,
    quotes: Sequence[MarketQuote],
) -> str:
    candidates = select_market_candidates(quotes, team_a=team_a, team_b=team_b)
    if not candidates:
        return "Polymarket: no confident matching active market found."
    lines = [
        "Polymarket candidates (read-only, verify manually):",
        "```text",
        f"{'Outcome':<18} {'Poly':>6} {'Model':>7} {'Edge':>7} {'Liquid':>9}  Market",
    ]
    for quote in candidates:
        if quote.implied_probability is None:
            continue
        price = float(quote.implied_probability)
        if _team_matches_text(team_a, quote.outcome):
            model_probability = team_a_probability
        elif _team_matches_text(team_b, quote.outcome):
            model_probability = team_b_probability
        else:
            model_probability = 0.0
        edge = (
            expected_edge(1.0 / price, model_probability)
            if price > 0 and model_probability > 0
            else None
        )
        edge_text = f"{edge * 100:+.1f}%" if edge is not None else "verify"
        liquidity_text = (
            f"{quote.liquidity:,.0f}" if quote.liquidity is not None else "n/a"
        )
        lines.append(
            f"{quote.outcome[:18]:<18} {price * 100:>5.0f}c "
            f"{model_probability * 100:>6.1f}% {edge_text:>7} {liquidity_text:>9}  "
            f"{quote.question[:38]}"
        )
    lines.append("```")
    urls = [quote.url for quote in candidates if quote.url]
    if urls:
        lines.append("Links: " + " | ".join(urls[:3]))
    return "\n".join(lines)


def build_match_prediction_message(  # noqa: PLR0915
    row: pd.Series,
    *,
    predictor: Predictor,
    market_search: MarketAdapter | None = None,
    snapshot_sink: list[dict[str, Any]] | None = None,
) -> str:
    """Build one Discord-safe daily prediction message for a scheduled match."""
    team_a_name = str(row.get("team_a") or "").strip()
    team_b_name = str(row.get("team_b") or "").strip()
    match_type = match_type_from_best_of(row.get("best_of"))
    team_a_roster, team_a_lineup_ready = expected_roster_from_schedule(row, team="a")
    team_b_roster, team_b_lineup_ready = expected_roster_from_schedule(row, team="b")
    lineup_ready = team_a_lineup_ready and team_b_lineup_ready
    team_a = Team(
        name=team_a_name,
        side="Blue",
        first_pick=None,
        as_of_date=row.get("start_utc"),
        roster=team_a_roster,
    )
    team_b = Team(
        name=team_b_name,
        side="Red",
        first_pick=None,
        as_of_date=row.get("start_utc"),
        roster=team_b_roster,
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
    drivers = [str(value) for value in prediction.get("drivers", [])]
    warnings: list[str] = []
    if not lineup_ready:
        warnings.append(
            "Expected lineup is incomplete or unknown; this prediction is shadow-only."
        )
    elif not roster_ready:
        warnings.append(
            "An expected roster is not yet stable; this prediction is shadow-only "
            f"({team_a_name}: {team_a_roster_gate.state}, "
            f"{team_a_roster_gate.completed_series}/"
            f"{team_a_roster_gate.required_completed_series} completed series; "
            f"{team_b_name}: {team_b_roster_gate.state}, "
            f"{team_b_roster_gate.completed_series}/"
            f"{team_b_roster_gate.required_completed_series})."
        )
    if getattr(predictor, "outcome_calibrator", None) is None:
        warnings.append(
            "Outcome calibration artifact is missing; raw model probability is being used."
        )
    if not uncertainty_method:
        warnings.append(
            "Held-out probability uncertainty is unavailable; the point estimate "
            "cannot be considered action-ready."
        )
    if not drivers:
        warnings.append(
            "Local model drivers are unavailable; the prediction is shadow-only."
        )

    output = format_winner_market_output(
        blue_team_name=team_a_name,
        red_team_name=team_b_name,
        match_type=match_type,
        blue_win=team_a_win,
        red_win=team_b_win,
        probability_source=outcome_probability_source(predictor),
        warnings=warnings,
        context=context_line(team_a, team_b, False, None),
        blue_range=(team_a_lower, team_a_upper),
        red_range=(team_b_lower, team_b_upper),
        uncertainty_confidence=uncertainty_confidence,
        drivers=drivers,
    )

    prop_values: dict[str, float] = {}
    prop_lines: list[str] = []
    for label, method_name, source_name in (
        ("Length", "predict_gamelength", "gamelength"),
        ("Kills", "predict_total_kills", "total_kills"),
        ("Towers", "predict_total_towers", "total_towers"),
    ):
        try:
            value = getattr(predictor, method_name)(
                team_a, team_b, account_for_side=False
            )
        except Exception as exc:
            prop_lines.append(f"- {label}: unavailable ({exc})")
            continue
        prop_values[source_name] = float(value)
        suffix = "m" if label == "Length" else ""
        prop_lines.append(
            f"- {label}: {value:.1f}{suffix} ({prop_probability_source(predictor, source_name)})"
        )
    if prop_lines:
        output += "\n\nProps snapshot:\n" + "\n".join(prop_lines)

    quotes: list[MarketQuote] = []
    if market_search is not None:
        query = str(row.get("market_query") or f"{team_a_name} {team_b_name} LoL")
        try:
            quotes = market_search.search(query, limit=12)
            output += "\n\n" + format_market_candidates(
                team_a=team_a_name,
                team_b=team_b_name,
                team_a_probability=team_a_win,
                team_b_probability=team_b_win,
                quotes=quotes,
            )
        except Exception as exc:
            quotes = []
            output += f"\n\nPolymarket: search failed ({exc})."

    if snapshot_sink is not None:
        snapshot_sink.extend(
            build_prediction_snapshot_rows(
                row,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                match_type=match_type,
                team_a_win=team_a_win,
                team_b_win=team_b_win,
                team_a_lower=team_a_lower,
                team_a_upper=team_a_upper,
                team_b_lower=team_b_lower,
                team_b_upper=team_b_upper,
                probability_source=outcome_probability_source(predictor),
                uncertainty_method=uncertainty_method,
                uncertainty_confidence=uncertainty_confidence,
                uncertainty_sample_count=uncertainty_sample_count,
                drivers=drivers,
                team_a_roster_gate=team_a_roster_gate,
                team_b_roster_gate=team_b_roster_gate,
                prop_values=prop_values,
                lineup_ready=lineup_ready,
                roster_ready=roster_ready,
                quotes=quotes,
            )
        )

    output += f"\n\nDaily confidence: **{confidence_label(warnings)}**"
    output += format_warnings(warnings)
    return output[: MESSAGE_LIMIT - 1]


PREDICTION_SNAPSHOTS = REPORTS_DIR / "prediction_snapshots.parquet"


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
    team_a_lower: float | None = None,
    team_a_upper: float | None = None,
    team_b_lower: float | None = None,
    team_b_upper: float | None = None,
    uncertainty_method: str | None = None,
    uncertainty_confidence: float | None = None,
    uncertainty_sample_count: int | None = None,
    drivers: Sequence[str] = (),
    team_a_roster_gate: RosterGateDecision | None = None,
    team_b_roster_gate: RosterGateDecision | None = None,
    lineup_ready: bool = False,
    roster_ready: bool = False,
    quotes: Sequence[MarketQuote] = (),
) -> list[dict[str, Any]]:
    """One snapshot row per (match, market, selection) for later CLV/backtests."""
    run_ts = dt.datetime.now(dt.UTC)
    paired_distribution = _paired_map_series_distribution(
        match_type=match_type,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        team_a_point=team_a_win,
        team_b_point=team_b_win,
        team_a_lower=team_a_lower,
        team_a_upper=team_a_upper,
        team_b_lower=team_b_lower,
        team_b_upper=team_b_upper,
    )
    base = {
        "run_ts": run_ts,
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
    candidates = select_market_candidates(
        quotes, team_a=team_a_name, team_b=team_b_name
    )

    def _best_quote_for(team_name: str) -> MarketQuote | None:
        for quote in candidates:
            if _team_matches_text(team_name, quote.outcome):
                return quote
        return None

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
        quote = _best_quote_for(selection)
        rows.append(
            {
                **base,
                "market": "winner",
                "selection": selection,
                "model_value": float(probability),
                "probability_lower": probability_lower,
                "probability_upper": probability_upper,
                "poly_price": (
                    float(quote.implied_probability)
                    if quote is not None and quote.implied_probability is not None
                    else None
                ),
                "poly_market_id": quote.market_id if quote is not None else None,
                "poly_question": quote.question if quote is not None else None,
                "poly_url": quote.url if quote is not None else None,
                "poly_liquidity": (
                    float(quote.liquidity)
                    if quote is not None and quote.liquidity is not None
                    else None
                ),
            }
        )
    for market, value in prop_values.items():
        rows.append(
            {
                **base,
                "market": f"{market}_mean",
                "selection": None,
                "model_value": float(value),
                "poly_price": None,
                "poly_question": None,
                "poly_url": None,
                "poly_liquidity": None,
            }
        )
    return rows


def _paired_map_series_distribution(
    *,
    match_type: str,
    team_a_name: str,
    team_b_name: str,
    team_a_point: float,
    team_b_point: float,
    team_a_lower: float | None,
    team_a_upper: float | None,
    team_b_lower: float | None,
    team_b_upper: float | None,
) -> dict[str, Any]:
    best_of = int(match_type.removeprefix("bo"))
    a_lower = float(team_a_lower if team_a_lower is not None else team_a_point)
    a_upper = float(team_a_upper if team_a_upper is not None else team_a_point)
    b_lower = float(team_b_lower if team_b_lower is not None else team_b_point)
    b_upper = float(team_b_upper if team_b_upper is not None else team_b_point)
    point = derive_series_distribution(
        best_of,
        [team_a_point] * best_of,
    )
    low = derive_series_distribution(best_of, [a_lower] * best_of)
    high = derive_series_distribution(best_of, [a_upper] * best_of)
    return {
        "method": "derived_from_map_engine",
        "best_of": best_of,
        "map_1": {
            team_a_name: {
                "point": team_a_point,
                "lower": a_lower,
                "upper": a_upper,
            },
            team_b_name: {
                "point": team_b_point,
                "lower": b_lower,
                "upper": b_upper,
            },
        },
        "series": {
            team_a_name: {
                "point": point.team_a_win,
                "lower": low.team_a_win,
                "upper": high.team_a_win,
            },
            team_b_name: {
                "point": point.team_b_win,
                "lower": high.team_b_win,
                "upper": low.team_b_win,
            },
            **({"draw": {"point": point.draw}} if point.draw else {}),
        },
        "score_probabilities": dict(point.score_probabilities),
        "total_maps_probabilities": {
            str(total): probability
            for total, probability in point.total_maps_probabilities.items()
        },
    }


def append_prediction_snapshots(
    rows: Sequence[dict[str, Any]],
    *,
    path: Any = PREDICTION_SNAPSHOTS,
) -> int:
    """
    Append snapshot rows, idempotent per (run_date, teams, market, selection):
    re-running the workflow on the same day replaces that day's rows for the
    same match/market instead of duplicating them.
    """
    if not rows:
        return 0
    new = pd.DataFrame(list(rows))
    key_cols = ["run_date", "team_a", "team_b", "start_utc", "market", "selection"]
    try:
        existing = pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        existing = pd.DataFrame()
    if not existing.empty:
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(
        drop=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return len(new)


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
    return "\n".join(lines)[: MESSAGE_LIMIT - 1]


def _run_mutating_steps(
    cfg: DailyWorkflowConfig,
    *,
    data_generator_factory: Callable[[], DataGenerator],
    train_fn: Callable[..., Path | None],
    module_factory: Callable[[], LoLBetsModule],
    store: EvidenceStore,
    scheduled_for: dt.datetime,
    effective_config: dict[str, Any],
) -> list[DailyStepResult]:
    def _sync_identities() -> str:
        result = sync_history_identity_graph(
            store,
            pd.read_parquet(RAW_DATA),
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
    workflow_key = (
        daily_run_key(
            "lol",
            scheduled_for=scheduled_for,
            config=effective_config,
        )
        + ":core"
    )
    journal = EvidenceWorkflowJournal(
        store,
        run_type="daily_lol_core",
        started_at=dt.datetime.now(dt.UTC),
    )
    outcome = run_workflow(
        workflow_key,
        (
            WorkflowStep(
                "ingest",
                lambda: data_generator_factory().run() or "artifacts refreshed",
                writes=True,
                retryable=True,
            ),
            WorkflowStep("identity-graph", _sync_identities, writes=True),
            WorkflowStep("validate-data", _validate, writes=False),
            WorkflowStep(
                "train",
                (
                    (lambda: "skipped by option")
                    if cfg.skip_retrain
                    else lambda: _train_triggered_candidate(cfg, train_fn)
                ),
                writes=not cfg.skip_retrain,
            ),
            WorkflowStep("health", _health, writes=False),
        ),
        journal=journal,
        dry_run=False,
    )
    return [
        DailyStepResult(
            name=step.name,
            ok=step.status in {"completed", "skipped_completed", "skipped_dry_run"}
            or step.status == "skipped_after_failure",
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
        for step in outcome.steps
    ]


def _train_triggered_candidate(
    cfg: DailyWorkflowConfig,
    train_fn: Callable[..., Path | None],
) -> str:
    evaluated_at = dt.datetime.now(dt.UTC)
    product = load_product_config()
    registry = ModelRegistry(MODEL_REGISTRY_DIR)
    history = pd.read_parquet(
        RAW_DATA,
        columns=["gameid", "date", "league", "datacompleteness"],
    )
    evaluation = evaluate_training_triggers_from_history(
        history,
        registry=registry,
        evaluated_at=evaluated_at,
        major_leagues=selected_leagues(product.leagues.profile),
        valid_map_threshold=product.training.new_valid_maps_trigger,
        major_map_threshold=product.training.new_major_maps_trigger,
    )
    if not evaluation.triggered:
        state = evaluation.state
        return (
            "not triggered: "
            f"{state.new_valid_maps} new valid maps, "
            f"{state.new_major_maps} new major maps"
        )
    code_version = str(os.getenv("ORACLE_BETS_CODE_VERSION") or "").strip()
    if not code_version:
        raise ValueError(
            "ORACLE_BETS_CODE_VERSION is required to register a triggered candidate"
        )

    def _train() -> None:
        train_fn(
            targets=cfg.targets,
            force_retune=False,
            feature_set=cfg.feature_set,
            max_features=cfg.max_features,
        )

    def _register(model_id: str) -> None:
        register_current_candidate(
            registry=registry,
            model_id=model_id,
            target="map_win",
            code_version=code_version,
            metrics=_latest_outcome_metrics(),
            created_at=evaluated_at,
        )

    result = orchestrate_candidate_training(
        evaluation,
        train_candidate=_train,
        register_candidate=_register,
    )
    reasons = ", ".join(evaluation.reasons)
    return f"registered immutable {result.registered_model_id}; triggers: {reasons}"


def _latest_outcome_metrics() -> dict[str, float]:
    root = INSIGHTS_DIR / "OutcomePrediction_LightGBM"
    candidates = sorted(root.glob("*/metrics.json"))
    if not candidates:
        raise ValueError("outcome evaluation metrics are unavailable")
    try:
        payload = json.loads(candidates[-1].read_text())
        return {
            key: float(payload[key]) for key in ("log_loss", "brier", "calibration_ece")
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("latest outcome metrics are invalid") from exc


def _format_unmatched_teams_message(
    unmatched: dict[str, tuple[str, ...]],
) -> str:
    lines = [
        "**Unmatched teams** — no prediction was generated for these. "
        "Add entries to `config/lol/data_ingestion/team_aliases.json`:"
    ]
    for name, suggestions in sorted(unmatched.items()):
        hint = f" (closest: {', '.join(suggestions)})" if suggestions else ""
        lines.append(f"- {name}{hint}")
    return "\n".join(lines)[: MESSAGE_LIMIT - 1]


def _build_prediction_messages(
    schedule: pd.DataFrame,
    *,
    cfg: DailyWorkflowConfig,
    predictor_factory: Callable[[], Predictor] | None,
    market_search_factory: Callable[[], MarketAdapter],
    snapshot_sink: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if schedule.empty:
        return [], []
    predictor = predictor_factory() if predictor_factory is not None else None
    if predictor is None:
        # Dry run included: exercising the real prediction path is the point of
        # the smoke test. Fall back to a stub only when artifacts can't load.
        try:
            from lol_bets.inference.match_predictor import MatchPredictor

            predictor = MatchPredictor()
        except Exception as exc:
            if cfg.dry_run:
                return (
                    [
                        "Dry run: prediction artifacts unavailable, "
                        f"prediction generation skipped ({exc})."
                    ],
                    [{"status": "artifacts_unavailable", "reason": str(exc)}],
                )
            raise

    market_search = None if cfg.skip_market_search else market_search_factory()
    messages: list[str] = []
    details: list[dict[str, Any]] = []
    unmatched: dict[str, tuple[str, ...]] = {}
    insufficient_history: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for _, row in schedule.iterrows():
        try:
            message = build_match_prediction_message(
                row,
                predictor=predictor,
                market_search=market_search,
                snapshot_sink=snapshot_sink,
            )
            messages.append(message)
            details.append(
                {
                    "status": "predicted",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "report": message,
                }
            )
        except TeamResolutionError as exc:
            unmatched[exc.resolved.query] = exc.resolved.suggestions
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
            insufficient_history.append(
                {"match": str(row.get("discord_label", "match")), "reason": str(exc)}
            )
            details.append(
                {
                    "status": "insufficient_history",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "reason": str(exc),
                }
            )
        except Exception as exc:
            failures.append(
                {"match": str(row.get("discord_label", "match")), "reason": str(exc)}
            )
            details.append(
                {
                    "status": "prediction_unavailable",
                    "match_key": row.get("match_key"),
                    "match": row.get("discord_label"),
                    "reason": str(exc),
                }
            )
    if unmatched:
        messages.append(_format_unmatched_teams_message(unmatched))
    if insufficient_history:
        messages.append(
            f"Insufficient roster history for {len(insufficient_history)} fixture(s); "
            "no prediction was fabricated. See the daily report artifact."
        )
    if failures:
        messages.append(
            f"Prediction unavailable for {len(failures)} fixture(s); see the daily report artifact."
        )
    return messages, details


def write_daily_report(
    *,
    cfg: DailyWorkflowConfig,
    steps: Sequence[DailyStepResult],
    schedule: pd.DataFrame,
    excluded_fixtures: Sequence[dict[str, Any]],
    prediction_details: Sequence[dict[str, Any]],
    messages: Sequence[str],
) -> tuple[Path, Path]:
    """Persist machine-readable and human-readable reviews for every daily run."""
    report_dir = REPORTS_DIR / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    json_path = report_dir / f"{timestamp}.json"
    markdown_path = report_dir / f"{timestamp}.md"
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "config": {
            key: value for key, value in vars(cfg).items() if key != "webhook_url"
        },
        "steps": [vars(step) for step in steps],
        "schedule": schedule.to_dict(orient="records"),
        "excluded_fixtures": list(excluded_fixtures),
        "predictions": list(prediction_details),
        "messages": list(messages),
    }
    _atomic_write_report(
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
    _atomic_write_report(markdown_path, markdown + "\n")
    return json_path, markdown_path


def _atomic_write_report(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _refresh_expected_lineups(
    schedule: pd.DataFrame,
    *,
    schedule_available: bool,
    observed_at: dt.datetime,
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
        refreshed = refresher.refresh(schedule, observed_at=observed_at)
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
        return schedule, DailyStepResult(
            "lineups",
            True,
            f"unavailable ({exc}); embedded fixture lineups retained",
        )


def _persist_prediction_snapshots(
    rows: Sequence[dict[str, Any]],
) -> DailyStepResult | None:
    if not rows:
        return None
    try:
        written = append_prediction_snapshots(rows)
        return DailyStepResult("snapshot", True, f"{written} prediction rows logged")
    except Exception as exc:
        logger.exception("Prediction snapshot write failed.")
        return DailyStepResult("snapshot", False, str(exc))


def run_daily_lol_workflow(
    config: DailyWorkflowConfig | None = None,
    *,
    schedule_fetcher: Callable[..., pd.DataFrame] = fetch_and_store_schedule,
    data_generator_factory: Callable[[], DataGenerator] = DataGenerator,
    train_fn: Callable[..., Path | None] = train_models,
    module_factory: Callable[[], LoLBetsModule] = LoLBetsModule,
    predictor_factory: Callable[[], Predictor] | None = None,
    lineup_refresher_factory: (Callable[[], PandaScoreLineupRefresher] | None) = None,
    market_search_factory: Callable[[], MarketAdapter] = PolymarketGammaAdapter,
    webhook_sender: Callable[
        [str, Sequence[str]], None
    ] = send_discord_webhook_messages,
) -> DailyWorkflowResult:
    """Run the daily LoL workflow and optionally send Discord webhook messages."""
    cfg = config or DailyWorkflowConfig()
    webhook_url = cfg.webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    steps: list[DailyStepResult] = []
    run_now, scheduled_for, fetch_window_days = _daily_time_window(cfg.horizon_hours)
    effective_config = {
        key: value for key, value in vars(cfg).items() if key != "webhook_url"
    }

    # The schedule fetch is an external dependency and must degrade like every
    # other step: a PandaScore outage should produce a FAILED step and a
    # report, never a stack trace. When the fetch fails but a stored schedule
    # exists, fall back to it (marked in the step detail).
    fetch_state: dict[str, pd.DataFrame] = {}

    def _fetch_schedule() -> str:
        try:
            fetch_state["schedule"] = schedule_fetcher(
                start_datetime=run_now,
                window_days=fetch_window_days,
                leagues=cfg.leagues,
                save_path=None if cfg.dry_run else SCHEDULE,
            )
        except Exception as exc:
            try:
                stored = pd.read_parquet(SCHEDULE)
            except Exception:
                raise exc from None
            fetch_state["schedule"] = stored
            return f"fetch failed ({exc}); using stored schedule"
        return "fetched"

    schedule_step = _step("schedule", _fetch_schedule)
    schedule = fetch_state.get("schedule", pd.DataFrame())
    schedule, lineup_step = _refresh_expected_lineups(
        schedule,
        schedule_available=schedule_step.ok,
        observed_at=run_now,
        refresher_factory=lineup_refresher_factory,
    )
    daily_schedule = filter_daily_schedule(
        schedule,
        now=run_now,
        horizon_hours=cfg.horizon_hours,
    )
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
        *format_schedule_messages(reportable_schedule),
    ]

    if not cfg.dry_run and schedule_step.ok:
        steps.extend(
            _run_mutating_steps(
                cfg,
                data_generator_factory=data_generator_factory,
                train_fn=train_fn,
                module_factory=module_factory,
                store=EvidenceStore(EVIDENCE_DB),
                scheduled_for=scheduled_for,
                effective_config=effective_config,
            )
        )

    should_predict = all(step.ok for step in steps) or (
        cfg.dry_run and not reportable_schedule.empty
    )
    snapshot_rows: list[dict[str, Any]] = []
    prediction_details: list[dict[str, Any]] = []
    if should_predict:
        prediction_messages, prediction_details = _build_prediction_messages(
            reportable_schedule,
            cfg=cfg,
            predictor_factory=predictor_factory,
            market_search_factory=market_search_factory,
            snapshot_sink=snapshot_rows,
        )
        messages.extend(prediction_messages)

    if not cfg.dry_run:
        snapshot_step = _persist_prediction_snapshots(snapshot_rows)
        if snapshot_step is not None:
            steps.append(snapshot_step)

    if not cfg.dry_run and schedule_step.ok:
        try:
            run_id = record_daily_evidence(
                store=EvidenceStore(EVIDENCE_DB),
                scheduled_for=scheduled_for,
                effective_config=effective_config,
                schedule=daily_schedule,
                snapshot_rows=snapshot_rows,
                steps=steps,
            )
            steps.append(
                DailyStepResult("evidence", True, f"canonical run recorded: {run_id}")
            )
        except Exception as exc:
            logger.exception("Canonical evidence write failed.")
            steps.append(DailyStepResult("evidence", False, str(exc)))

    messages[0] = format_step_summary(steps)

    report_paths = write_daily_report(
        cfg=cfg,
        steps=steps,
        schedule=reportable_schedule,
        excluded_fixtures=excluded_fixtures,
        prediction_details=prediction_details,
        messages=messages,
    )
    if webhook_url and not cfg.dry_run:
        webhook_sender(webhook_url, messages)
    return DailyWorkflowResult(
        schedule=reportable_schedule,
        messages=messages,
        steps=steps,
        excluded_fixtures=excluded_fixtures,
        prediction_details=prediction_details,
        report_paths=report_paths,
    )
