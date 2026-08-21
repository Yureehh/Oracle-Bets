"""Owner-selected Polymarket event review without the automated daily pipeline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from oracle_bets_core.evidence import EvidenceStore
from oracle_bets_core.league_selection import selected_leagues
from oracle_bets_core.markets import (
    MarketDataError,
    PolymarketClobClient,
    PolymarketEvent,
    PolymarketGammaAdapter,
    PolymarketMarket,
)
from oracle_bets_core.operations.paper_evidence import daily_position_exposure
from oracle_bets_core.paths import EVIDENCE_DB, REPORTS_DIR, SCHEDULE
from oracle_bets_core.pd import pd

from lol_bets.daily import (
    DailyStepResult,
    DailyWorkflowConfig,
    Predictor,
    _build_prediction_messages,
)
from lol_bets.data_generation.ingestion.schedule import (
    fixture_version,
    normalize_schedule_frame,
)
from lol_bets.inference.team_resolver import canonical_team_name
from lol_bets.operations.evidence import record_daily_evidence

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from oracle_bets_core.markets import OrderBookClient

_MARKET_TYPES_SHOWN = frozenset({"child_moneyline", "moneyline", "totals"})
_NON_TEAM_OUTCOMES = frozenset({"yes", "no", "over", "under"})
_SCHEDULE_MATCH_TOLERANCE = dt.timedelta(hours=6)
_LEAGUE_SLUG_OVERRIDES = {
    "world-championship": "WLDs",
    "worlds": "WLDs",
}


@dataclass(frozen=True)
class ManualMarketReviewResult:
    """One report pair plus the evidence created by an explicit owner review."""

    report_paths: tuple[Path, Path]
    fixtures: int
    predictions: int
    actionable_proposals: int
    evidence_run_id: str | None
    failures: tuple[dict[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.failures and self.predictions == self.fixtures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "report_json": str(self.report_paths[0]),
            "report_markdown": str(self.report_paths[1]),
            "fixtures": self.fixtures,
            "predictions": self.predictions,
            "actionable_proposals": self.actionable_proposals,
            "evidence_run_id": self.evidence_run_id,
            "failures": list(self.failures),
        }


def review_polymarket_events(
    urls: Sequence[str],
    *,
    publish: bool = False,
    gamma: PolymarketGammaAdapter | None = None,
    predictor_factory: Callable[[], Predictor] | None = None,
    clob_client_factory: Callable[[], OrderBookClient] = PolymarketClobClient,
    sleeper: Callable[[float], None] | None = None,
    report_dir: Path | None = None,
    store: EvidenceStore | None = None,
    now: dt.datetime | None = None,
) -> ManualMarketReviewResult:
    """Review exact events and optionally publish eligible evidence to Discord."""
    if not urls:
        raise ValueError("at least one Polymarket event URL is required")
    reviewed_at = _as_utc(now or dt.datetime.now(dt.UTC))
    adapter = gamma or PolymarketGammaAdapter()
    failures: list[dict[str, str]] = []
    events: list[PolymarketEvent] = []
    rows: list[dict[str, Any]] = []
    markets_by_fixture: dict[str, tuple[PolymarketMarket, ...]] = {}
    for url in dict.fromkeys(urls):
        try:
            event = adapter.event(url)
            row = _fixture_row(event, url=url)
            row = _enrich_from_stored_schedule(row)
        except Exception as exc:
            failures.append(
                {"url": url, "reason": type(exc).__name__, "detail": str(exc)}
            )
            continue
        events.append(event)
        start = cast("pd.Timestamp", pd.Timestamp(row["start_utc"]))
        reviewed_timestamp = cast("pd.Timestamp", pd.Timestamp(reviewed_at))
        if row["status"] != "not_started" or start <= reviewed_timestamp:
            failures.append(
                {
                    "url": url,
                    "reason": "event_not_open",
                    "detail": "fixture is closed, unavailable, or already started",
                }
            )
            continue
        rows.append(row)
        markets_by_fixture[str(row["match_key"])] = tuple(
            market
            for market in event.markets
            if market.sports_market_type in _MARKET_TYPES_SHOWN
        )

    schedule = pd.DataFrame(rows)
    snapshots: list[dict[str, Any]] = []
    market_reviews: list[dict[str, Any]] = []
    market_actions: list[dict[str, Any]] = []
    prediction_details: list[dict[str, Any]] = []
    messages: list[str] = []
    run_key = _manual_run_key(events)
    if not schedule.empty:
        kwargs: dict[str, Any] = {}
        if sleeper is not None:
            kwargs["market_sleeper"] = sleeper
        messages, prediction_details = _build_prediction_messages(
            schedule,
            cfg=DailyWorkflowConfig(dry_run=True, ai_review=False),
            predictor_factory=predictor_factory,
            market_search_factory=PolymarketGammaAdapter,
            snapshot_sink=snapshots,
            market_review_sink=market_reviews,
            market_action_sink=market_actions,
            clob_client_factory=clob_client_factory,
            market_run_key=run_key,
            existing_exposure_units=daily_position_exposure(
                store or EvidenceStore(EVIDENCE_DB), at=reviewed_at
            ),
            typed_markets_override=markets_by_fixture,
            **kwargs,
        )

    evidence_run_id = None
    if publish and not schedule.empty and snapshots:
        evidence_store = store or EvidenceStore(EVIDENCE_DB)
        evidence_run_id = record_daily_evidence(
            store=evidence_store,
            scheduled_for=reviewed_at,
            effective_config={"polymarket_urls": list(dict.fromkeys(urls))},
            schedule=schedule,
            snapshot_rows=snapshots,
            steps=(
                DailyStepResult(
                    "manual_market_review",
                    not failures,
                    f"reviewed {len(events)} exact Polymarket event(s)",
                ),
            ),
            market_actions=market_actions,
            run_type="manual_lol_market_review",
            run_key=run_key,
        )

    paths = _write_report(
        report_dir=report_dir or REPORTS_DIR / "market_reviews",
        reviewed_at=reviewed_at,
        publish=publish,
        urls=urls,
        events=events,
        schedule=schedule,
        failures=failures,
        prediction_details=prediction_details,
        snapshots=snapshots,
        market_reviews=market_reviews,
        market_actions=market_actions,
        messages=messages,
        evidence_run_id=evidence_run_id,
    )
    predicted = sum(
        detail.get("status") == "predicted" for detail in prediction_details
    )
    actionable = sum(
        action.get("state") == "paper_actionable" for action in market_actions
    )
    return ManualMarketReviewResult(
        report_paths=paths,
        fixtures=len(schedule),
        predictions=predicted,
        actionable_proposals=actionable,
        evidence_run_id=evidence_run_id,
        failures=tuple(failures),
    )


def _fixture_row(event: PolymarketEvent, *, url: str) -> dict[str, Any]:
    series_markets = [
        market
        for market in event.markets
        if market.sports_market_type == "moneyline"
        and market.game_number is None
        and len(market.outcomes) == 2  # noqa: PLR2004
    ]
    if not series_markets:
        raise MarketDataError("Polymarket event has no series-winner market")
    identities = {
        tuple(outcome.name.strip() for outcome in market.outcomes)
        for market in series_markets
    }
    if len(identities) != 1:
        raise MarketDataError("Polymarket event has conflicting team orientations")
    market = sorted(series_markets, key=lambda item: item.market_id)[0]
    team_a, team_b = identities.pop()
    if (
        team_a.casefold() in _NON_TEAM_OUTCOMES
        or team_b.casefold() in _NON_TEAM_OUTCOMES
        or team_a.casefold() == team_b.casefold()
    ):
        raise MarketDataError("Series-winner outcomes do not identify two teams")
    if market.event_start_time is None:
        raise MarketDataError("Series-winner market has no fixture start time")
    if market.best_of not in {1, 3, 5}:
        raise MarketDataError("Series-winner market has no supported BO format")
    league = _league_from_url(url)
    frame = normalize_schedule_frame(
        pd.DataFrame(
            [
                {
                    "provider": "polymarket",
                    "provider_match_id": event.event_id,
                    "league": league,
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_lineup_json": "[]",
                    "team_b_lineup_json": "[]",
                    "lineup_source": "historical_fallback",
                    "start_utc": market.event_start_time,
                    "best_of": market.best_of,
                    "status": (
                        "not_started"
                        if market.active
                        and not market.closed
                        and market.accepting_orders
                        else "closed"
                    ),
                    "market_query": event.title,
                }
            ]
        )
    )
    row = frame.iloc[0].to_dict()
    row["provider"] = "polymarket"
    row["match_key"] = f"polymarket:{event.event_id}"
    row["polymarket_event_url"] = event.url
    return row


def _league_from_url(url: str) -> str:
    parts = [
        unquote(part).strip().casefold()
        for part in urlsplit(url).path.split("/")
        if part
    ]
    try:
        route = parts[parts.index("league-of-legends") + 1]
    except (ValueError, IndexError) as exc:
        raise MarketDataError(
            "Polymarket URL has no League of Legends league route"
        ) from exc
    league = _LEAGUE_SLUG_OVERRIDES.get(route, route.upper())
    if league not in selected_leagues("research_all_supported"):
        raise MarketDataError(f"Unsupported LoL league route: {route}")
    return league


def _enrich_from_stored_schedule(row: dict[str, Any]) -> dict[str, Any]:
    """Prefer exact PandaScore fixture/lineup facts when the owner fetched schedule."""
    if not SCHEDULE.is_file():
        return row
    schedule = normalize_schedule_frame(pd.read_parquet(SCHEDULE))
    if schedule.empty:
        return row
    target_start = cast("pd.Timestamp", pd.Timestamp(row["start_utc"]))
    target_teams = {
        canonical_team_name(str(row["team_a"])).casefold(),
        canonical_team_name(str(row["team_b"])).casefold(),
    }
    candidates: list[pd.Series] = []
    for _, candidate in schedule.iterrows():
        candidate_teams = {
            canonical_team_name(str(candidate.get("team_a") or "")).casefold(),
            canonical_team_name(str(candidate.get("team_b") or "")).casefold(),
        }
        candidate_start = pd.Timestamp(candidate.get("start_utc"))
        if pd.isna(candidate_start):
            continue
        valid_start = cast("pd.Timestamp", candidate_start)
        if (
            str(candidate.get("league") or "") == row["league"]
            and candidate_teams == target_teams
            and abs(valid_start - target_start) <= _SCHEDULE_MATCH_TOLERANCE
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        return row
    enriched = candidates[0].to_dict()
    for field in (
        "league",
        "team_a",
        "team_b",
        "start_utc",
        "best_of",
        "status",
        "market_query",
        "polymarket_event_url",
    ):
        enriched[field] = row[field]
    enriched["fixture_version"] = fixture_version(enriched)
    return enriched


def _manual_run_key(events: Sequence[PolymarketEvent]) -> str:
    identity = "|".join(sorted(f"{event.event_id}:{event.slug}" for event in events))
    return "manual-lol-market-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _write_report(
    *,
    report_dir: Path,
    reviewed_at: dt.datetime,
    publish: bool,
    urls: Sequence[str],
    events: Sequence[PolymarketEvent],
    schedule: pd.DataFrame,
    failures: Sequence[dict[str, str]],
    prediction_details: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]],
    market_reviews: Sequence[dict[str, Any]],
    market_actions: Sequence[dict[str, Any]],
    messages: Sequence[str],
    evidence_run_id: str | None,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = reviewed_at.strftime("%Y%m%dT%H%M%S_%fZ")
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    payload = {
        "schema_version": 1,
        "generated_at": reviewed_at.isoformat(),
        "publish_requested": publish,
        "evidence_run_id": evidence_run_id,
        "urls": list(urls),
        "events": [
            {
                "event_id": event.event_id,
                "title": event.title,
                "slug": event.slug,
                "url": event.url,
                "markets": [
                    asdict(market)
                    for market in event.markets
                    if market.sports_market_type in _MARKET_TYPES_SHOWN
                ],
            }
            for event in events
        ],
        "fixtures": schedule.to_dict(orient="records"),
        "failures": list(failures),
        "prediction_details": list(prediction_details),
        "prediction_snapshots": list(snapshots),
        "market_reviews": list(market_reviews),
        "market_actions": list(market_actions),
        "messages": list(messages),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Manual LoL market review", ""]
    if failures:
        lines.extend(["## Link failures", ""])
        lines.extend(
            f"- `{failure['url']}`: {failure['reason']} — {failure['detail']}"
            for failure in failures
        )
        lines.append("")
    lines.extend(["## Predictions and quotes", ""])
    lines.extend(messages or ["No prediction was available."])
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Published: {'yes' if evidence_run_id else 'no'}",
            f"- Canonical run: `{evidence_run_id or 'none'}`",
            "- Polymarket access: read-only; no trading surface exists.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)
