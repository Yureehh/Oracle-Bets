"""Owner-selected LoL market reviews; market prices never enter a model."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit, urlunsplit

from oracle_bets_core.betting import (
    decimal_odds_from_probability,
    expected_edge,
    probability_from_decimal_odds,
)
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.league_selection import actionable_leagues
from oracle_bets_core.markets import (
    MarketDataError,
    PolymarketClobClient,
    PolymarketEvent,
    PolymarketGammaAdapter,
    PolymarketMarket,
)
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
from lol_bets.inference.team_resolver import canonical_team_name, team_name_variants
from lol_bets.operations.evidence import record_daily_evidence
from lol_bets.operations.market_actions import (
    active_strategy_readiness,
    evaluate_daily_market_actions,
    price_manual_lines,
)
from lol_bets.operations.market_strategies import rank_market_decisions

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from oracle_bets_core.markets import OrderBookClient

_NON_TEAM_OUTCOMES = frozenset({"yes", "no", "over", "under"})
_SCHEDULE_MATCH_TOLERANCE = dt.timedelta(hours=6)
_LEAGUE_SLUG_OVERRIDES = {
    "world-championship": "WLDs",
    "worlds": "WLDs",
}
_LEAGUE_LABELS = {"lrn": "Liga Regional Norte (LRN)"}
_ALLOWED_HOSTS = frozenset(
    {"polymarket.com", "www.polymarket.com", "thunderpick.io", "www.thunderpick.io"}
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualMarketReviewResult:
    """One report pair plus the evidence created by an explicit owner review."""

    report_paths: tuple[Path, Path]
    fixtures: int
    predictions: int
    comparisons: int
    evidence_run_id: str | None
    failures: tuple[dict[str, str], ...]
    ignored_links: tuple[dict[str, str], ...] = ()
    discord_message: str = ""
    market_ids: tuple[str, ...] = ()

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
            "comparisons": self.comparisons,
            "evidence_run_id": self.evidence_run_id,
            "failures": list(self.failures),
            "ignored_links": list(self.ignored_links),
            "discord_message": self.discord_message,
            "market_ids": list(self.market_ids),
        }


def review_polymarket_events(  # noqa: PLR0912, PLR0915
    urls: Sequence[str],
    *,
    manual_lines: Sequence[dict[str, Any]] = (),
    fixture_key: str | None = None,
    gamma: PolymarketGammaAdapter | None = None,
    predictor_factory: Callable[[], Predictor] | None = None,
    clob_client_factory: Callable[[], OrderBookClient] = PolymarketClobClient,
    report_dir: Path | None = None,
    store: EvidenceStore | None = None,
    now: dt.datetime | None = None,
) -> ManualMarketReviewResult:
    """Review one fixture from explicit owner-selected provider URLs."""
    started = time.perf_counter()
    normalized_urls = normalize_market_urls(urls)
    reviewed_at = _as_utc(now or dt.datetime.now(dt.UTC))
    adapter = gamma or PolymarketGammaAdapter()
    failures: list[dict[str, str]] = []
    ignored_links: list[dict[str, str]] = []
    events: list[PolymarketEvent] = []
    rows: list[dict[str, Any]] = []
    thunderpick_urls: list[str] = []
    unapproved_polymarket_urls: list[str] = []
    stored_schedule = _load_stored_schedule()
    markets_by_fixture: dict[str, tuple[PolymarketMarket, ...]] = {}
    provider_started = time.perf_counter()
    for url in normalized_urls:
        host = urlsplit(url).netloc.casefold()
        if "thunderpick" in host:
            thunderpick_urls.append(url)
            continue
        league = _league_from_url(url)
        if league not in actionable_leagues():
            label = _LEAGUE_LABELS.get(league.casefold(), league)
            ignored_links.append(
                {
                    "url": url,
                    "reason": "league_not_supported_or_trained",
                    "detail": f"{label} is not included in the trained or actionable LoL universe; no prediction was generated.",
                }
            )
            logger.info("Ignored owner market review for unsupported league %s", label)
            continue
        try:
            event = adapter.event(url)
            events.append(event)
            row = _fixture_row(event, league=league)
            approved_row = _enrich_from_stored_schedule(row, stored_schedule)
            if approved_row is None:
                unapproved_polymarket_urls.append(url)
                continue
            row = approved_row
        except Exception as exc:
            failures.append(
                {"url": url, "reason": type(exc).__name__, "detail": str(exc)}
            )
            continue
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
        markets_by_fixture[str(row["match_key"])] = event.markets
    if thunderpick_urls:
        selected_fixture = _resolve_thunderpick_fixture(
            thunderpick_urls[0],
            reviewed_at,
            fixture_key=fixture_key,
            schedule=stored_schedule,
        )
        if rows and (
            all(_url_matches_fixture(url, rows[0]) for url in thunderpick_urls)
            or (
                selected_fixture is not None
                and _same_fixture([rows[0], selected_fixture])
            )
        ):
            pass
        elif not rows and events:
            failures.append(
                {
                    "url": thunderpick_urls[0],
                    "reason": "fixture_mismatch",
                    "detail": "A provider link failed to resolve an approved fixture; no model inference was run.",
                }
            )
        elif not rows:
            if selected_fixture is not None:
                rows.append(selected_fixture)
            else:
                failures.append(
                    {
                        "url": thunderpick_urls[0],
                        "reason": "fixture_selection_required",
                        "detail": "Thunderpick is manual-only and the URL did not uniquely match the saved schedule; select the fixture in Discord.",
                    }
                )
        else:
            failures.append(
                {
                    "url": thunderpick_urls[0],
                    "reason": "fixture_mismatch",
                    "detail": "The Thunderpick link does not identify the same teams as the other submitted link.",
                }
            )
            rows.clear()
            markets_by_fixture.clear()
    if thunderpick_urls and manual_lines:
        manual_lines = tuple(
            line | {"url": line.get("url") or thunderpick_urls[0]}
            for line in manual_lines
        )
    if unapproved_polymarket_urls and not thunderpick_urls:
        failures.extend(
            {
                "url": url,
                "reason": "approved_fixture_required",
                "detail": (
                    "market metadata cannot authorize model inference; "
                    "fetch and approve the matching PandaScore fixture first"
                ),
            }
            for url in unapproved_polymarket_urls
        )
    fixture_resolution_seconds = time.perf_counter() - provider_started

    if len(rows) > 1 and not _same_fixture(rows):
        failures.append(
            {
                "url": ", ".join(normalized_urls),
                "reason": "fixture_mismatch",
                "detail": "Submitted links do not resolve to the same fixture.",
            }
        )
        rows.clear()
    elif len(rows) > 1:
        first = rows[0]
        fixture_key = str(first["match_key"])
        markets_by_fixture = {
            fixture_key: tuple(market for event in events for market in event.markets)
        }
        rows = [first]

    schedule = pd.DataFrame(rows)
    snapshots: list[dict[str, Any]] = []
    market_actions: list[dict[str, Any]] = []
    prediction_details: list[dict[str, Any]] = []
    run_key = _manual_run_key(events, reviewed_at=reviewed_at)
    inference_started = time.perf_counter()
    if not schedule.empty:
        _messages, prediction_details = _build_prediction_messages(
            schedule,
            cfg=DailyWorkflowConfig(dry_run=True),
            predictor_factory=predictor_factory,
            snapshot_sink=snapshots,
        )
    model_inference_seconds = time.perf_counter() - inference_started

    provider_started = time.perf_counter()
    readiness = (
        active_strategy_readiness()
        if not schedule.empty and (markets_by_fixture or manual_lines)
        else {}
    )
    if not schedule.empty and markets_by_fixture:
        market_actions.extend(
            evaluate_daily_market_actions(
                schedule=schedule,
                snapshot_rows=snapshots,
                markets=markets_by_fixture,
                clob_client=clob_client_factory(),
                model_healthy=True,
                strategy_readiness=readiness,
                run_key=run_key,
                clock=lambda: reviewed_at,
                rank=False,
            ).actions
        )
    provider_retrieval_seconds = time.perf_counter() - provider_started
    if not schedule.empty and manual_lines:
        market_actions.extend(
            price_manual_lines(
                fixture=schedule.iloc[0],
                snapshot_rows=snapshots,
                lines=manual_lines,
                reviewed_at=reviewed_at,
                strategy_readiness=readiness,
                rank=False,
            )
        )
    market_actions = rank_market_decisions(market_actions)

    evidence_run_id = None
    evidence_store = store or EvidenceStore(EVIDENCE_DB)
    if not schedule.empty and snapshots:
        evidence_run_id = record_daily_evidence(
            store=evidence_store,
            scheduled_for=reviewed_at,
            effective_config={"market_urls": list(normalized_urls)},
            schedule=schedule,
            snapshot_rows=snapshots,
            steps=(
                DailyStepResult(
                    "manual_market_review",
                    not failures,
                    f"reviewed {len(events)} Polymarket event(s) and {len(manual_lines)} manual line(s)",
                ),
            ),
            market_actions=market_actions,
            run_type="manual_lol_market_review",
            run_key=run_key,
        )

    candidate_refs = _candidate_refs(evidence_store, evidence_run_id)
    comparisons = provider_comparisons(market_actions)
    summary = discord_review_summary(
        schedule,
        snapshots,
        comparisons,
        ignored_links=ignored_links,
        failures=failures,
        unsupported_contracts=_unsupported_contract_count(market_actions),
        warnings=_review_warnings(market_actions),
    )
    paths = _write_report(
        report_dir=report_dir or REPORTS_DIR / "market_reviews",
        reviewed_at=reviewed_at,
        urls=normalized_urls,
        events=events,
        schedule=schedule,
        failures=failures,
        ignored_links=ignored_links,
        prediction_details=prediction_details,
        snapshots=snapshots,
        market_actions=market_actions,
        comparisons=comparisons,
        evidence_run_id=evidence_run_id,
        bet_references=candidate_refs,
        timings={
            "fixture_resolution_seconds": fixture_resolution_seconds,
            "model_inference_seconds": model_inference_seconds,
            "provider_retrieval_seconds": provider_retrieval_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    )
    logger.info(
        "Market review completed: fixtures=%d predictions=%d comparisons=%d elapsed=%.2fs",
        len(schedule),
        sum(detail.get("status") == "predicted" for detail in prediction_details),
        len(comparisons),
        time.perf_counter() - started,
    )
    predicted = sum(
        detail.get("status") == "predicted" for detail in prediction_details
    )
    return ManualMarketReviewResult(
        report_paths=paths,
        fixtures=len(schedule),
        predictions=predicted,
        comparisons=len(comparisons),
        evidence_run_id=evidence_run_id,
        failures=tuple(failures),
        ignored_links=tuple(ignored_links),
        discord_message=summary,
        market_ids=tuple(item["market_id"] for item in candidate_refs),
    )


def _fixture_row(event: PolymarketEvent, *, league: str) -> dict[str, Any]:
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
    return _LEAGUE_SLUG_OVERRIDES.get(route, route.upper())


def _load_stored_schedule() -> pd.DataFrame:
    if not SCHEDULE.is_file():
        return pd.DataFrame()
    return normalize_schedule_frame(pd.read_parquet(SCHEDULE))


def _enrich_from_stored_schedule(
    row: dict[str, Any], schedule: pd.DataFrame
) -> dict[str, Any] | None:
    """Prefer exact PandaScore fixture/lineup facts when the owner fetched schedule."""
    if schedule.empty:
        return None
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
        return None
    enriched = candidates[0].to_dict()
    for field in ("market_query", "polymarket_event_url"):
        enriched[field] = row[field]
    enriched["fixture_version"] = fixture_version(enriched)
    return enriched


def _manual_run_key(
    events: Sequence[PolymarketEvent], *, reviewed_at: dt.datetime
) -> str:
    identity = "|".join(sorted(f"{event.event_id}:{event.slug}" for event in events))
    payload = f"{reviewed_at.isoformat()}|{identity}"
    return "manual-lol-market-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def discord_review_summary(
    schedule: pd.DataFrame,
    snapshots: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    *,
    ignored_links: Sequence[dict[str, str]] = (),
    failures: Sequence[dict[str, str]] = (),
    unsupported_contracts: int = 0,
    warnings: Sequence[str] = (),
) -> str:
    """Build one compact owner review without proposal or recovery boilerplate."""
    if schedule.empty:
        lines = ["**LoL market review**"]
        for item in ignored_links:
            lines.extend(
                (
                    f"**Ignored:** {item['detail'].split(' is not included', 1)[0]}",
                    f"• {item['detail']}",
                )
            )
        lines.extend(f"• Failed: {item['detail']}" for item in failures)
        return "\n".join(lines)

    lines: list[str] = []
    for _, fixture in schedule.iterrows():
        fixture_key = str(fixture.get("match_key") or "")
        fixture_rows = [
            row
            for row in snapshots
            if not fixture_key or str(row.get("source_match_key") or "") == fixture_key
        ]
        team_a = str(fixture.get("team_a") or "")
        team_b = str(fixture.get("team_b") or "")
        start = pd.Timestamp(fixture.get("start_utc"))
        stamp = (
            "time unknown"
            if pd.isna(start)
            else cast("pd.Timestamp", start).strftime("%Y-%m-%d %H:%M UTC")
        )
        lines.extend(
            (
                f"**{team_a} vs {team_b}**",
                f"{stamp} · {fixture.get('league')} · BO{fixture.get('best_of')}",
                "",
                "**Model**",
            )
        )
        for market, label in (
            ("series_winner", "Series"),
            ("map_winner", "Map 1 research"),
        ):
            probabilities = [
                (str(row.get("selection") or ""), float(row["model_value"]))
                for row in fixture_rows
                if row.get("market") == market
            ]
            if probabilities:
                lines.append(
                    f"• **{label}:** "
                    + " · ".join(
                        f"{selection} {probability:.1%} (fair {1 / probability:.2f})"
                        for selection, probability in probabilities
                        if probability > 0
                    )
                )
        prop_labels = {
            "gamelength_mean": ("Length", "m"),
            "total_kills_mean": ("Kills", ""),
            "total_towers_mean": ("Towers", ""),
        }
        props = [
            f"• **{label}:** {float(row['model_value']):.1f}{suffix}"
            for row in fixture_rows
            if row.get("market") in prop_labels
            for label, suffix in (prop_labels[str(row["market"])],)
        ]
        if props:
            lines.extend(("", "**Props · research**", *props))
        fixture_comparisons = [
            comparison
            for comparison in comparisons
            if not fixture_key
            or str(comparison.get("fixture_key") or "") == fixture_key
        ]
        if fixture_comparisons:
            ranked = sorted(
                fixture_comparisons,
                key=lambda row: float(row.get("point_ev") or float("-inf")),
                reverse=True,
            )
            lines.extend(("", "**Best comparisons**"))
            for comparison in ranked[:5]:
                probability = comparison.get("model_probability")
                fair = 1 / float(probability) if probability else None
                edge = comparison.get("point_ev")
                target = str(comparison.get("target") or "market").replace("_", " ")
                lines.append(
                    f"• **{comparison.get('selection')} · {target}:** "
                    f"{str(comparison.get('best_provider') or '').title()} "
                    f"{float(comparison['best_decimal_odds']):.2f}"
                    + (f" · fair {fair:.2f}" if fair else "")
                    + (f" · EV {float(edge):+.1%}" if edge is not None else "")
                )
            positive = sum(
                float(row.get("point_ev") or 0) > 0 for row in fixture_comparisons
            )
            lines.extend(
                (
                    "",
                    "**Result**",
                    f"• {positive} positive-EV model-backed comparison(s).",
                )
            )
        if unsupported_contracts:
            lines.append(
                f"• {unsupported_contracts} unsupported contract(s) retained in JSON."
            )
        if warnings:
            lines.extend(("", "**Warnings**", *(f"• {item}" for item in warnings[:3])))
    return "\n".join(lines)[:1900]


def provider_comparisons(
    actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group only semantically identical provider outcomes and select best odds."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for action in actions:
        odds = action.get("decimal_odds")
        if odds is None or action.get("hard_blocks"):
            continue
        semantic_fingerprint = str(action.get("semantic_fingerprint") or "")
        if not semantic_fingerprint:
            continue
        key = (str(action.get("fixture_key") or ""), semantic_fingerprint)
        grouped.setdefault(key, []).append(action)

    output: list[dict[str, Any]] = []
    for (fixture_key, semantic_fingerprint), rows in grouped.items():
        best = max(rows, key=lambda row: float(row["decimal_odds"]))
        probability = best.get("probability")
        odds = float(best["decimal_odds"])
        output.append(
            {
                "fixture_key": fixture_key,
                "semantic_fingerprint": semantic_fingerprint,
                "semantic_key": best.get("semantic_key"),
                "target": best.get("target"),
                "game_number": best.get("game_number"),
                "selection": best.get("selection"),
                "selection_key": (best.get("semantic_key") or {}).get("selection"),
                "line": best.get("line"),
                "model_probability": probability,
                "fair_decimal_odds": decimal_odds_from_probability(float(probability))
                if probability
                else None,
                "best_provider": best.get("provider") or "polymarket",
                "best_decimal_odds": odds,
                "implied_probability": probability_from_decimal_odds(odds),
                "point_ev": expected_edge(odds, float(probability))
                if probability is not None
                else None,
                "providers": [
                    {
                        "provider": row.get("provider") or "polymarket",
                        "market_id": row.get("market_id"),
                        "decimal_odds": float(row["decimal_odds"]),
                        "implied_probability": probability_from_decimal_odds(
                            float(row["decimal_odds"])
                        ),
                        "point_ev": row.get("point_edge"),
                    }
                    for row in sorted(
                        rows, key=lambda row: float(row["decimal_odds"]), reverse=True
                    )
                ],
            }
        )
    return output


def _write_report(
    *,
    report_dir: Path,
    reviewed_at: dt.datetime,
    urls: Sequence[str],
    events: Sequence[PolymarketEvent],
    schedule: pd.DataFrame,
    failures: Sequence[dict[str, str]],
    ignored_links: Sequence[dict[str, str]],
    prediction_details: Sequence[dict[str, Any]],
    snapshots: Sequence[dict[str, Any]],
    market_actions: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    evidence_run_id: str | None,
    bet_references: Sequence[dict[str, str]],
    timings: dict[str, float],
) -> tuple[Path, Path]:
    persistence_started = time.perf_counter()
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = reviewed_at.strftime("%Y%m%dT%H%M%S_%fZ")
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    payload = {
        "schema_version": 4,
        "workflow": "owner_market_review_v4",
        "generated_at": reviewed_at.isoformat(),
        "evidence_run_id": evidence_run_id,
        "input_links": list(urls),
        "ignored_links": list(ignored_links),
        "provider_results": [
            {
                "provider": "polymarket",
                "event_id": event.event_id,
                "title": event.title,
                "slug": event.slug,
                "url": event.url,
                "contract_count": len(event.markets),
                "market_failures": [
                    asdict(failure) for failure in event.market_failures
                ],
            }
            for event in events
        ]
        + (
            [
                {
                    "provider": "thunderpick",
                    "access": "manual_only",
                    "line_count": sum(
                        action.get("provider") == "thunderpick"
                        for action in market_actions
                    ),
                }
            ]
            if any("thunderpick.io" in url for url in urls)
            else []
        ),
        "fixture": schedule.iloc[0].to_dict() if not schedule.empty else None,
        "roster_evidence": _roster_evidence(schedule),
        "failures": list(failures),
        "prediction_details": list(prediction_details),
        "model_forecasts": list(snapshots),
        "contracts": [
            _contract_payload(event, market)
            for event in events
            for market in event.markets
        ]
        + _manual_contract_payloads(market_actions),
        "quotes": [_compact_action(action) for action in market_actions],
        "comparisons": list(comparisons),
        "warnings": sorted(
            {
                str(warning)
                for action in market_actions
                for warning in action.get("warnings") or []
            }
        ),
        "timings": timings,
        "bet_references": list(bet_references),
    }
    lines = ["# LoL market review", ""]
    if ignored_links:
        lines.extend(["## Ignored links", ""])
        lines.extend(f"- **Ignored:** {item['detail']}" for item in ignored_links)
        lines.append("")
    if failures:
        lines.extend(["## Link failures", ""])
        lines.extend(
            f"- `{failure['url']}`: {failure['reason']} — {failure['detail']}"
            for failure in failures
        )
        lines.append("")
    lines.extend(["## Review", ""])
    lines.extend(
        discord_review_summary(
            schedule,
            snapshots,
            comparisons,
            ignored_links=ignored_links,
            failures=failures,
            unsupported_contracts=_unsupported_contract_count(market_actions),
            warnings=_review_warnings(market_actions),
        ).splitlines()
    )
    if comparisons:
        lines.extend(
            [
                "",
                "## Supported comparisons",
                "",
                "| Provider | Market | Selection | Model | Odds | Fair | EV |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in sorted(
            comparisons,
            key=lambda row: float(row.get("point_ev") or float("-inf")),
            reverse=True,
        )[:20]:
            probability = comparison.get("model_probability")
            fair = 1 / float(probability) if probability else None
            lines.append(
                f"| {str(comparison.get('best_provider') or '').title()} | {str(comparison.get('target') or 'unknown').replace('_', ' ')} | {comparison.get('selection')} | "
                f"{float(probability):.1%} | {float(comparison['best_decimal_odds']):.2f} | {fair:.2f} | {float(comparison.get('point_ev') or 0):+.1%} |"
                if probability and fair
                else f"| {str(comparison.get('best_provider') or '').title()} | {comparison.get('target')} | {comparison.get('selection')} | n/a | {float(comparison['best_decimal_odds']):.2f} | n/a | n/a |"
            )
    contract_types: dict[str, int] = {}
    for event in events:
        for market in event.markets:
            key = str(market.sports_market_type or "unknown")
            contract_types[key] = contract_types.get(key, 0) + 1
    lines.extend(["", "## Provider coverage", ""])
    lines.extend(
        f"- **{name}:** {count}" for name, count in sorted(contract_types.items())
    )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Review ID: `{evidence_run_id or 'none'}`",
            f"- Supported comparisons: {len(comparisons)}",
            "- Polymarket access: read-only; no trading surface exists.",
        ]
    )
    payload["timings"]["report_persistence_seconds"] = (
        time.perf_counter() - persistence_started
    )
    payload["timings"]["total_seconds"] += payload["timings"][
        "report_persistence_seconds"
    ]
    json_temp = json_path.with_suffix(".json.tmp")
    markdown_temp = markdown_path.with_suffix(".md.tmp")
    try:
        json_temp.write_text(
            json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_temp.replace(json_path)
        markdown_temp.replace(markdown_path)
    except Exception:
        json_temp.unlink(missing_ok=True)
        markdown_temp.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def normalize_market_urls(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize one or two allowlisted provider URLs in owner-supplied order."""
    raw = [
        token
        for value in values
        for token in re.split(r"[,\s]+", str(value).strip())
        if token
    ]
    output: list[str] = []
    for value in raw:
        parsed = urlsplit(value)
        host = parsed.netloc.casefold()
        if parsed.scheme != "https" or host not in _ALLOWED_HOSTS or parsed.username:
            raise MarketDataError(
                "Market links must be HTTPS Polymarket or Thunderpick URLs."
            )
        normalized = urlunsplit(
            ("https", host.removeprefix("www."), parsed.path.rstrip("/"), "", "")
        )
        if normalized not in output:
            output.append(normalized)
    if not output or len(output) > 2:  # noqa: PLR2004
        raise MarketDataError("Submit one or two market links.")
    providers = [urlsplit(url).netloc for url in output]
    if len(providers) != len(set(providers)):
        raise MarketDataError(
            "Submit at most one Polymarket link and one Thunderpick link."
        )
    return tuple(output)


def parse_manual_lines_text(value: str) -> list[dict[str, Any]]:
    """Parse `target | selection | odds | line | game` owner entries."""
    output: list[dict[str, Any]] = []
    for number, raw in enumerate(value.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split("|")]
        if len(fields) not in {3, 4, 5}:
            raise MarketDataError(
                f"Thunderpick line {number} must use target | selection | odds | line | game."
            )
        target, selection, odds_text, *optional = fields
        try:
            odds = float(odds_text)
            line = float(optional[0]) if optional and optional[0] else None
            game_number = (
                int(optional[1]) if len(optional) > 1 and optional[1] else None
            )
        except ValueError as exc:
            raise MarketDataError(
                f"Thunderpick line {number} has invalid numeric values."
            ) from exc
        if not math.isfinite(odds) or odds <= 1:
            raise MarketDataError(
                f"Thunderpick line {number} decimal odds must be finite and exceed 1."
            )
        if line is not None and not math.isfinite(line):
            raise MarketDataError(f"Thunderpick line {number} line must be finite.")
        output.append(
            {
                "target": target,
                "selection": selection,
                "decimal_odds": odds,
                "line": line,
                "game_number": game_number,
            }
        )
    if len(output) > 10:  # noqa: PLR2004
        raise MarketDataError("Enter at most ten Thunderpick lines per review.")
    return output


def _same_fixture(rows: Sequence[dict[str, Any]]) -> bool:
    first = rows[0]
    teams = {
        canonical_team_name(str(first["team_a"])).casefold(),
        canonical_team_name(str(first["team_b"])).casefold(),
    }
    raw_start = pd.Timestamp(first["start_utc"])
    if pd.isna(raw_start):
        return False
    start = cast("pd.Timestamp", raw_start)
    best_of = int(first["best_of"])
    for row in rows[1:]:
        row_start = pd.Timestamp(row["start_utc"])
        if pd.isna(row_start):
            return False
        if (
            {
                canonical_team_name(str(row["team_a"])).casefold(),
                canonical_team_name(str(row["team_b"])).casefold(),
            }
            != teams
            or abs(cast("pd.Timestamp", row_start) - start) > _SCHEDULE_MATCH_TOLERANCE
            or int(row["best_of"]) != best_of
        ):
            return False
    return True


def _resolve_thunderpick_fixture(
    url: str,
    reviewed_at: dt.datetime,
    *,
    fixture_key: str | None = None,
    schedule: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    candidates = thunderpick_fixture_options(
        url, reviewed_at=reviewed_at, schedule=schedule
    )
    if fixture_key is not None:
        selected = next(
            (row for row in candidates if str(row.get("match_key")) == fixture_key),
            None,
        )
        return (
            {key: value for key, value in selected.items() if key != "_url_match"}
            if selected
            else None
        )
    matches = [row for row in candidates if bool(row.get("_url_match", False))]
    if len(matches) != 1:
        return None
    return {key: value for key, value in matches[0].items() if key != "_url_match"}


def thunderpick_fixture_options(
    url: str,
    *,
    reviewed_at: dt.datetime | None = None,
    schedule: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return actionable saved fixtures; never contact Thunderpick."""
    now = _as_utc(reviewed_at or dt.datetime.now(dt.UTC))
    schedule = _load_stored_schedule() if schedule is None else schedule
    candidates: list[dict[str, Any]] = []
    for _, row in schedule.iterrows():
        start = pd.Timestamp(row.get("start_utc"))
        if (
            str(row.get("league") or "") not in actionable_leagues()
            or pd.isna(start)
            or cast("pd.Timestamp", start) <= pd.Timestamp(now)
        ):
            continue
        candidates.append(
            row.to_dict() | {"_url_match": _url_matches_fixture(url, row)}
        )
    return sorted(
        candidates,
        key=lambda row: (
            not bool(row["_url_match"]),
            pd.Timestamp(row["start_utc"]),
        ),
    )


def _url_matches_fixture(url: str, row: Any) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", unquote(urlsplit(url).path).casefold()))
    return all(
        any(
            set(re.findall(r"[a-z0-9]+", alias.casefold())) <= tokens
            for alias in team_name_variants(name)
        )
        for name in (str(row.get("team_a") or ""), str(row.get("team_b") or ""))
    )


def _candidate_refs(
    store: EvidenceStore,
    run_id: str | None,
) -> list[dict[str, str]]:
    if run_id is None:
        return []
    return [
        {
            "market_id": str(row["id"]),
            "provider": str(row["provider"]),
            "provider_market_id": str(row["provider_market_id"]),
            "selection_id": str(row.get("provider_selection_id") or ""),
        }
        for row in store.list(EvidenceTable.MARKET_CANDIDATES)
        if row["run_id"] == run_id
    ]


def _roster_evidence(schedule: pd.DataFrame) -> dict[str, Any] | None:
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    return {
        "source": row.get("lineup_source"),
        "team_a": row.get("team_a_roster_evidence"),
        "team_b": row.get("team_b_roster_evidence"),
    }


def _contract_payload(
    event: PolymarketEvent,
    market: PolymarketMarket,
) -> dict[str, Any]:
    return {
        "provider": "polymarket",
        "event_id": event.event_id,
        "market_id": market.market_id,
        "question": market.question,
        "target_type": market.sports_market_type or "unknown",
        "game_number": market.game_number,
        "line": market.total_line,
        "outcomes": [asdict(outcome) for outcome in market.outcomes],
        "url": market.url,
        "resolution_source": market.resolution_source,
    }


def _manual_contract_payloads(
    actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "provider": "thunderpick",
            "market_id": action.get("market_id"),
            "target_type": action.get("target"),
            "game_number": action.get("game_number"),
            "line": action.get("line"),
            "selection": action.get("selection"),
            "url": action.get("market_url"),
            "resolution_source": "owner_entered",
        }
        for action in actions
        if action.get("provider") == "thunderpick"
    ]


def _unsupported_contract_count(actions: Sequence[dict[str, Any]]) -> int:
    return len(
        {
            str(action.get("market_id"))
            for action in actions
            if action.get("target") == "unknown"
        }
    )


def _review_warnings(actions: Sequence[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(warning).replace("_", " ")
            for action in actions
            for warning in action.get("warnings") or []
            if warning != "no_model_target"
        }
    )


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in action.items() if key != "observations"}
    compact["provider"] = str(action.get("provider") or "polymarket")
    compact["observations"] = [
        {
            key: observation.get(key)
            for key in (
                "observed_at",
                "book_hash",
                "decimal_odds",
                "minimum_order_size",
                "requested_shares",
                "filled_shares",
                "hypothetical_cost",
                "complete",
            )
        }
        for observation in action.get("observations") or []
    ]
    return compact


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
