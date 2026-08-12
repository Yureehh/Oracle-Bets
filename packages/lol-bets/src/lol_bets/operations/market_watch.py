"""Hourly read-only Polymarket observations for entry-timing research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.league_selection import actionable_leagues
from oracle_bets_core.markets import (
    MarketFixture,
    PolymarketClobClient,
    PolymarketGammaAdapter,
    select_best_market,
)
from oracle_bets_core.paths import REPORTS_DIR, SCHEDULE
from oracle_bets_core.pd import pd

from lol_bets.inference.team_resolver import team_name_variants
from lol_bets.operations.market_actions import LOL_RESOLUTION_RULE_TERMS

HORIZON_HOURS = (336, 168, 72, 48, 36, 24, 12, 6, 1, 0)
HOURS_PER_DAY = 24
HORIZON_TOLERANCE_HOURS = 0.75

if TYPE_CHECKING:
    from pathlib import Path


def observe_winner_markets(  # noqa: PLR0915
    *,
    store: EvidenceStore | None = None,
    schedule_path: Path = SCHEDULE,
    gamma: PolymarketGammaAdapter | None = None,
    clob: PolymarketClobClient | None = None,
    observed_at: datetime | None = None,
    report_root: Path = REPORTS_DIR / "market_watch",
) -> dict[str, Any]:
    """Record one independent market observation without pricing or decisions."""
    now = observed_at or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
        raise ValueError("market-watch observed_at must be timezone-aware UTC")
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule artifact missing: {schedule_path}")
    evidence = store or EvidenceStore()
    evidence.initialize_schema()
    gamma_client = gamma or PolymarketGammaAdapter()
    clob_client = clob or PolymarketClobClient()
    seen_market_ids = _seen_winner_market_ids(evidence)
    schedule = pd.read_parquet(schedule_path)
    allowed = set(actionable_leagues())
    observations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    run_id = _stable_id("market-watch", now.isoformat())
    records: list[tuple[EvidenceTable, dict[str, Any]]] = [
        (
            EvidenceTable.RUNS,
            {
                "id": run_id,
                "run_type": "lol_market_watch",
                "started_at": now,
                "status": "completed",
                "idempotency_key": run_id,
                "payload_json": {"read_only": True, "trading_surface": False},
            },
        )
    ]
    for _, row in schedule.iterrows():
        league = str(row.get("league") or "").strip()
        team_a = str(row.get("team_a") or "").strip()
        team_b = str(row.get("team_b") or "").strip()
        match_key = str(row.get("match_key") or "").strip()
        if league not in allowed or not team_a or not team_b or not match_key:
            continue
        try:
            start_time = _fixture_start(row["start_utc"])
            if start_time <= now:
                continue
            fixture = MarketFixture(
                fixture_id=match_key,
                competition_names=(league,),
                team_a_id=team_a,
                team_b_id=team_b,
                team_a_names=tuple(team_name_variants(team_a)),
                team_b_names=tuple(team_name_variants(team_b)),
                start_time=start_time,
                best_of=int(row.get("best_of") or 1),
                resolution_rule_terms=LOL_RESOLUTION_RULE_TERMS,
            )
            markets = gamma_client.search_markets(
                f"{team_a} {team_b} League of Legends", limit=50
            )
            selection = select_best_market(fixture, markets)
            if selection.selected_market_id is None:
                failures.append({"match_key": match_key, "reason": "market_not_found"})
                continue
            market = next(
                item
                for item in markets
                if item.market_id == selection.selected_market_id
            )
            books: list[dict[str, Any]] = []
            for outcome in market.outcomes:
                try:
                    book = clob_client.get_order_book(outcome.token_id)
                    books.append(
                        {
                            "outcome": outcome.name,
                            "token_id": outcome.token_id,
                            "displayed_price": outcome.displayed_price,
                            "book": asdict(book),
                        }
                    )
                except Exception as error:
                    books.append(
                        {
                            "outcome": outcome.name,
                            "token_id": outcome.token_id,
                            "error": type(error).__name__,
                        }
                    )
            hours = (start_time - now).total_seconds() / 3600
            payload = {
                "fixture_id": match_key,
                "league": league,
                "team_a": team_a,
                "team_b": team_b,
                "start_time": start_time,
                "hours_to_start": hours,
                "horizon": _horizon_label(hours),
                "first_seen": market.market_id not in seen_market_ids,
                "market_id": market.market_id,
                "market_url": market.url,
                "resolution_source": market.resolution_source,
                "books": books,
                "read_only": True,
            }
            observations.append(payload)
            seen_market_ids.add(market.market_id)
            snapshot_id = _stable_id(
                "market-snapshot", f"{run_id}|{match_key}|{market.market_id}"
            )
            records.append(
                (
                    EvidenceTable.SOURCE_SNAPSHOTS,
                    {
                        "id": snapshot_id,
                        "run_id": run_id,
                        "provider": "polymarket",
                        "source_type": "winner_market_watch",
                        "observed_at": now,
                        "source_uri": market.url,
                        "schema_fingerprint": "winner-market-watch-v1",
                        "idempotency_key": snapshot_id,
                        "payload_json": payload,
                    },
                )
            )
        except Exception as error:
            failures.append({"match_key": match_key, "reason": type(error).__name__})
    evidence.append_transaction(records)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "observed_at": now,
        "read_only": True,
        "observations": observations,
        "failures": failures,
    }
    json_path, markdown_path = _write_report(report, root=report_root, at=now)
    return report | {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


def _horizon_label(hours: float) -> str | None:
    nearest = min(HORIZON_HOURS, key=lambda value: abs(value - hours))
    if abs(nearest - hours) > HORIZON_TOLERANCE_HOURS:
        return None
    if nearest == 0:
        return "close"
    if nearest >= HOURS_PER_DAY and nearest % HOURS_PER_DAY == 0:
        return f"{nearest // HOURS_PER_DAY}d"
    return f"{nearest}h"


def _seen_winner_market_ids(store: EvidenceStore) -> set[str]:
    seen: set[str] = set()
    for row in store.list(EvidenceTable.SOURCE_SNAPSHOTS):
        if row.get("source_type") != "winner_market_watch":
            continue
        try:
            payload = json.loads(row["payload_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        market_id = str(payload.get("market_id") or "").strip()
        if market_id:
            seen.add(market_id)
    return seen


def _fixture_start(value: Any) -> datetime:
    start = pd.Timestamp(value)
    if pd.isna(start):
        raise ValueError("fixture start time is missing")
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    parsed = start.tz_convert("UTC").to_pydatetime()
    if not isinstance(parsed, datetime):
        raise TypeError("fixture start time is invalid")
    return parsed


def _write_report(
    payload: dict[str, Any], *, root: Path, at: datetime
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stem = at.strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lines = ["# LoL market watch", "", f"Observed: {at.isoformat()}"]
    lines.extend(
        (
            f"- {item['team_a']} vs {item['team_b']}: "
            f"{item['hours_to_start']:.1f}h ({item['horizon']})"
        )
        for item in payload["observations"]
    )
    if payload["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(
            f"- {item['match_key']}: {item['reason']}" for item in payload["failures"]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"
