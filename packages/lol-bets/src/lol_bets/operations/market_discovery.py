"""Public listing observations, never executable quotes or entry decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from oracle_bets_core.evidence import EvidenceTable
from oracle_bets_core.markets import MarketFixture, PolymarketGammaAdapter, match_market
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from oracle_bets_core.evidence import EvidenceStore


def discover_markets(
    schedule: pd.DataFrame, *, adapter: Any = None
) -> list[dict[str, Any]]:
    """Search bounded public metadata; exact review and executable price are deferred."""
    adapter = adapter if adapter is not None else PolymarketGammaAdapter()
    observed_at = datetime.now(UTC).isoformat()
    observations = []
    for row in schedule.to_dict("records"):
        key = str(
            row.get("match_key")
            or f"{row['league']}|{row['team_a']}|{row['team_b']}|{row['start_utc']}"
        )
        observation = {
            "provider": "polymarket",
            "fixture_key": key,
            "observed_at": observed_at,
            "opening_time_known": False,
            "executable_quote_verified": False,
            "candidates": [],
        }
        try:
            start = pd.Timestamp(row["start_utc"])
            if pd.isna(start) or start.tzinfo is None:
                raise ValueError("fixture start must be an aware timestamp")  # noqa: TRY301
            fixture = MarketFixture(
                fixture_id=key,
                competition_names=(str(row["league"]),),
                team_a_id=str(row["team_a"]),
                team_b_id=str(row["team_b"]),
                team_a_names=(str(row["team_a"]),),
                team_b_names=(str(row["team_b"]),),
                start_time=datetime.fromisoformat(start.isoformat()).astimezone(UTC),
                best_of=int(row["best_of"]),
            )
        except (ValueError, TypeError, KeyError) as exc:
            observation.update(status="fixture_unavailable", reason=str(exc))
            observations.append(observation)
            continue
        try:
            markets = adapter.search_markets(
                str(
                    row.get("market_query")
                    or f"{row['league']} {row['team_a']} {row['team_b']}"
                ),
                limit=25,
            )
        except Exception as exc:
            observation.update(status="provider_unavailable", reason=str(exc))
            observations.append(observation)
            continue
        observation["observed_at"] = datetime.now(UTC).isoformat()
        for market in markets:
            assessment = match_market(fixture, market)
            observation["candidates"].append(
                {
                    "market_id": market.market_id,
                    "question": market.question,
                    "market_slug": market.slug,
                    "matched": assessment.matched,
                    "reasons": list(assessment.reasons),
                    "warnings": list(assessment.warnings),
                    "status": "needs_exact_review",
                    "executable_quote_verified": False,
                }
            )
        observation["status"] = "needs_exact_review" if markets else "not_seen"
        observations.append(observation)
    observations.extend(
        {
            "provider": "thunderpick",
            "fixture_key": str(row.get("match_key") or ""),
            "observed_at": observed_at,
            "status": "manual_discovery_required",
            "opening_time_known": False,
            "executable_quote_verified": False,
            "candidates": [],
        }
        for row in schedule.to_dict("records")
    )
    return observations


def record_market_observations(
    store: EvidenceStore,
    *,
    run_id: str,
    observations: list[dict[str, Any]],
    observed_at: datetime,
) -> None:
    """Append observations and recover first observed listings without inferring opening."""
    first_seen = {}
    for event in store.list(EvidenceTable.RUN_EVENTS):
        if event["event_type"] != "market_discovery":
            continue
        payload = json.loads(event["payload_json"])
        for candidate in payload.get("candidates", []):
            key = (payload["provider"], candidate["market_id"])
            at = candidate.get("first_observed_at") or payload["observed_at"]
            first_seen[key] = min(at, first_seen.get(key, at))
    for observation in observations:
        observation.setdefault("observed_at", observed_at.isoformat())
        observation["opening_time_known"] = False
        for candidate in observation.get("candidates", []):
            key = (observation["provider"], candidate["market_id"])
            candidate["first_observed_at"] = first_seen.setdefault(
                key, observation["observed_at"]
            )
        identity = f"{run_id}|{observation['provider']}|{observation['fixture_key']}|{observed_at.isoformat()}"
        event_id = "market-discovery-" + hashlib.sha256(identity.encode()).hexdigest()
        if store.get(EvidenceTable.RUN_EVENTS, event_id) is None:
            store.append(
                EvidenceTable.RUN_EVENTS,
                {
                    "id": event_id,
                    "run_id": run_id,
                    "event_at": observed_at,
                    "event_type": "market_discovery",
                    "status": observation["status"],
                    "idempotency_key": event_id,
                    "payload_json": observation,
                },
            )
