"""Normalize LoL players, teams, leagues, series, and maps into evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.identity import EntityType, canonical_identity_id
from oracle_bets_core.pd import pd

if TYPE_CHECKING:
    from collections.abc import Iterable

SERIES_GAP = pd.Timedelta(hours=6)


@dataclass(frozen=True)
class IdentityGraphSyncResult:
    discovered_identities: int
    discovered_links: int
    added_identities: int
    added_links: int


def sync_schedule_identity_graph(
    store: EvidenceStore,
    schedule: pd.DataFrame,
    *,
    observed_at: datetime,
) -> IdentityGraphSyncResult:
    """Persist provider-backed series and player identities from fixtures."""
    _require_utc(observed_at)
    identities: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    for _, row in schedule.iterrows():
        series_id = _text(row.get("serie_id"))
        if series_id:
            _add_identity(
                identities,
                links,
                entity_type=EntityType.SERIES,
                provider="pandascore",
                provider_entity_id=series_id,
                canonical_name=_text(row.get("serie"))
                or f"PandaScore series {series_id}",
                classification="provider_series",
                created_at=observed_at,
                payload={
                    "league": _text(row.get("league")),
                    "tournament_id": _text(row.get("tournament_id")),
                },
            )
        for side in ("a", "b"):
            for player in _lineup(row.get(f"team_{side}_lineup_json")):
                provider_player_id = _text(player.get("provider_player_id"))
                name = _text(player.get("name"))
                if not provider_player_id or not name:
                    continue
                _add_identity(
                    identities,
                    links,
                    entity_type=EntityType.PLAYER,
                    provider="pandascore",
                    provider_entity_id=provider_player_id,
                    canonical_name=name,
                    classification="professional_player",
                    created_at=observed_at,
                    payload={
                        "role": _text(player.get("role")),
                        "source_match_key": _text(row.get("match_key")),
                    },
                )
    return _sync_records(store, identities.values(), links.values())


def sync_history_identity_graph(
    store: EvidenceStore,
    history: pd.DataFrame,
    *,
    observed_at: datetime,
) -> IdentityGraphSyncResult:
    """Persist normalized historical identities and inferred series relations."""
    _require_utc(observed_at)
    required = {
        "gameid",
        "date",
        "game",
        "league",
        "teamid",
        "teamname",
        "playerid",
        "playername",
        "position",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history identity graph missing columns: {sorted(missing)}")
    normalized = history.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="coerce",
        utc=True,
    )
    normalized = normalized.dropna(subset=["date", "gameid"])
    identities: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    _history_people_teams_and_leagues(normalized, identities, links)
    map_rows = _historical_map_rows(normalized)
    series_rows, map_series = _infer_historical_series(map_rows)
    _history_series_and_maps(
        series_rows,
        map_rows,
        map_series,
        identities,
        links,
    )
    return _sync_records(store, identities.values(), links.values())


def _history_people_teams_and_leagues(
    history: pd.DataFrame,
    identities: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
) -> None:
    entity_specs = (
        ("teamid", "teamname", EntityType.TEAM, "professional_team"),
        ("playerid", "playername", EntityType.PLAYER, "professional_player"),
    )
    for id_column, name_column, entity_type, classification in entity_specs:
        available = history[
            history[id_column].notna() & history[name_column].notna()
        ].sort_values("date", kind="mergesort")
        for provider_id, rows in available.groupby(id_column, sort=False):
            first = rows.iloc[0]
            _add_identity(
                identities,
                links,
                entity_type=entity_type,
                provider="oracles_elixir",
                provider_entity_id=str(provider_id),
                canonical_name=str(first[name_column]),
                classification=classification,
                created_at=_timestamp(first["date"]),
                payload={},
            )
    leagues = history[history["league"].notna()].sort_values(
        "date",
        kind="mergesort",
    )
    for league, rows in leagues.groupby("league", sort=False):
        _add_identity(
            identities,
            links,
            entity_type=EntityType.LEAGUE,
            provider="oracles_elixir",
            provider_entity_id=str(league),
            canonical_name=str(league),
            classification="competition",
            created_at=_timestamp(rows.iloc[0]["date"]),
            payload={},
        )


def _historical_map_rows(history: pd.DataFrame) -> pd.DataFrame:
    maps: list[dict[str, Any]] = []
    for game_id, rows in history.groupby("gameid", sort=False):
        teams = (
            rows[rows["teamid"].notna()][["teamid", "teamname"]]
            .drop_duplicates(subset=["teamid"])
            .sort_values("teamid", kind="mergesort")
        )
        team_ids = tuple(teams["teamid"].astype(str))
        team_names = tuple(teams["teamname"].fillna("").astype(str))
        maps.append(
            {
                "gameid": str(game_id),
                "date": rows["date"].min(),
                "game": int(pd.to_numeric(rows["game"], errors="coerce").min()),
                "league": _text(rows["league"].dropna().iloc[0])
                if rows["league"].notna().any()
                else "unknown",
                "team_ids": team_ids,
                "team_names": team_names,
            }
        )
    return (
        pd.DataFrame(maps).sort_values("date", kind="mergesort").reset_index(drop=True)
    )


def _infer_historical_series(
    maps: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    series_rows: list[dict[str, Any]] = []
    map_series: dict[str, str] = {}
    state: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for _, row in maps.iterrows():
        pair = tuple(sorted(row["team_ids"]))
        key = (str(row["league"]), pair)
        prior = state.get(key)
        starts_new = (
            prior is None
            or int(row["game"]) == 1
            or int(row["game"]) <= int(prior["game"])
            or row["date"] - prior["date"] > SERIES_GAP
        )
        if starts_new:
            provider_id = _series_provider_id(
                str(row["league"]),
                pair,
                _timestamp(row["date"]),
            )
            identity_id = canonical_identity_id(
                "lol",
                EntityType.SERIES,
                "oracles_elixir_inferred",
                provider_id,
            )
            series_rows.append(
                {
                    "identity_id": identity_id,
                    "provider_id": provider_id,
                    "date": row["date"],
                    "league": row["league"],
                    "team_ids": pair,
                    "team_names": tuple(row["team_names"]),
                }
            )
        else:
            identity_id = str(prior["identity_id"])
        map_series[str(row["gameid"])] = identity_id
        state[key] = {
            "identity_id": identity_id,
            "date": row["date"],
            "game": row["game"],
        }
    return series_rows, map_series


def _history_series_and_maps(
    series_rows: list[dict[str, Any]],
    map_rows: pd.DataFrame,
    map_series: dict[str, str],
    identities: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
) -> None:
    for row in series_rows:
        matchup = " vs ".join(row["team_names"]) or "unknown teams"
        _add_identity(
            identities,
            links,
            entity_type=EntityType.SERIES,
            provider="oracles_elixir_inferred",
            provider_entity_id=str(row["provider_id"]),
            canonical_name=(
                f"{row['league']} {matchup} "
                f"{_timestamp(row['date']).date().isoformat()}"
            ),
            classification="inferred_historical_series",
            created_at=_timestamp(row["date"]),
            payload={"team_ids": list(row["team_ids"])},
        )
    for _, row in map_rows.iterrows():
        matchup = " vs ".join(row["team_names"]) or "unknown teams"
        _add_identity(
            identities,
            links,
            entity_type=EntityType.MAP,
            provider="oracles_elixir",
            provider_entity_id=str(row["gameid"]),
            canonical_name=(
                f"{row['league']} {matchup} map {row['game']} "
                f"{_timestamp(row['date']).isoformat()}"
            ),
            classification="completed_map",
            created_at=_timestamp(row["date"]),
            payload={
                "series_identity_id": map_series[str(row["gameid"])],
                "map_number": int(row["game"]),
                "team_ids": list(row["team_ids"]),
            },
        )


def _add_identity(
    identities: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
    *,
    entity_type: EntityType,
    provider: str,
    provider_entity_id: str,
    canonical_name: str,
    classification: str,
    created_at: datetime,
    payload: dict[str, Any],
) -> None:
    identity_id = canonical_identity_id(
        "lol",
        entity_type,
        provider,
        provider_entity_id,
    )
    identities.setdefault(
        identity_id,
        {
            "id": identity_id,
            "entity_type": entity_type.value,
            "canonical_name": canonical_name,
            "created_at": created_at,
            "idempotency_key": f"identity:{identity_id}",
            "payload_json": {
                "sport": "lol",
                "classification": classification,
                **payload,
            },
        },
    )
    link_id = _stable_id(
        "provider-link",
        f"{identity_id}|{provider.casefold()}|{provider_entity_id}",
    )
    links.setdefault(
        link_id,
        {
            "id": link_id,
            "identity_id": identity_id,
            "provider": provider,
            "provider_entity_id": provider_entity_id,
            "valid_from": created_at,
            "valid_to": None,
            "idempotency_key": link_id,
            "payload_json": {"source": "identity_graph_sync"},
        },
    )


def _sync_records(
    store: EvidenceStore,
    identities: Iterable[dict[str, Any]],
    links: Iterable[dict[str, Any]],
) -> IdentityGraphSyncResult:
    store.initialize_schema()
    identity_rows = list(identities)
    link_rows = list(links)
    existing_identities = {
        str(row["id"]) for row in store.list(EvidenceTable.IDENTITIES)
    }
    existing_links = {
        str(row["id"]) for row in store.list(EvidenceTable.PROVIDER_LINKS)
    }
    new_identities = [
        row for row in identity_rows if row["id"] not in existing_identities
    ]
    new_links = [row for row in link_rows if row["id"] not in existing_links]
    store.append_many(EvidenceTable.IDENTITIES, new_identities)
    store.append_many(EvidenceTable.PROVIDER_LINKS, new_links)
    return IdentityGraphSyncResult(
        discovered_identities=len(identity_rows),
        discovered_links=len(link_rows),
        added_identities=len(new_identities),
        added_links=len(new_links),
    )


def _lineup(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return (
        [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, list)
        else []
    )


def _series_provider_id(
    league: str,
    team_ids: tuple[str, ...],
    started_at: datetime,
) -> str:
    raw = "|".join((league, *team_ids, started_at.isoformat()))
    return f"inferred-series-{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _timestamp(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("identity timestamp cannot be missing")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    return cast("datetime", parsed.tz_convert(UTC).to_pydatetime())


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("identity graph observation must be timezone-aware UTC")
