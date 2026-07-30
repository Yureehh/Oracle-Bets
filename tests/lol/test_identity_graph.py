from __future__ import annotations

import json
from datetime import UTC, datetime

from lol_bets.operations.identity import (
    sync_history_identity_graph,
    sync_schedule_identity_graph,
)
from oracle_bets_core.cli import build_parser
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 27, 8, tzinfo=UTC)
EXPECTED_HISTORY_IDENTITIES = 8
EXPECTED_SCHEDULE_IDENTITIES = 3


def _history() -> pd.DataFrame:
    rows = []
    for map_number, hour in ((1, 1), (2, 2)):
        for team_id, team_name, player_id, player_name, side in (
            ("team-a", "T1", "player-a", "Player A", "Blue"),
            ("team-b", "Gen.G", "player-b", "Player B", "Red"),
        ):
            common = {
                "gameid": f"game-{map_number}",
                "date": f"2026-07-01T0{hour}:00:00Z",
                "game": map_number,
                "league": "LCK",
                "teamid": team_id,
                "teamname": team_name,
                "side": side,
            }
            rows.extend(
                (
                    {
                        **common,
                        "playerid": player_id,
                        "playername": player_name,
                        "position": "top",
                    },
                    {
                        **common,
                        "playerid": None,
                        "playername": None,
                        "position": "team",
                    },
                )
            )
    return pd.DataFrame(rows)


def test_history_sync_creates_first_class_series_and_map_relations(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")

    first = sync_history_identity_graph(store, _history(), observed_at=NOW)
    second = sync_history_identity_graph(store, _history(), observed_at=NOW)

    assert first.added_identities == EXPECTED_HISTORY_IDENTITIES
    assert first.added_links == EXPECTED_HISTORY_IDENTITIES
    assert second.added_identities == 0
    assert second.added_links == 0
    identities = store.list(EvidenceTable.IDENTITIES)
    assert {row["entity_type"] for row in identities} == {
        "team",
        "player",
        "league",
        "series",
        "map",
    }
    map_rows = [row for row in identities if row["entity_type"] == "map"]
    series_ids = {row["id"] for row in identities if row["entity_type"] == "series"}
    assert {
        json.loads(row["payload_json"])["series_identity_id"] for row in map_rows
    } == series_ids


def test_schedule_sync_preserves_provider_series_and_player_ids(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    schedule = pd.DataFrame(
        [
            {
                "serie_id": "series-1",
                "serie": "LCK Summer",
                "tournament_id": "tournament-1",
                "league": "LCK",
                "match_key": "pandascore:77",
                "team_a_lineup_json": json.dumps(
                    [
                        {
                            "provider_player_id": "pa",
                            "name": "Player A",
                            "role": "top",
                        }
                    ]
                ),
                "team_b_lineup_json": json.dumps(
                    [
                        {
                            "provider_player_id": "pb",
                            "name": "Player B",
                            "role": "top",
                        }
                    ]
                ),
            }
        ]
    )

    result = sync_schedule_identity_graph(store, schedule, observed_at=NOW)

    assert result.added_identities == EXPECTED_SCHEDULE_IDENTITIES
    assert {row["entity_type"] for row in store.list(EvidenceTable.IDENTITIES)} == {
        "series",
        "player",
    }
    assert {
        row["provider_entity_id"] for row in store.list(EvidenceTable.PROVIDER_LINKS)
    } == {"series-1", "pa", "pb"}


def test_cli_exposes_identity_graph_sync():
    args = build_parser().parse_args(["lol", "sync-identities"])

    assert args.domain == "lol"
    assert args.action == "sync-identities"
