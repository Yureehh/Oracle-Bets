from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from lol_bets.data_generation.ingestion.history import (
    HistoryRefreshMode,
    SourceHistoryError,
    merge_history,
    refresh_years,
    write_history_manifest,
)
from oracle_bets_core.cli import build_parser
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)


def _row(game_id, *, result, kills=10):
    return {
        "gameid": game_id,
        "side": "Blue",
        "position": "team",
        "teamid": "team-a",
        "playerid": None,
        "result": result,
        "kills": kills,
    }


def test_incremental_refresh_adds_new_rows_and_replaces_corrected_rows():
    existing = pd.DataFrame([_row("g1", result=0), _row("g2", result=1)])
    incoming = pd.DataFrame(
        [_row("g1", result=1), _row("g2", result=1), _row("g3", result=0)]
    )

    merged, manifest = merge_history(
        existing,
        incoming,
        mode=HistoryRefreshMode.INCREMENTAL,
        refreshed_at=NOW,
    )

    assert set(merged["gameid"]) == {"g1", "g2", "g3"}
    assert merged.set_index("gameid").loc["g1", "result"] == 1
    assert manifest.added_rows == 1
    assert manifest.updated_rows == 1
    assert manifest.unchanged_rows == 1
    assert manifest.removed_rows == 0


def test_incremental_refresh_replaces_the_complete_incoming_game():
    existing = pd.DataFrame(
        [
            _row("g1", result=1),
            {**_row("g1", result=0), "side": "Red", "teamid": "team-b"},
            _row("older", result=1),
        ]
    )
    incoming = pd.DataFrame([_row("g1", result=1)])

    merged, manifest = merge_history(
        existing,
        incoming,
        mode=HistoryRefreshMode.INCREMENTAL,
        refreshed_at=NOW,
    )

    assert len(merged.loc[merged["gameid"] == "g1"]) == 1
    assert "older" in set(merged["gameid"])
    assert manifest.removed_rows == 1


def test_full_refresh_uses_the_new_audited_snapshot_and_reports_removed_rows():
    existing = pd.DataFrame([_row("old", result=1), _row("keep", result=0)])
    incoming = pd.DataFrame([_row("keep", result=0), _row("new", result=1)])

    merged, manifest = merge_history(
        existing,
        incoming,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW,
    )

    assert set(merged["gameid"]) == {"keep", "new"}
    assert manifest.added_rows == 1
    assert manifest.removed_rows == 1
    assert manifest.updated_rows == 0


def test_refresh_rejects_schema_drift():
    existing = pd.DataFrame([_row("g1", result=0)])
    incoming = pd.DataFrame([_row("g2", result=1)]).drop(columns=["kills"])

    with pytest.raises(SourceHistoryError, match="schema"):
        merge_history(
            existing,
            incoming,
            mode=HistoryRefreshMode.INCREMENTAL,
            refreshed_at=NOW,
        )


def test_refresh_rejects_duplicate_row_identity():
    duplicated = pd.DataFrame([_row("g1", result=0), _row("g1", result=1)])

    with pytest.raises(SourceHistoryError, match="duplicate row identities"):
        merge_history(
            pd.DataFrame(columns=duplicated.columns),
            duplicated,
            mode=HistoryRefreshMode.INCREMENTAL,
            refreshed_at=NOW,
        )


def test_refresh_uses_names_when_provider_entity_ids_are_missing():
    team_row = _row("g1", result=1)
    team_row.update({"teamid": None, "teamname": "Team A"})
    player_row = _row("g2", result=1)
    player_row.update(
        {
            "position": "top",
            "teamid": None,
            "playerid": None,
            "playername": "Player A",
        }
    )
    incoming = pd.DataFrame([team_row, player_row])

    merged, _ = merge_history(
        pd.DataFrame(columns=incoming.columns),
        incoming,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW,
    )

    assert set(merged["gameid"]) == {"g1", "g2"}


def test_refresh_rejects_rows_without_ids_or_names():
    incoming = pd.DataFrame([_row("g1", result=1)])
    incoming.loc[0, ["teamid", "playerid"]] = None

    with pytest.raises(SourceHistoryError, match="identity"):
        merge_history(
            pd.DataFrame(columns=incoming.columns),
            incoming,
            mode=HistoryRefreshMode.FULL,
            refreshed_at=NOW,
        )


def test_refresh_year_selection_is_explicit():
    assert refresh_years(2026, HistoryRefreshMode.INCREMENTAL) == [2026]
    assert refresh_years(2026, HistoryRefreshMode.FULL, years_back=3) == [
        2026,
        2025,
        2024,
    ]


def test_cli_exposes_explicit_full_history_reconciliation():
    args = build_parser().parse_args(["lol", "reconcile-history"])

    assert args.domain == "lol"
    assert args.action == "reconcile-history"


def test_history_manifest_write_is_machine_readable(tmp_path):
    _, manifest = merge_history(
        pd.DataFrame([_row("g1", result=0)]),
        pd.DataFrame([_row("g1", result=0)]),
        mode=HistoryRefreshMode.INCREMENTAL,
        refreshed_at=NOW,
    )
    path = tmp_path / "history.json"

    write_history_manifest(manifest, path)

    payload = json.loads(path.read_text())
    assert payload["mode"] == "incremental"
    assert payload["refreshed_at"] == "2026-07-26T08:15:00Z"
    assert not list(tmp_path.glob("*.tmp"))
