from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from lol_bets.data_generation.ingestion.history import (
    HistoryRefreshMode,
    SourceHistoryError,
    merge_history,
    publish_history_snapshot,
    read_history_snapshot,
    refresh_years,
    write_history_manifest,
)
from oracle_bets_core.cli import build_parser
from oracle_bets_core.pd import pd

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)
POINTER_WRITE_CALL = 2


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


def test_history_snapshot_publishes_pointer_and_verifies_content(tmp_path):
    existing = pd.DataFrame([_row("g1", result=0)])
    merged, manifest = merge_history(
        pd.DataFrame(columns=existing.columns),
        existing,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW,
        source_snapshot_id="source-0123456789abcdef01234567",
    )
    raw_path = tmp_path / "raw_data.parquet"
    manifest_path = tmp_path / "history.json"
    pointer_path = tmp_path / "current.json"
    snapshot_id = publish_history_snapshot(
        merged,
        manifest,
        raw_path=raw_path,
        manifest_path=manifest_path,
        generations_dir=tmp_path / "generations",
        pointer_path=pointer_path,
    )

    data_path, payload = read_history_snapshot(pointer_path=pointer_path)
    assert payload["snapshot_id"] == snapshot_id
    assert payload["source_snapshot_id"] == "source-0123456789abcdef01234567"
    assert pd.read_parquet(data_path).equals(merged)
    data_path.write_bytes(data_path.read_bytes() + b"corrupt")
    with pytest.raises(SourceHistoryError, match="checksum"):
        read_history_snapshot(pointer_path=pointer_path)


def test_history_snapshot_pointer_failure_keeps_previous_generation(
    tmp_path, monkeypatch
):
    existing = pd.DataFrame([_row("g1", result=0)])
    merged, _ = merge_history(
        pd.DataFrame(columns=existing.columns),
        existing,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW,
    )
    initial_manifest = merge_history(
        pd.DataFrame(columns=existing.columns),
        existing,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW,
    )[1]
    publish_history_snapshot(
        merged,
        initial_manifest,
        raw_path=tmp_path / "raw_data.parquet",
        manifest_path=tmp_path / "history.json",
        generations_dir=tmp_path / "generations",
        pointer_path=tmp_path / "current.json",
    )
    before = (tmp_path / "current.json").read_text()
    calls = 0
    original = __import__(
        "lol_bets.data_generation.ingestion.history", fromlist=["_atomic_json"]
    )._atomic_json

    def fail_pointer(path, payload):
        nonlocal calls
        calls += 1
        if calls == POINTER_WRITE_CALL:
            raise OSError("injected pointer failure")
        return original(path, payload)

    monkeypatch.setattr(
        "lol_bets.data_generation.ingestion.history._atomic_json", fail_pointer
    )
    changed = existing.assign(result=[1])
    changed_manifest = merge_history(
        pd.DataFrame(columns=existing.columns),
        changed,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=NOW.replace(hour=9),
    )[1]
    with pytest.raises(OSError, match="pointer"):
        publish_history_snapshot(
            changed,
            changed_manifest,
            raw_path=tmp_path / "raw_data.parquet",
            manifest_path=tmp_path / "history.json",
            generations_dir=tmp_path / "generations",
            pointer_path=tmp_path / "current.json",
        )
    assert (tmp_path / "current.json").read_text() == before
    data_path, _ = read_history_snapshot(pointer_path=tmp_path / "current.json")
    assert pd.read_parquet(data_path).equals(merged)
    assert len(list((tmp_path / "generations").iterdir())) == 1


def test_history_snapshot_retains_current_and_previous_generations(tmp_path):
    pointer = tmp_path / "current.json"
    generations = tmp_path / "generations"
    published = []
    for hour, result in ((8, 0), (9, 1), (10, 0)):
        data = pd.DataFrame([_row("g1", result=result, kills=hour)])
        _, manifest = merge_history(
            pd.DataFrame(columns=data.columns),
            data,
            mode=HistoryRefreshMode.FULL,
            refreshed_at=NOW.replace(hour=hour),
        )
        published.append(
            publish_history_snapshot(
                data,
                manifest,
                raw_path=tmp_path / "raw_data.parquet",
                manifest_path=tmp_path / "history.json",
                generations_dir=generations,
                pointer_path=pointer,
            )
        )

    retained = {path.name for path in generations.iterdir()}
    assert retained == set(published[-2:])
