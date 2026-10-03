from datetime import UTC, datetime

import pytest
from lol_bets.inference.snapshots import load_feature_snapshot, publish_feature_snapshot
from oracle_bets_core.pd import pd


def _history():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01T10:00Z", "2026-09-01T18:00Z"]),
            "gameid": ["early", "late"],
            "teamname": ["A", "A"],
            # Ingestion normalizes Oracle's Elixir seconds to minutes.
            "gamelength": [30.0, 30.0],
            "elo_after": [1500, 1600],
        }
    )


def _publish(root, *, observed_at=datetime(2026, 9, 2, tzinfo=UTC)):
    columns = ["date", "gameid", "teamname", "elo_after"]
    return publish_feature_snapshot(
        _history(),
        _history().assign(playername="P", playerid="p", position="top"),
        team_columns=columns,
        player_columns=[*columns, "playername", "playerid", "position"],
        source_manifest={"snapshot_id": "history-one", "data_sha256": "source-hash"},
        code_sha256="code-one",
        training_generation_id="training-one",
        root=root,
        observed_at=observed_at,
    )


def test_snapshot_preserves_history_and_requires_known_at_cutoff(tmp_path):
    snapshot = _publish(tmp_path)
    loaded = load_feature_snapshot(
        tmp_path, decision_at=datetime(2026, 9, 3, tzinfo=UTC)
    )
    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.manifest["code_sha256"] == "code-one"
    assert loaded.manifest["training_generation_id"] == "training-one"
    assert loaded.read("teams")["elo"].tolist() == [1500, 1600]
    assert loaded.read("teams")["state_available_at"].tolist() == list(
        pd.to_datetime(["2026-09-01T10:30Z", "2026-09-01T18:30Z"])
    )
    with pytest.raises(ValueError, match="not available"):
        load_feature_snapshot(
            tmp_path, decision_at=datetime(2026, 9, 1, 12, tzinfo=UTC)
        )


def test_pinned_snapshot_survives_new_publication(tmp_path):
    first = _publish(tmp_path)
    second = _publish(tmp_path, observed_at=datetime(2026, 9, 4, tzinfo=UTC))
    pinned = load_feature_snapshot(
        tmp_path,
        snapshot_id=first.snapshot_id,
        decision_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert pinned.snapshot_id == first.snapshot_id
    assert load_feature_snapshot(tmp_path).snapshot_id == second.snapshot_id


def test_failed_publication_preserves_pointer(tmp_path):
    first = _publish(tmp_path)
    with pytest.raises(ValueError, match="gamelength"):
        publish_feature_snapshot(
            _history().drop(columns="gamelength"),
            _history(),
            team_columns=["date", "elo_after"],
            player_columns=["date", "elo_after"],
            source_manifest={},
            code_sha256="code-one",
            training_generation_id="training-one",
            root=tmp_path,
        )
    assert load_feature_snapshot(tmp_path).snapshot_id == first.snapshot_id


def test_snapshot_rejects_corrupt_files_and_path_escape(tmp_path):
    snapshot = _publish(tmp_path)
    (snapshot.directory / "teams.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_feature_snapshot(tmp_path)
    with pytest.raises(ValueError, match="snapshot ID"):
        load_feature_snapshot(tmp_path, snapshot_id="../other")
