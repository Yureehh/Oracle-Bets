from __future__ import annotations

import pytest
from lol_bets.operations.feature_reuse import (
    feature_build_matches,
    record_feature_build,
)


@pytest.mark.parametrize(
    "change", [None, "source", "code", "output", "missing", "receipt"]
)
def test_feature_reuse_requires_unchanged_identity_and_every_output(tmp_path, change):
    output = tmp_path / "teams.parquet"
    output.write_bytes(b"verified-output")
    receipt = tmp_path / "feature-build.json"
    paths = {"teams": output}
    source = {
        "snapshot_id": "history-a",
        "data_sha256": "a",
        "source_snapshot_id": "source-a",
    }
    record_feature_build(receipt, paths=paths, source=source, code="code-a")
    code = "code-a"
    if change == "source":
        source = {**source, "source_snapshot_id": "source-b"}
    elif change == "code":
        code = "code-b"
    elif change == "output":
        output.write_bytes(b"corrupt-output")
    elif change == "missing":
        output.unlink()
    elif change == "receipt":
        receipt.write_text("{broken")
    assert feature_build_matches(receipt, paths=paths, source=source, code=code) is (
        change is None
    )


def test_feature_reuse_rejects_missing_output_contract(tmp_path):
    output = tmp_path / "teams.parquet"
    output.write_bytes(b"output")
    receipt = tmp_path / "feature-build.json"
    record_feature_build(receipt, paths={"teams": output}, source={}, code="code")
    assert not feature_build_matches(receipt, paths={}, source={}, code="code")


@pytest.mark.parametrize("change", [None, "rows", "provider_source"])
def test_unchanged_ingestion_retains_history_identity(tmp_path, monkeypatch, change):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from lol_bets import pipeline
    from lol_bets.data_generation.ingestion.history import (
        HistoryRefreshMode,
        merge_history,
        publish_history_snapshot,
    )
    from oracle_bets_core.pd import pd

    original = pd.DataFrame(
        [
            {
                "gameid": "g1",
                "side": "Blue",
                "position": "team",
                "teamid": "a",
                "playerid": None,
                "result": 1,
                "kills": 10,
            }
        ]
    )
    _, manifest = merge_history(
        pd.DataFrame(columns=original.columns),
        original,
        mode=HistoryRefreshMode.FULL,
        refreshed_at=datetime(2026, 10, 10, tzinfo=UTC),
        source_snapshot_id="source-stable",
    )
    raw, receipt, pointer, generations = (
        tmp_path / name
        for name in ("raw.parquet", "history.json", "current.json", "generations")
    )
    publish_history_snapshot(
        original,
        manifest,
        raw_path=raw,
        manifest_path=receipt,
        pointer_path=pointer,
        generations_dir=generations,
    )
    before = pointer.read_bytes()
    for name, path in (
        ("RAW_DATA", raw),
        ("HISTORY_REFRESH_MANIFEST", receipt),
        ("RAW_CURRENT_POINTER", pointer),
        ("RAW_GENERATIONS_DIR", generations),
    ):
        monkeypatch.setattr(pipeline, name, path)
    monkeypatch.setattr(
        pipeline,
        "source_snapshot_id",
        lambda _: "source-changed" if change == "provider_source" else "source-stable",
    )
    incoming = original.assign(kills=11) if change == "rows" else original.copy()
    generator = pipeline.DataGenerator.__new__(pipeline.DataGenerator)
    generator.history_mode = HistoryRefreshMode.INCREMENTAL
    generator.training_file_evidence = {}
    generator.oracle = SimpleNamespace(
        local_data_dir=tmp_path, ingest_data=lambda **_: incoming
    )
    result = generator.ingest_data()
    assert result["kills"].iloc[0] == (11 if change == "rows" else 10)
    assert (pointer.read_bytes() == before) is (change is None)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "rating_league_elo_missing",
        "rating_league_elo_changed",
        "rating_team_leagues_missing",
        "rating_team_leagues_changed",
        "serving_corrupt",
        "map_manifest",
        "code",
        "source",
    ],
)
def test_pipeline_reuse_checks_complete_outputs_before_skipping_work(
    tmp_path, monkeypatch, change
):
    from lol_bets import pipeline
    from lol_bets.inference.snapshots import publish_feature_snapshot
    from oracle_bets_core.pd import pd

    source = {
        "snapshot_id": "history-a",
        "data_sha256": "a",
        "source_snapshot_id": "source-a",
    }
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    for name in (
        "MAP_INPUT_MANIFEST",
        "INTERIM_TEAM_DATA",
        "INTERIM_PLAYER_DATA",
        "PROCESSED_TEAMS",
        "PROCESSED_PLAYERS",
        "DATA_QUALITY_REPORT",
        "QUARANTINED_RAW_DATA",
        "RATING_LEAGUE_ELO",
        "RATING_TEAM_LEAGUES_MAPPING",
    ):
        monkeypatch.setattr(pipeline, name, tmp_path / name)
    history = pd.DataFrame(
        {"date": ["2026-10-10T12:00:00Z"], "gamelength": [30], "score_after": [1]}
    )
    publish_feature_snapshot(
        history,
        history,
        team_columns=["score_after"],
        player_columns=["score_after"],
        source_manifest=source,
        code_sha256="code-a",
        training_generation_id="generation-a",
        root=tmp_path / "serving",
    )
    generator = pipeline.DataGenerator.__new__(pipeline.DataGenerator)
    generator.source_manifest = source
    generator.generation_code = "code-a"
    paths = generator._feature_files()
    for path in paths.values():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
    record_feature_build(
        tmp_path / "feature_build.json", paths=paths, source=source, code="code-a"
    )
    monkeypatch.setattr(
        pipeline, "code_fingerprint", lambda: "code-b" if change == "code" else "code-a"
    )
    monkeypatch.setattr(
        pipeline,
        "current_source",
        lambda _: (
            {**source, "snapshot_id": "history-b"} if change == "source" else source
        ),
    )
    monkeypatch.setattr(pipeline.DataGenerator, "ingest_data", lambda _: history)

    def rebuild(_self, _raw):
        raise RuntimeError("full rebuild required")

    monkeypatch.setattr(pipeline.DataGenerator, "clean_and_store_data", rebuild)
    if change and change.startswith("rating_"):
        name, operation = change.rsplit("_", 1)
        if operation == "missing":
            paths[name].unlink()
        else:
            paths[name].write_bytes(b"changed")
    elif change == "serving_corrupt":
        paths["serving_teams.parquet"].write_bytes(b"corrupt")
    elif change == "map_manifest":
        paths["map_manifest"].write_bytes(b"changed")
    if change is None:
        generator.run()
    else:
        with pytest.raises(RuntimeError, match="full rebuild required"):
            generator.run()
