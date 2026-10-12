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
