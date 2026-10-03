from __future__ import annotations

import hashlib

import pytest
from lol_bets.operations import training_inputs as inputs
from oracle_bets_core.pd import pd


@pytest.fixture
def generation(tmp_path, monkeypatch):
    source = {
        "snapshot_id": "raw-1",
        "data_sha256": "raw-hash",
        "source_snapshot_id": "upstream-1",
    }
    monkeypatch.setattr(inputs, "current_source", lambda _pointer: source.copy())
    monkeypatch.setattr(inputs, "code_fingerprint", lambda: "code-1")
    paths = {}
    for key in ("map_teams", "map_players", "interim_teams"):
        path = tmp_path / f"{key}.parquet"
        pd.DataFrame({"value": [1]}).to_parquet(path)
        paths[key] = path
    manifest = tmp_path / "map.json"
    inputs.publish_generation(
        manifest, paths, source=source, code="code-1", pointer=tmp_path / "current.json"
    )
    return manifest, paths, source


def test_loaded_bytes_and_manifest_are_bound(generation, tmp_path):
    manifest, paths, _ = generation
    pinned = inputs.load_training_inputs(manifest, pointer=tmp_path / "current.json")
    assert pinned.frames["map_teams"]["value"].tolist() == [1]
    assert (
        pinned.manifests["map"]["files"]["map_teams"]["sha256"]
        == hashlib.sha256(paths["map_teams"].read_bytes()).hexdigest()
    )
    pinned.assert_unchanged()


def test_missing_manifest_requires_rebuild(tmp_path):
    with pytest.raises(RuntimeError, match="rebuild"):
        inputs.load_training_inputs(
            tmp_path / "missing.json", pointer=tmp_path / "current.json"
        )


def test_tampered_table_fails_before_loading(generation, tmp_path):
    manifest, paths, _ = generation
    pd.DataFrame({"value": [2]}).to_parquet(paths["map_players"])
    with pytest.raises(RuntimeError, match="changed"):
        inputs.load_training_inputs(manifest, pointer=tmp_path / "current.json")


@pytest.mark.parametrize("change", ["pointer", "table", "code"])
def test_change_after_loading_fails_before_fit(
    generation, tmp_path, monkeypatch, change
):
    manifest, paths, source = generation
    pinned = inputs.load_training_inputs(manifest, pointer=tmp_path / "current.json")
    if change == "pointer":
        source["snapshot_id"] = "raw-2"
    elif change == "code":
        monkeypatch.setattr(inputs, "code_fingerprint", lambda: "code-2")
    else:
        pd.DataFrame({"value": [2]}).to_parquet(paths["map_players"])
    with pytest.raises(RuntimeError, match="changed"):
        pinned.assert_unchanged()


def test_series_cannot_mix_map_generations(generation, tmp_path):
    manifest, paths, source = generation
    series = tmp_path / "series.json"
    inputs.publish_generation(
        series,
        {"series_teams": paths["map_teams"], "series_players": paths["map_players"]},
        source=source,
        code="code-1",
        pointer=tmp_path / "current.json",
        parent_generation="different",
    )
    with pytest.raises(RuntimeError, match="map generation"):
        inputs.load_training_inputs(
            manifest, series_manifest=series, pointer=tmp_path / "current.json"
        )


def test_publication_rejects_source_or_code_changed(generation, tmp_path):
    _, paths, source = generation
    with pytest.raises(RuntimeError, match="changed"):
        inputs.publish_generation(
            tmp_path / "bad.json",
            paths,
            source=source | {"snapshot_id": "older"},
            code="code-1",
            pointer=tmp_path / "current.json",
        )
    assert not (tmp_path / "bad.json").exists()


def test_series_publication_binds_exact_parent_and_outputs(
    generation, tmp_path, monkeypatch
):
    from lol_bets.data_generation import series

    manifest, _, _ = generation
    pinned = inputs.load_training_inputs(manifest, pointer=tmp_path / "current.json")
    monkeypatch.setattr(series, "load_training_inputs", lambda: pinned)
    frame = pd.DataFrame({"value": [1]})
    monkeypatch.setattr(
        series,
        "reconstruct_series",
        lambda _frame: series.SeriesBuildResult(frame, frame),
    )
    monkeypatch.setattr(
        series, "_series_winner_training_tables", lambda *_args: (frame, frame)
    )
    monkeypatch.setattr(
        series, "_next_map_training_tables", lambda *_args: (frame, frame)
    )
    for name in (
        "SERIES_MANIFEST",
        "SERIES_REJECTIONS",
        "SERIES_WINNER_TEAM_DATA",
        "SERIES_WINNER_PLAYER_DATA",
        "NEXT_MAP_TEAM_DATA",
        "NEXT_MAP_PLAYER_DATA",
    ):
        monkeypatch.setattr(series, name, tmp_path / f"{name}.parquet")
    series_manifest = tmp_path / "series.json"
    monkeypatch.setattr(series, "SERIES_INPUT_MANIFEST", series_manifest)
    series.build_series_artifacts()
    loaded = inputs.load_training_inputs(
        manifest, series_manifest=series_manifest, pointer=tmp_path / "current.json"
    )
    assert (
        loaded.manifests["series"]["parent_generation"]
        == pinned.manifests["map"]["generation_id"]
    )
    assert loaded.frames["series_teams"]["value"].tolist() == [1]


def test_map_publication_rejects_mixed_pair_written_during_generation(
    generation, tmp_path
):
    manifest, paths, source = generation
    expected = inputs.file_evidence(paths)
    pd.DataFrame({"value": [2]}).to_parquet(paths["map_players"])
    with pytest.raises(RuntimeError, match="changed during generation"):
        inputs.publish_generation(
            manifest,
            paths,
            source=source,
            code="code-1",
            expected_files=expected,
            pointer=tmp_path / "current.json",
        )


def test_training_failure_preserves_pinned_input_manifest(
    generation, tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from lol_bets import training

    manifest, paths, _ = generation
    pd.DataFrame({"date": ["2026-01-01"]}).to_parquet(paths["map_teams"])
    inputs.publish_generation(
        manifest,
        paths,
        source=generation[2],
        code="code-1",
        pointer=tmp_path / "current.json",
    )
    pinned = inputs.load_training_inputs(manifest, pointer=tmp_path / "current.json")
    monkeypatch.setattr(
        training,
        "_training_preflight",
        lambda: SimpleNamespace(revision="code-revision", clean=True),
    )
    monkeypatch.setattr(
        training, "_validate_reviewed_winner_parameters", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(training, "load_training_inputs", lambda **_kwargs: pinned)
    monkeypatch.setattr(training, "validate_training_tables", lambda *_args: None)
    monkeypatch.setattr(training, "_archive_training_exposure", lambda *_args: None)
    monkeypatch.setattr(training, "reserve_exposure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training, "_parameter_provenance", lambda: ("defaults", None))
    monkeypatch.setattr(training, "_parameter_provenance_by_target", dict)
    monkeypatch.setattr(training, "REPORTS_DIR", tmp_path / "reports")
    captured = []

    def fit(**kwargs):
        captured.append(kwargs["dataset_fingerprint"])
        pd.DataFrame({"value": [3]}).to_parquet(paths["map_players"])

    monkeypatch.setattr(training, "initialize_and_train_model", fit)
    with pytest.raises(RuntimeError, match="Training failed"):
        training.train_models(targets="map_winner", force_retune=True)
    report = next((tmp_path / "reports").rglob("manifest.json"))
    content = json.loads(report.read_text())
    assert content["status"] == "failed"
    assert content["training_input_generations"] == pinned.manifests
    assert captured == [pinned.manifests["map"]["generation_id"]]


def test_feature_configuration_variant_changes_generation_fingerprint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(inputs, "SUITE_ROOT", tmp_path)
    monkeypatch.delenv("TRAINING_CONFIG_VARIANT", raising=False)
    full = inputs.code_fingerprint()
    monkeypatch.setenv("TRAINING_CONFIG_VARIANT", "compact")
    assert inputs.code_fingerprint() != full
