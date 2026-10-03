import datetime as dt
import json

import pytest
from lol_bets import training
from lol_bets.operations import training_exposure


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MODEL_REGISTRY_DIR", tmp_path / "registry")


def _run(tmp_path, name, *, end="2026-07-31T00:00:00+00:00", status="completed"):
    root = tmp_path / "runs" / name
    model = root / training.WINNER_TUNING_MODELS["series_winner"]
    model.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"status": status}))
    (model / "split_report.json").write_text(
        json.dumps(
            {
                "test": {
                    "date_min": "2026-07-01T00:00:00+00:00",
                    "date_max": end,
                }
            }
        )
    )
    return root


def test_missing_history_freezes_future_cutoff(tmp_path):
    first = training_exposure.initialize_history(tmp_path / "runs")
    again = training_exposure.initialize_history(tmp_path / "runs")
    assert first == again
    assert first["history_status"] == "legacy_coverage_unknown"
    assert dt.datetime.fromisoformat(first["baseline_cutoff"]) <= dt.datetime.now(
        dt.UTC
    )


def test_unreviewed_and_failed_evaluations_survive_pruning(tmp_path, monkeypatch):
    _run(tmp_path, "00-old", status="failed")
    _run(tmp_path, "99-new", end="2026-08-31T00:00:00+00:00")
    monkeypatch.setattr(training, "TRAINING_REPORT_RETENTION", 1)
    training._prune_training_reports(tmp_path / "runs")
    records = training_exposure.read_exposures(tmp_path / "runs")
    assert {record["run_id"] for record in records} == {"00-old", "99-new"}
    assert not (tmp_path / "runs" / "00-old").exists()


def test_reservation_survives_without_any_reports(tmp_path):
    root = tmp_path / "runs" / "interrupted"
    training_exposure.reserve_exposure(
        root, target="series_winner", date_max="2026-08-01"
    )
    records = training_exposure.read_exposures(root.parent)
    assert len(records) == 1
    assert records[0]["kind"] == "reservation"
    assert records[0]["date_max"].startswith("2026-08-01")


def test_registry_evaluation_is_inventoried(tmp_path):
    root = tmp_path / "registry" / "candidates" / "old-champion" / "_evaluation"
    model = root / training.WINNER_TUNING_MODELS["series_winner"]
    model.mkdir(parents=True)
    (model / "split_report.json").write_text(
        json.dumps(
            {
                "test": {
                    "date_min": "2026-08-01",
                    "date_max": "2026-08-31",
                }
            }
        )
    )
    training._archive_training_exposure(tmp_path / "runs")
    records = training_exposure.read_exposures(tmp_path / "runs")
    assert records[0]["run_id"] == "registry:old-champion"
    assert records[0]["target"] == "series_winner"


def _sealed_run(tmp_path, monkeypatch, name="current"):
    root = _run(tmp_path, name)
    model = training.WINNER_TUNING_MODELS["series_winner"]
    labels = root / "artifacts" / "_evaluation" / model / "labels.parquet"
    labels.parent.mkdir(parents=True)
    labels.write_bytes(b"sealed-labels")
    monkeypatch.setattr(
        training, "_tuning_review_input_fingerprint", lambda _root: "fixed-inputs"
    )
    return root


def test_unknown_history_cannot_pass_freshness(tmp_path, monkeypatch):
    root = _sealed_run(tmp_path, monkeypatch)
    result = training._sealed_tuning_holdout(root, target="series_winner")
    assert result["fresh_for_promotion"] is False
    assert result["history_status"] == "legacy_coverage_unknown"


def test_unreviewed_evaluation_blocks_later_review_after_pruning(tmp_path, monkeypatch):
    root = _sealed_run(tmp_path, monkeypatch, "99-current")
    _run(tmp_path, "00-prior")
    training_exposure.initialize_history(
        root.parent, now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    monkeypatch.setattr(training, "TRAINING_REPORT_RETENTION", 1)
    training._prune_training_reports(root.parent)
    assert not (root.parent / "00-prior").exists()
    result = training._sealed_tuning_holdout(root, target="series_winner")
    assert result["fresh_for_promotion"] is False
    assert result["latest_prior_run_id"] == "00-prior"


def test_changed_self_fingerprint_cannot_be_rebound(tmp_path, monkeypatch):
    root = _sealed_run(tmp_path, monkeypatch)
    training_exposure.initialize_history(
        root.parent, now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    assert (
        training._sealed_tuning_holdout(root, target="series_winner")[
            "fresh_for_promotion"
        ]
        is True
    )
    monkeypatch.setattr(
        training, "_tuning_review_input_fingerprint", lambda _root: "modified-inputs"
    )
    assert (
        training._sealed_tuning_holdout(root, target="series_winner")[
            "fresh_for_promotion"
        ]
        is False
    )


def test_review_idempotent_for_unchanged_inputs(tmp_path, monkeypatch):
    root = _sealed_run(tmp_path, monkeypatch)
    reports = tmp_path / "reports"
    new_root = reports / "training" / "runs" / root.name
    new_root.parent.mkdir(parents=True)
    root.rename(new_root)
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(
        training, "_validated_winner_tuning_target", lambda _manifest: "series_winner"
    )
    training_exposure.initialize_history(
        new_root.parent, now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    calls = []

    def evidence(*_args, **_kwargs):
        calls.append(1)
        return {"status": "approved", "reasons": []}

    monkeypatch.setattr(training, "_winner_tuning_review_evidence", evidence)
    first = training.review_tuning_run(new_root.name)
    assert training.review_tuning_run(new_root.name) == first
    assert len(calls) == 1


def test_interrupted_training_reserves_exposure_before_fit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from oracle_bets_core.pd import pd

    reports = tmp_path / "reports"
    raw = tmp_path / "raw.csv"
    raw.write_text("fixture")
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "RAW_CURRENT_POINTER", tmp_path / "missing-pointer")
    monkeypatch.setattr(training, "RAW_DATA", raw)
    monkeypatch.setattr(
        training,
        "_training_preflight",
        lambda: SimpleNamespace(revision="source", clean=True),
    )
    monkeypatch.setattr(
        training, "_validate_reviewed_winner_parameters", lambda *_args, **_kwargs: None
    )
    frames = (pd.DataFrame({"date": ["2026-08-31"]}), pd.DataFrame())
    pinned = SimpleNamespace(
        frames={
            "map_teams": frames[0],
            "map_players": frames[1],
            "series_teams": frames[0],
            "series_players": frames[1],
        },
        manifests={
            "map": {"source": {"snapshot_id": "history", "data_sha256": "raw-hash"}},
            "series": {"generation_id": "tables-hash"},
        },
        assert_unchanged=lambda: None,
    )
    monkeypatch.setattr(training, "load_training_inputs", lambda **_kwargs: pinned)
    monkeypatch.setattr(training, "validate_training_tables", lambda *_args: None)
    monkeypatch.setattr(training, "_feature_set_for_config", lambda *_args: "full")
    monkeypatch.setattr(training, "_parameter_provenance", lambda: ("defaults", None))
    monkeypatch.setattr(training, "_parameter_provenance_by_target", dict)

    def interrupt(**_kwargs):
        assert training_exposure.read_exposures(reports / "training" / "runs")
        raise KeyboardInterrupt

    monkeypatch.setattr(training, "initialize_and_train_model", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train_models(targets="series_winner")
    records = training_exposure.read_exposures(reports / "training" / "runs")
    assert len(records) == 1
    assert records[0]["date_max"] == "2026-08-31T00:00:00+00:00"
    manifests = list((reports / "training" / "runs").glob("*/manifest.json"))
    assert json.loads(manifests[0].read_text())["status"] == "interrupted"
