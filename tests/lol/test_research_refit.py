from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from lol_bets import training
from lol_bets.prediction_models.lightgbm_model import LightGBMModel
from oracle_bets_core.cli import build_parser
from oracle_bets_core.pd import pd


def _studies(tmp_path):
    runs = []
    for cfg in training.ALL_MODEL_CONFIGS:
        run = tmp_path / "reports" / "training" / "runs" / cfg.target_name
        run.mkdir(parents=True)
        run.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run.name,
                    "status": "completed",
                    "retune": True,
                    "targets_requested": [cfg.target_name],
                    "targets_trained": [cfg.model_name],
                    "targets_failed": [],
                    "code_version": "code",
                    "training_input_generations": {
                        "map": {
                            "source": {"data_sha256": "raw"},
                            "generation_id": "generation",
                        },
                        "series": {
                            "source": {"data_sha256": "raw"},
                            "generation_id": "generation",
                        },
                    },
                }
            )
        )
        model = LightGBMModel(
            model_name=f"{cfg.model_name}_LightGBM",
            problem_type=cfg.problem_type,
            team_data=pd.DataFrame({"date": ["2026-01-01"]}),
            player_data=pd.DataFrame(),
            feature_set="compact",
            max_features=80,
        )
        path = run / model.model_name / "tuned_hyperparameters.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "metadata": model._hparam_cache_metadata()
                    | {"validation_score": 0.4},
                    "params": {"learning_rate": 0.05},
                }
            )
        )
        runs.append(run.name)
    return runs


def test_research_refit_cli_accepts_independent_study_ids():
    args = build_parser().parse_args(["lol", "refit-research", "map-run", "series-run"])
    assert args.run_ids == ["map-run", "series-run"]
    assert args.targets == "all"


def test_research_refit_keeps_all_targets_research_only(tmp_path, monkeypatch):
    runs = _studies(tmp_path)
    monkeypatch.setattr(training, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path / "tracked"
    )
    monkeypatch.setattr(training, "MODEL_REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(
        training,
        "_training_preflight",
        lambda: SimpleNamespace(revision="code", clean=True),
    )
    frame = pd.DataFrame({"date": ["2026-01-01"]})
    generation = {"source": {"data_sha256": "raw"}, "generation_id": "generation"}
    inputs = SimpleNamespace(
        frames={
            "map_teams": frame,
            "map_players": pd.DataFrame(),
            "series_teams": frame,
            "series_players": pd.DataFrame(),
        },
        manifests={"map": generation, "series": generation},
        assert_unchanged=lambda: None,
    )
    monkeypatch.setattr(training, "load_training_inputs", lambda **_kw: inputs)
    monkeypatch.setattr(training, "validate_training_tables", lambda *_args: None)
    monkeypatch.setattr(training, "_write_training_summary", lambda *_args: None)
    fitted = []

    def fit(**options):
        model = LightGBMModel(
            model_name=f"{options['cfg'].model_name}_LightGBM",
            problem_type=options["cfg"].problem_type,
            team_data=options["training_team_data"],
            player_data=options["training_player_data"],
            feature_set=options["feature_set"],
            max_features=options["max_features"],
            research_hparams_path=options["research_hparams_path"],
        )
        params = model._maybe_load_cached_hparams()
        assert params == {"learning_rate": 0.05}
        assert options["force_retune"] is False
        fitted.append(options)
        options["artifact_root"].mkdir(parents=True, exist_ok=True)
        options["artifact_root"].joinpath(options["cfg"].model_name).write_text(
            "research model"
        )

    monkeypatch.setattr(training, "initialize_and_train_model", fit)
    report = training.train_models(research_tuning_runs=tuple(runs))
    manifest = json.loads((report / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["research_only"] is True
    assert manifest["promotable_full_bundle"] is False
    assert manifest["optuna_allowed"] is False
    assert manifest["parameter_source"] == "optuna_development_selected"
    assert "candidate_id" not in manifest
    assert len(fitted) == len(training.ALL_MODEL_CONFIGS)
    assert not (tmp_path / "tracked").exists()
    assert not (tmp_path / "registry").exists()
    for options in fitted:
        path = options["research_hparams_path"]
        assert path.is_relative_to(report)
        provenance = manifest["parameter_provenance"][options["cfg"].target_name]
        assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert provenance["run_id"] == options["cfg"].target_name
        assert provenance["review_status"] == "review_required"
        assert (report / "artifacts" / options["cfg"].model_name).is_file()


def test_research_parameters_reject_schema_drift(tmp_path):
    model = LightGBMModel(
        model_name="Map",
        problem_type="classification",
        team_data=pd.DataFrame(),
        player_data=pd.DataFrame(),
    )
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "params": {"learning_rate": 0.05},
                "metadata": model._hparam_cache_metadata()
                | {"feature_schema_fingerprint": "different"},
            }
        )
    )
    model.research_hparams_path = path
    model.allow_hparam_schema_drift = True
    with pytest.raises(ValueError, match="research parameter"):
        model._maybe_load_cached_hparams()
