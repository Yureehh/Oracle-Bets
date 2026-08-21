import json
from types import SimpleNamespace

import numpy as np
import pytest
from lol_bets import training
from lol_bets.training import parse_training_targets, validate_training_tables
from oracle_bets_core.cli import build_parser
from oracle_bets_core.pd import pd

SELECTED_FEATURE_COUNT = 90
PLAYER_ROWS_PER_GAME = 10
SCHEDULE_DAY_COUNT = 7
EXPECTED_BRIER = 0.19


def _targets(selector: str) -> list[str]:
    return [cfg.target_name for cfg in parse_training_targets(selector)]


def test_training_target_parser_supports_all_outcome_props_and_commas():
    assert _targets("all") == [
        "map_winner",
        "series_winner",
        "gamelength",
        "total_kills",
        "total_towers",
    ]
    assert _targets("outcome") == ["map_winner"]
    assert _targets("series_winner,next_map_winner") == [
        "series_winner",
        "next_map_winner",
    ]
    assert _targets("props") == ["gamelength", "total_kills", "total_towers"]
    assert _targets("total_kills,total_towers") == ["total_kills", "total_towers"]


def test_training_target_parser_rejects_unknown_targets():
    with pytest.raises(ValueError, match="Unknown training target"):
        parse_training_targets("barons")


def test_training_parser_supports_feature_set_and_max_features():
    args = build_parser().parse_args(
        [
            "lol",
            "train",
            "--targets",
            "outcome",
            "--feature-set",
            "selected",
            "--max-features",
            str(SELECTED_FEATURE_COUNT),
        ]
    )

    assert args.feature_set == "selected"
    assert args.max_features == SELECTED_FEATURE_COUNT


def test_routine_training_defaults_to_reviewed_compact_contract():
    args = build_parser().parse_args(["lol", "train"])

    assert args.feature_set == "compact"


def test_winner_v2_targets_always_use_full_feature_contract():
    configs = {config.target_name: config for config in training.MODEL_CONFIGS}

    assert (
        training._feature_set_for_config(configs["series_winner"], "compact") == "full"
    )
    assert (
        training._feature_set_for_config(configs["next_map_winner"], "selected")
        == "full"
    )
    assert (
        training._feature_set_for_config(configs["gamelength"], "compact") == "compact"
    )


def test_routine_all_excludes_experimental_next_map_target():
    routine = training.parse_training_targets("all")
    experimental = training.parse_training_targets("next_map_winner")

    assert "next_map_winner" not in {config.target_name for config in routine}
    assert [config.target_name for config in experimental] == ["next_map_winner"]


def test_routine_training_preflight_requires_all_selected_winner_parameters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)
    selected = training.parse_training_targets("series_winner,next_map_winner")

    with pytest.raises(
        RuntimeError,
        match="reviewed fixed parameters for: series_winner, next_map_winner",
    ):
        training._validate_reviewed_winner_parameters(
            selected,
            force_retune=False,
        )


def test_routine_training_preflight_accepts_independently_reviewed_parameters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)
    selected = training.parse_training_targets("series_winner,next_map_winner")
    for config in selected:
        model_name = training.WINNER_TUNING_MODELS[config.target_name]
        tmp_path.joinpath(f"{model_name}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "model_name": model_name,
                        "source": "optuna_reviewed",
                        "promoted_tuning_run_id": f"run-{config.target_name}",
                    },
                    "params": {"num_leaves": 15},
                }
            )
        )

    training._validate_reviewed_winner_parameters(selected, force_retune=False)


def test_research_retune_does_not_require_existing_reviewed_parameters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)

    training._validate_reviewed_winner_parameters(
        training.parse_training_targets("series_winner"),
        force_retune=True,
    )


def test_research_retune_requires_one_independent_winner_target():
    selected = training.parse_training_targets("series_winner,next_map_winner")

    with pytest.raises(ValueError, match="independent tuning runs"):
        training._validate_reviewed_winner_parameters(selected, force_retune=True)


def test_training_parser_supports_validate_data_action():
    args = build_parser().parse_args(["lol", "validate-data"])

    assert args.domain == "lol"
    assert args.action == "validate-data"


def test_lol_schedule_parser_supports_days_and_league_filter():
    args = build_parser().parse_args(
        ["lol", "schedule", "--days", str(SCHEDULE_DAY_COUNT), "--leagues", "LCK,LEC"]
    )

    assert args.action == "schedule"
    assert args.days == SCHEDULE_DAY_COUNT
    assert args.leagues == "LCK,LEC"


def test_lol_market_review_parser_requires_explicit_urls_and_publish_flag():
    args = build_parser().parse_args(
        [
            "lol",
            "market-review",
            "https://polymarket.com/event/lol-t1-geng",
            "--publish",
        ]
    )

    assert args.action == "market-review"
    assert args.urls == ["https://polymarket.com/event/lol-t1-geng"]
    assert args.publish is True


def _valid_training_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    team_df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2", "g2"],
            "side": ["Blue", "Red", "Blue", "Red"],
            "result": [1, 0, 0, 1],
            "gamelength": [31.0, 31.0, 33.0, 33.0],
            "total_kills": [28.0, 28.0, 24.0, 24.0],
            "total_towers": [13.0, 13.0, 11.0, 11.0],
        }
    )
    player_df = pd.DataFrame(
        [
            {"gameid": gameid, "side": side, "position": position}
            for gameid in ("g1", "g2")
            for side in ("Blue", "Red")
            for position in ("top", "jng", "mid", "bot", "sup")
        ]
    )
    return team_df, player_df


def test_validate_training_tables_accepts_well_formed_tables():
    team_df, player_df = _valid_training_tables()

    validate_training_tables(team_df, player_df)

    assert len(player_df) == PLAYER_ROWS_PER_GAME * team_df["gameid"].nunique()


def test_validate_training_tables_rejects_games_without_one_winner():
    team_df, player_df = _valid_training_tables()
    team_df.loc[team_df["gameid"] == "g1", "result"] = [1, 1]

    with pytest.raises(ValueError, match="exactly one winner"):
        validate_training_tables(team_df, player_df)


def test_validate_training_tables_rejects_prop_target_disagreement():
    team_df, player_df = _valid_training_tables()
    team_df.loc[
        (team_df["gameid"] == "g1") & (team_df["side"] == "Red"), "total_kills"
    ] = 29.0

    with pytest.raises(ValueError, match="must agree across both sides"):
        validate_training_tables(team_df, player_df)


def test_validate_training_tables_rejects_incomplete_player_roles():
    team_df, player_df = _valid_training_tables()
    player_df = player_df[
        ~(
            (player_df["gameid"] == "g1")
            & (player_df["side"] == "Blue")
            & (player_df["position"] == "sup")
        )
    ]

    with pytest.raises(ValueError, match="ten player rows"):
        validate_training_tables(team_df, player_df)


def test_training_stages_shared_rating_artifacts(tmp_path, monkeypatch):
    mapping = tmp_path / "source" / "team_league_mapping.parquet"
    league_elo = tmp_path / "source" / "league_elo.parquet"
    mapping.parent.mkdir()
    mapping.write_bytes(b"mapping")
    league_elo.write_bytes(b"elo")
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(training, "RATING_TEAM_LEAGUES_MAPPING", mapping)
    monkeypatch.setattr(training, "RATING_LEAGUE_ELO", league_elo)

    training._stage_shared_inference_artifacts(staging)

    assert (staging / mapping.name).read_bytes() == b"mapping"
    assert (staging / league_elo.name).read_bytes() == b"elo"


def _tuning_candidate(model_name: str) -> dict:
    return {
        "metadata": {
            "model_name": model_name,
            "validation_score": 0.42,
            "feature_set": "compact",
            "max_features": 120,
            "feature_schema_fingerprint": "schema-sha",
            "dataset_fingerprint": "data-sha",
            "code_version": "code-sha",
            "random_seed": 42,
        },
        "params": {"boosting_type": "gbdt", "learning_rate": 0.05},
    }


def _write_complete_tuning_manifest(root) -> None:
    selected = training.parse_training_targets(
        "map_winner,gamelength,total_kills,total_towers"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": [config.target_name for config in selected],
                "targets_trained": [config.model_name for config in selected],
                "targets_failed": [],
            }
        )
    )
    root.joinpath("tuning_review.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "run_id": root.name,
                "review_input_sha256": "test-review-inputs",
            }
        )
    )


def test_promote_tuning_run_requires_complete_four_target_bundle(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "test-review-inputs",
    )
    _write_complete_tuning_manifest(reports / "training" / "runs" / "incomplete")
    existing = tuned / "OutcomePrediction_LightGBM.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"preserve": true}\n')

    candidate = (
        reports
        / "training"
        / "runs"
        / "incomplete"
        / "OutcomePrediction_LightGBM"
        / "tuned_hyperparameters.json"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(_tuning_candidate("OutcomePrediction_LightGBM")))

    with pytest.raises(FileNotFoundError, match="Tuning candidate is missing"):
        training.promote_tuning_run("incomplete")

    assert existing.read_text() == '{"preserve": true}\n'


def test_promote_tuning_run_rejects_incomplete_manifest(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    run_root = reports / "training" / "runs" / "interrupted"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        '{"status":"interrupted","retune":true,"targets_trained":[]}'
    )
    with pytest.raises(ValueError, match="not complete for its requested targets"):
        training.promote_tuning_run("interrupted")


def test_promote_tuning_run_publishes_only_requested_targets(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "test-review-inputs",
    )
    run_root = reports / "training" / "runs" / "winner-v2"
    selected = training.parse_training_targets("series_winner")
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": [config.target_name for config in selected],
                "targets_trained": [config.model_name for config in selected],
                "targets_failed": [],
            }
        )
    )
    run_root.joinpath("tuning_review.json").write_text(
        json.dumps(
            {
                "schema_version": training.TUNING_REVIEW_SCHEMA_VERSION,
                "status": "approved",
                "run_id": "winner-v2",
                "target": "series_winner",
                "review_input_sha256": "test-review-inputs",
                "sealed_holdout": {"fresh_for_promotion": True},
            }
        )
    )
    for index, config in enumerate(selected):
        name = training._lightgbm_model_name(config.model_name)
        candidate = run_root / name / "tuned_hyperparameters.json"
        candidate.parent.mkdir(parents=True)
        payload = _tuning_candidate(name)
        payload["metadata"]["feature_schema_fingerprint"] = f"schema-{index}"
        candidate.write_text(json.dumps(payload))

    promoted = training.promote_tuning_run("winner-v2")

    assert [path.stem for path in promoted] == [
        training._lightgbm_model_name(config.model_name) for config in selected
    ]


def test_promote_tuning_accepts_approved_review_with_disclosed_warnings(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "test-review-inputs",
    )
    run_root = reports / "training" / "runs" / "winner-v2"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": ["series_winner"],
                "targets_trained": ["SeriesWinnerPrediction"],
                "targets_failed": [],
            }
        )
    )
    run_root.joinpath("tuning_review.json").write_text(
        json.dumps(
            {
                "schema_version": training.TUNING_REVIEW_SCHEMA_VERSION,
                "status": "approved_with_warnings",
                "warnings": ["holdout_overlaps_previously_exposed_window"],
                "run_id": "winner-v2",
                "target": "series_winner",
                "review_input_sha256": "test-review-inputs",
                "sealed_holdout": {"fresh_for_promotion": False},
            }
        )
    )
    name = "SeriesWinnerPrediction_LightGBM"
    candidate = run_root / name / "tuned_hyperparameters.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(_tuning_candidate(name)))

    [promoted] = training.promote_tuning_run("winner-v2")
    payload = json.loads(promoted.read_text())

    assert payload["metadata"]["review_warnings"] == [
        "holdout_overlaps_previously_exposed_window"
    ]


def test_promote_tuning_run_rejects_unreviewed_next_map_target(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    run_root = reports / "training" / "runs" / "next-map"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": ["next_map_winner"],
                "targets_trained": ["NextMapWinnerPrediction"],
                "targets_failed": [],
            }
        )
    )

    with pytest.raises(ValueError, match=r"must pass.*review-tuning"):
        training.promote_tuning_run("next-map")


def test_promote_tuning_run_accepts_independently_reviewed_next_map(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "next-map-review-inputs",
    )
    run_root = reports / "training" / "runs" / "next-map"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": ["next_map_winner"],
                "targets_trained": ["NextMapWinnerPrediction"],
                "targets_failed": [],
            }
        )
    )
    run_root.joinpath("tuning_review.json").write_text(
        json.dumps(
            {
                "schema_version": training.TUNING_REVIEW_SCHEMA_VERSION,
                "status": "approved",
                "run_id": "next-map",
                "target": "next_map_winner",
                "review_input_sha256": "next-map-review-inputs",
                "sealed_holdout": {"fresh_for_promotion": True},
            }
        )
    )
    name = "NextMapWinnerPrediction_LightGBM"
    candidate = run_root / name / "tuned_hyperparameters.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(_tuning_candidate(name)))

    promoted = training.promote_tuning_run("next-map")

    assert [path.name for path in promoted] == [f"{name}.json"]


def test_promote_tuning_run_atomically_publishes_requested_targets(
    tmp_path, monkeypatch
):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "test-review-inputs",
    )

    selected = training.parse_training_targets(
        "map_winner,gamelength,total_kills,total_towers"
    )
    names = [training._lightgbm_model_name(config.model_name) for config in selected]
    _write_complete_tuning_manifest(reports / "training" / "runs" / "complete")
    for name in names:
        candidate = (
            reports
            / "training"
            / "runs"
            / "complete"
            / name
            / "tuned_hyperparameters.json"
        )
        candidate.parent.mkdir(parents=True)
        candidate.write_text(json.dumps(_tuning_candidate(name)))

    promoted = training.promote_tuning_run("complete")

    assert [path.name for path in promoted] == [f"{name}.json" for name in names]
    for name in names:
        payload = json.loads((tuned / f"{name}.json").read_text())
        assert payload["metadata"]["model_name"] == name


def test_promote_tuning_run_rejects_mixed_provenance(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: "test-review-inputs",
    )
    run_root = reports / "training" / "runs" / "mixed"
    _write_complete_tuning_manifest(run_root)
    selected = training.parse_training_targets(
        "map_winner,gamelength,total_kills,total_towers"
    )
    names = [training._lightgbm_model_name(config.model_name) for config in selected]
    for index, name in enumerate(names):
        candidate = run_root / name / "tuned_hyperparameters.json"
        candidate.parent.mkdir(parents=True)
        payload = _tuning_candidate(name)
        payload["metadata"]["dataset_fingerprint"] = f"data-{index}"
        candidate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="inconsistent provenance"):
        training.promote_tuning_run("mixed")


@pytest.mark.parametrize(
    ("review", "fingerprint", "message"),
    [
        (
            {"status": "blocked", "run_id": "winner"},
            "expected",
            "stale, or blocked",
        ),
        (
            {"status": "approved", "run_id": "other"},
            "expected",
            "stale, or blocked",
        ),
        (
            {
                "status": "approved",
                "run_id": "winner",
                "review_input_sha256": "old",
            },
            "expected",
            "stale, or blocked",
        ),
    ],
)
def test_promote_winner_tuning_rejects_invalid_review(
    tmp_path, monkeypatch, review, fingerprint, message
):
    reports = tmp_path / "reports"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(
        training, "_tuning_review_input_fingerprint", lambda _root: fingerprint
    )
    run_root = reports / "training" / "runs" / "winner"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": ["series_winner"],
                "targets_trained": ["SeriesWinnerPrediction"],
                "targets_failed": [],
            }
        )
    )
    run_root.joinpath("tuning_review.json").write_text(json.dumps(review))

    with pytest.raises(ValueError, match=message):
        training.promote_tuning_run("winner")


def test_promote_winner_tuning_rejects_change_during_copy(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
    fingerprints = iter(("expected", "changed"))
    monkeypatch.setattr(
        training,
        "_tuning_review_input_fingerprint",
        lambda _root: next(fingerprints),
    )
    run_root = reports / "training" / "runs" / "winner"
    run_root.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": ["series_winner"],
                "targets_trained": ["SeriesWinnerPrediction"],
                "targets_failed": [],
            }
        )
    )
    run_root.joinpath("tuning_review.json").write_text(
        json.dumps(
            {
                "schema_version": training.TUNING_REVIEW_SCHEMA_VERSION,
                "status": "approved",
                "run_id": "winner",
                "target": "series_winner",
                "review_input_sha256": "expected",
                "sealed_holdout": {"fresh_for_promotion": True},
            }
        )
    )
    name = "SeriesWinnerPrediction_LightGBM"
    candidate = run_root / name / "tuned_hyperparameters.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(_tuning_candidate(name)))

    with pytest.raises(ValueError, match="changed during promotion"):
        training.promote_tuning_run("winner")


class _ReviewPipeline:
    @staticmethod
    def transform(frame):
        return frame


class _ReviewMember:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get_params(self):
        return {"learning_rate": self.learning_rate}


class _ReviewWinnerModel:
    def __init__(self, learning_rate):
        self.members = (_ReviewMember(learning_rate), _ReviewMember(learning_rate))

    @staticmethod
    def rating_baseline_probability(frame):
        return np.full(len(frame), 0.5)

    @staticmethod
    def predict_proba(frame):
        probability = frame["candidate_probability"].to_numpy(dtype=float)
        return np.column_stack([1.0 - probability, probability])


class _ReviewCalibrator:
    version = 3

    @staticmethod
    def predict(values, metadata=None):
        del metadata
        return np.asarray(values, dtype=float)


class _ReviewUncertainty:
    version = 1
    fit_split = "uncertainty_fit"
    sample_count = 100

    @staticmethod
    def interval(values):
        return values, values


def _winner_review_run(
    tmp_path,
    *,
    learning_rate=0.05,
    target="series_winner",
    run_id="winner-review",
):
    run_root = tmp_path / "reports" / "training" / "runs" / run_id
    model_name = training.WINNER_TUNING_MODELS[target]
    model_root = run_root / "artifacts" / model_name
    evaluation = run_root / "artifacts" / "_evaluation" / model_name
    candidate = run_root / model_name / "tuned_hyperparameters.json"
    model_root.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    run_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_requested": [target],
                "targets_trained": [model_name.removesuffix("_LightGBM")],
                "targets_failed": [],
            }
        )
    )
    payload = _tuning_candidate(model_name)
    payload["params"] = {"learning_rate": learning_rate}
    candidate.write_text(json.dumps(payload))
    run_root.joinpath(model_name, "split_report.json").write_text(
        json.dumps(
            {
                "test": {
                    "date_min": "2026-07-01T00:00:00+00:00",
                    "date_max": "2026-07-31T00:00:00+00:00",
                }
            }
        )
    )
    for path in (
        evaluation / "features.parquet",
        evaluation / "labels.parquet",
        model_root / f"{model_name}.pkl",
        model_root / f"{model_name}_feature_pipeline.pkl",
        model_root / f"{model_name}_probability_calibrator.pkl",
        model_root / f"{model_name}_probability_uncertainty.pkl",
    ):
        path.write_bytes(b"review-fixture")
    return run_root


@pytest.mark.parametrize(
    ("member_rate", "uncertainty", "expected_status", "expected_reason"),
    [
        (0.05, _ReviewUncertainty(), "approved", None),
        (0.10, _ReviewUncertainty(), "blocked", "winner_member_parameter_mismatch"),
        (
            0.05,
            SimpleNamespace(
                version=1,
                fit_split="uncertainty_fit",
                sample_count=100,
            ),
            "blocked",
            "invalid_probability_calibration_artifacts",
        ),
    ],
)
def test_review_winner_tuning_binds_model_parameters_and_artifacts(
    tmp_path,
    monkeypatch,
    member_rate,
    uncertainty,
    expected_status,
    expected_reason,
):
    import lol_bets.operations.models as model_operations
    from lol_bets.prediction_models.gbdt_model import GradientBoostingModel

    run_root = _winner_review_run(tmp_path)
    monkeypatch.setattr(training, "REPORTS_DIR", tmp_path / "reports")
    features = pd.DataFrame({"candidate_probability": np.tile([0.1, 0.9], 50)})
    labels = pd.DataFrame(
        {
            "actual": np.tile([0, 1], 50),
            "actionable": True,
            "league": "LCK",
        }
    )
    monkeypatch.setattr(
        training.pd,
        "read_parquet",
        lambda path: (
            labels.copy() if path.name == "labels.parquet" else features.copy()
        ),
    )
    model = _ReviewWinnerModel(member_rate)

    def fake_load(path):
        name = path.name
        if name.endswith("feature_pipeline.pkl"):
            return _ReviewPipeline()
        if name.endswith("probability_calibrator.pkl"):
            return _ReviewCalibrator()
        if name.endswith("probability_uncertainty.pkl"):
            return uncertainty
        return model

    monkeypatch.setattr(training, "load_model", fake_load)
    monkeypatch.setattr(
        model_operations,
        "evaluate_promotion",
        lambda evidence, **_kwargs: SimpleNamespace(
            promote=not evidence.operational_failures,
            reasons=tuple(
                f"operational_failure:{reason}"
                for reason in evidence.operational_failures
            ),
            relative_improvement=0.10,
            confidence_lower_bound=0.05,
        ),
    )
    monkeypatch.setattr(
        GradientBoostingModel,
        "compute_probability_calibration_metrics",
        staticmethod(
            lambda *_args, **_kwargs: {
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        ),
    )

    result = training.review_tuning_run(run_root.name)

    assert result["status"] == expected_status
    assert result["target"] == "series_winner"
    if expected_reason:
        assert any(expected_reason in reason for reason in result["reasons"])
    else:
        assert result["reasons"] == []
    assert json.loads(run_root.joinpath("tuning_review.json").read_text()) == result


def test_review_next_map_tuning_bootstraps_by_series(tmp_path, monkeypatch):
    import lol_bets.operations.models as model_operations
    from lol_bets.prediction_models.gbdt_model import GradientBoostingModel

    run_root = _winner_review_run(tmp_path, target="next_map_winner")
    monkeypatch.setattr(training, "REPORTS_DIR", tmp_path / "reports")
    features = pd.DataFrame({"candidate_probability": np.tile([0.1, 0.9], 50)})
    labels = pd.DataFrame(
        {
            "actual": np.tile([0, 1], 50),
            "series_id": np.repeat([f"series-{index}" for index in range(50)], 2),
            "actionable": True,
            "league": "LCK",
        }
    )
    monkeypatch.setattr(
        training.pd,
        "read_parquet",
        lambda path: (
            labels.copy() if path.name == "labels.parquet" else features.copy()
        ),
    )
    model = _ReviewWinnerModel(0.05)

    def fake_load(path):
        name = path.name
        if name.endswith("feature_pipeline.pkl"):
            return _ReviewPipeline()
        if name.endswith("probability_calibrator.pkl"):
            return _ReviewCalibrator()
        if name.endswith("probability_uncertainty.pkl"):
            return _ReviewUncertainty()
        return model

    observed: dict[str, int] = {}

    def fake_evaluate(evidence, **_kwargs):
        observed["bootstrap_units"] = len(evidence.candidate_log_losses)
        observed["cohort_units"] = evidence.cohort_log_loss[
            "all_research_all_supported"
        ][2]
        return SimpleNamespace(
            promote=True,
            reasons=(),
            relative_improvement=0.10,
            confidence_lower_bound=0.05,
        )

    monkeypatch.setattr(training, "load_model", fake_load)
    monkeypatch.setattr(model_operations, "evaluate_promotion", fake_evaluate)
    monkeypatch.setattr(
        GradientBoostingModel,
        "compute_probability_calibration_metrics",
        staticmethod(
            lambda *_args, **_kwargs: {
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        ),
    )

    result = training.review_tuning_run(run_root.name)

    assert result["status"] == "approved"
    assert result["target"] == "next_map_winner"
    assert result["bootstrap_unit"] == "series_id"
    assert result["cluster_identity_source"] == "sealed_series_id"
    assert observed == {"bootstrap_units": 50, "cohort_units": 50}


def test_next_map_review_recovers_strict_synthetic_series_identity() -> None:
    labels = pd.DataFrame(
        {
            "gameid": [
                "series-0123456789abcdef01234567:map-2",
                "series-0123456789abcdef01234567:map-3",
                "series-fedcba9876543210fedcba98:map-2",
            ]
        }
    )

    recovered, source = training._next_map_review_labels(labels)

    assert source == "derived_from_synthetic_gameid"
    assert recovered["series_id"].tolist() == [
        "series-0123456789abcdef01234567",
        "series-0123456789abcdef01234567",
        "series-fedcba9876543210fedcba98",
    ]


def test_next_map_review_rejects_unverified_game_identity() -> None:
    with pytest.raises(ValueError, match="complete series_id clusters"):
        training._next_map_review_labels(pd.DataFrame({"gameid": ["provider-map-123"]}))


@pytest.mark.parametrize(
    ("date_min", "expected_fresh"),
    [
        ("2026-07-15T00:00:00+00:00", False),
        ("2026-08-01T00:00:00+00:00", True),
    ],
)
def test_tuning_holdout_must_follow_every_previously_exposed_row(
    tmp_path,
    date_min,
    expected_fresh,
):
    _winner_review_run(tmp_path, run_id="previous")
    current = _winner_review_run(tmp_path, run_id="current")
    model_name = training.WINNER_TUNING_MODELS["series_winner"]
    current.joinpath(model_name, "split_report.json").write_text(
        json.dumps(
            {
                "test": {
                    "date_min": date_min,
                    "date_max": "2026-08-31T00:00:00+00:00",
                }
            }
        )
    )

    holdout = training._sealed_tuning_holdout(
        current,
        target="series_winner",
    )

    assert holdout["fresh_for_promotion"] is expected_fresh
    assert holdout["latest_prior_run_id"] == "previous"


def test_legacy_naive_holdout_dates_are_interpreted_as_utc() -> None:
    parsed = training._parse_utc_holdout_date("2026-08-13T16:27:00")

    assert parsed.isoformat() == "2026-08-13T16:27:00+00:00"


def test_training_summary_combines_metrics_cards_and_top_features(tmp_path):
    report_root = tmp_path / "runs" / "run-1"
    config = training.ALL_MODEL_CONFIGS[0]
    model_name = training._lightgbm_model_name(config.model_name)
    model_root = report_root / model_name
    model_root.mkdir(parents=True)
    (model_root / "metrics.json").write_text('{"brier": 0.19, "log_loss": 0.60}\n')
    (model_root / "model_card_run-1.json").write_text('{"feature_hash": "abc"}\n')
    (model_root / "calibration_report.json").write_text(
        '{"pairwise_test_metrics": {"max_probability_sum_error": 0.0}}\n'
    )
    pd.DataFrame(
        [
            {"feature": "rating_delta", "gain": 9.0},
            {"feature": "form_delta", "gain": 2.0},
        ]
    ).to_parquet(model_root / "feature_importances.parquet")

    json_path, markdown_path = training._write_training_summary(
        report_root,
        (config,),
        {
            "run_id": "run-1",
            "created_at_utc": "2026-07-30T00:00:00+00:00",
            "retune": False,
            "optuna_allowed": False,
        },
    )

    payload = json.loads(json_path.read_text())
    assert payload["models"][0]["metrics"]["brier"] == EXPECTED_BRIER
    assert payload["models"][0]["evidence_status"] == "meets_basic_sanity"
    assert (
        payload["models"][0]["calibration"]["pairwise_test_metrics"][
            "max_probability_sum_error"
        ]
        == 0.0
    )
    assert payload["models"][0]["top_features"][0]["feature"] == "rating_delta"
    assert "rating_delta" in markdown_path.read_text()


def test_training_report_retention_keeps_newest_completed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "TRAINING_REPORT_RETENTION", 2)
    runs_root = tmp_path / "runs"
    for run_id in ("run-1", "run-2", "run-3"):
        root = runs_root / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
    incomplete = runs_root / "run-0"
    incomplete.mkdir()

    training._prune_training_reports(runs_root)

    assert not (runs_root / "run-1").exists()
    assert (runs_root / "run-2").exists()
    assert (runs_root / "run-3").exists()
    assert incomplete.exists()
