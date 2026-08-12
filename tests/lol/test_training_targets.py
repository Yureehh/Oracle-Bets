import json

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
        "next_map_winner",
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
    monkeypatch.setattr(training, "TEAM_LEAGUES_MAPPING", mapping)
    monkeypatch.setattr(training, "LEAGUE_ELO", league_elo)

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
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retune": True,
                "targets_trained": [
                    config.model_name for config in training.ALL_MODEL_CONFIGS
                ],
                "targets_failed": [],
            }
        )
    )


def test_promote_tuning_run_requires_complete_four_target_bundle(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)
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
    run_root = reports / "training" / "runs" / "winner-v2"
    selected = training.parse_training_targets("series_winner,next_map_winner")
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


def test_promote_tuning_run_atomically_publishes_complete_bundle(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    tuned = tmp_path / "tuned"
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(training, "TUNED_LIGHTGBM_HYPERPARAMETERS", tuned)

    names = [
        training._lightgbm_model_name(config.model_name)
        for config in training.ALL_MODEL_CONFIGS
    ]
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
    run_root = reports / "training" / "runs" / "mixed"
    _write_complete_tuning_manifest(run_root)
    names = [
        training._lightgbm_model_name(config.model_name)
        for config in training.ALL_MODEL_CONFIGS
    ]
    for index, name in enumerate(names):
        candidate = run_root / name / "tuned_hyperparameters.json"
        candidate.parent.mkdir(parents=True)
        payload = _tuning_candidate(name)
        payload["metadata"]["dataset_fingerprint"] = f"data-{index}"
        candidate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="inconsistent provenance"):
        training.promote_tuning_run("mixed")


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
