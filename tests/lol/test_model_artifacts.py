import json
import warnings

import numpy as np
import pytest
from lol_bets.prediction_models import lightgbm_model
from lol_bets.prediction_models.gbdt_model import (
    FeaturePipeline,
    GradientBoostingModel,
    valid_probability_calibration_artifacts,
)
from lol_bets.prediction_models.lightgbm_model import LightGBMModel
from oracle_bets_core.pd import pd
from pandas.errors import PerformanceWarning

PAIRWISE_GAME_COUNT = 2
EXPECTED_ROW_ACCURACY = 0.75
EXPECTED_PAIRWISE_LOG_LOSS_UPPER_BOUND = 0.7
SHA256_HEX_LENGTH = 64
EXPECTED_ECE = 0.15
EXPECTED_NUM_LEAVES = 31
EXPECTED_ALPHA = 0.25
EXPECTED_LEARNING_RATE = 0.05
EXPECTED_LAST_MEDIAN = 159.0
EXPECTED_VALIDATION_SCORE = 0.64


def test_probability_artifact_contract_requires_serving_methods():
    calibrator = type(
        "Calibrator",
        (),
        {"version": 3, "predict": staticmethod(lambda values: values)},
    )()
    uncertainty = type(
        "Uncertainty",
        (),
        {
            "version": 2,
            "fit_split": "uncertainty_fit",
            "sample_count": 100,
            "calibration_units": 10,
            "method": "week_block_q10_plus_one_sided_calibration_bias",
            "unit": "iso_week",
            "interval": staticmethod(lambda values: (values, values)),
        },
    )()

    assert valid_probability_calibration_artifacts(calibrator, uncertainty)
    assert not valid_probability_calibration_artifacts(
        type("MetadataOnlyCalibrator", (), {"version": 3})(), uncertainty
    )
    assert not valid_probability_calibration_artifacts(
        calibrator,
        type(
            "MetadataOnlyUncertainty",
            (),
            {
                "version": 2,
                "fit_split": "uncertainty_fit",
                "sample_count": 100,
                "calibration_units": 10,
            },
        )(),
    )


def test_prop_evaluation_declares_baseline_lines_and_cohorts(tmp_path):
    model = _lightgbm_model(
        model_name="TotalKillsPrediction_LightGBM",
        problem_type="regression",
        artifact_root=tmp_path,
        report_root=tmp_path,
    )
    model.regression_baseline_value_ = 25.0
    model.store_prop_evaluation_report(
        pd.Series([24.0, 26.0, 27.0]),
        np.array([25.0, 25.5, 26.0]),
        pd.DataFrame({"league": ["LCK"] * 3, "game": [1, 2, 3]}),
    )

    report = json.loads(model.insight_path("prop_evaluation_report.json").read_text())
    assert report["constant_baseline"]["fit_split"] == "train"
    assert report["line_semantics"] == {
        "version": 1,
        "supported": "half_lines_only",
        "push_model": False,
    }
    assert report["cohort_dimensions"] == ["league", "map_number"]


def test_feature_pipeline_adds_missing_columns_without_fragmentation_warning():
    columns = [f"feature_{index}" for index in range(160)]
    pipeline = FeaturePipeline(
        train_columns=columns,
        categorical_features=[],
        categorical_levels={},
        numeric_medians={column: float(index) for index, column in enumerate(columns)},
        drop_high_missing=[],
        drop_low_variance=[],
        drop_high_correlation=[],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", PerformanceWarning)
        transformed = pipeline.transform(pd.DataFrame({"feature_0": [1.0]}))

    assert transformed.columns.tolist() == columns
    assert transformed.loc[0, "feature_159"] == EXPECTED_LAST_MEDIAN
    assert not any(
        isinstance(warning.message, PerformanceWarning) for warning in captured
    )


def test_numeric_imputation_avoids_future_downcasting_warning():
    frame = pd.DataFrame({"feature": pd.Series([1, None], dtype=object)})

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        transformed = GradientBoostingModel._impute_apply_numeric(
            frame, {"feature": 0.0}
        )

    assert transformed["feature"].tolist() == [1.0, 0.0]
    assert not any(isinstance(warning.message, FutureWarning) for warning in captured)


def test_regression_residual_summary_contains_pricing_fields():
    summary = GradientBoostingModel.build_residual_summary(
        pd.Series([10.0, 12.0, 14.0]),
        np.array([9.0, 12.5, 13.0]),
        model_name="TotalKillsPrediction_LightGBM",
    )

    assert summary["residual_sigma"] > 0
    assert summary["mae"] > 0
    assert "50" in summary["percentiles"]


def test_pairwise_classification_metrics_force_one_market_pick_per_game():
    metrics = GradientBoostingModel.compute_pairwise_classification_metrics(
        y_true=pd.Series([1, 0, 0, 1]),
        y_proba=np.array([0.7, 0.6, 0.45, 0.55]),
        eval_gameids=pd.Series(["g1", "g1", "g2", "g2"]),
    )

    assert metrics["pairwise_game_count"] == PAIRWISE_GAME_COUNT
    assert metrics["pairwise_argmax_accuracy"] == 1.0
    assert metrics["pairwise_log_loss"] < EXPECTED_PAIRWISE_LOG_LOSS_UPPER_BOUND
    assert metrics["pairwise_brier"] < EXPECTED_PAIRWISE_LOG_LOSS_UPPER_BOUND
    assert metrics["pairwise_both_predicted_win_at_0_5"] == 1
    assert metrics["row_accuracy_at_0_5"] == EXPECTED_ROW_ACCURACY


def test_probability_calibration_metrics_report_ece_and_reliability_line():
    metrics = GradientBoostingModel.compute_probability_calibration_metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert metrics["calibration_ece"] == EXPECTED_ECE
    assert metrics["calibration_slope"] > 0
    assert metrics["calibration_intercept"] < 0


def test_probability_calibration_slope_uses_log_odds_scale():
    probabilities = np.repeat(np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]), 1000)
    logits = np.log(probabilities / (1.0 - probabilities))
    observed = 1.0 / (1.0 + np.exp(-(-0.2 + 0.7 * logits)))
    outcomes = np.concatenate(
        [
            np.r_[np.ones(round(rate * 1000)), np.zeros(1000 - round(rate * 1000))]
            for rate in observed[::1000]
        ]
    )

    metrics = GradientBoostingModel.compute_probability_calibration_metrics(
        pd.Series(outcomes), probabilities
    )

    assert metrics["calibration_slope"] == pytest.approx(0.7, abs=0.03)
    assert metrics["calibration_intercept"] == pytest.approx(-0.2, abs=0.03)


def test_stable_dataframe_hash_ignores_column_order_but_tracks_content():
    left = pd.DataFrame(
        {
            "gameid": ["g1", "g2"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "target": [1, 0],
        }
    )
    reordered = left[["target", "gameid", "date"]]
    changed = left.copy()
    changed.loc[1, "target"] = 1

    first_hash = GradientBoostingModel.stable_dataframe_hash(left)

    assert len(first_hash) == SHA256_HEX_LENGTH
    assert first_hash == GradientBoostingModel.stable_dataframe_hash(reordered)
    assert first_hash != GradientBoostingModel.stable_dataframe_hash(changed)


def test_model_hyperparameters_extracts_json_safe_wrapped_params():
    class _RawModel:
        def get_params(self):
            return {
                "num_leaves": np.int64(EXPECTED_NUM_LEAVES),
                "objective": "binary",
                "nested": {"alpha": np.float64(EXPECTED_ALPHA)},
                "callbacks": (None, "early_stop"),
            }

    class _WrappedModel:
        raw_model = _RawModel()

    params = GradientBoostingModel.model_hyperparameters(_WrappedModel())

    assert params["num_leaves"] == EXPECTED_NUM_LEAVES
    assert params["objective"] == "binary"
    assert params["nested"]["alpha"] == EXPECTED_ALPHA
    assert params["callbacks"] == [None, "early_stop"]


def _lightgbm_model(**overrides) -> LightGBMModel:
    values = {
        "model_name": "OutcomePrediction",
        "problem_type": "classification",
        "team_data": pd.DataFrame(),
        "player_data": pd.DataFrame(),
    }
    values.update(overrides)
    return LightGBMModel(**values)


def test_lightgbm_hyperparameter_cache_requires_matching_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lightgbm_model, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)
    model = _lightgbm_model(feature_set="compact")
    params = {"boosting_type": "gbdt", "learning_rate": EXPECTED_LEARNING_RATE}

    model.store_best_hyperparameters(params, score=EXPECTED_VALIDATION_SCORE)

    assert model._maybe_load_cached_hparams() == params
    payload = json.loads(model._hparams_path().read_text())
    assert payload["metadata"]["validation_score"] == EXPECTED_VALIDATION_SCORE
    assert payload["params"] == params
    assert _lightgbm_model(feature_set="selected")._maybe_load_cached_hparams() is None


def test_shadow_model_may_reuse_reviewed_parameters_across_schema_drift(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lightgbm_model, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)
    original = _lightgbm_model(
        feature_set="compact",
        team_data=pd.DataFrame(columns=["old_feature"]),
    )
    params = {"boosting_type": "gbdt", "learning_rate": EXPECTED_LEARNING_RATE}
    original.store_best_hyperparameters(params, score=EXPECTED_VALIDATION_SCORE)

    strict = _lightgbm_model(
        feature_set="compact",
        team_data=pd.DataFrame(columns=["new_feature"]),
    )
    shadow = _lightgbm_model(
        feature_set="compact",
        team_data=pd.DataFrame(columns=["new_feature"]),
        allow_hparam_schema_drift=True,
    )

    assert strict._maybe_load_cached_hparams() is None
    assert shadow._maybe_load_cached_hparams() == params


def test_lightgbm_hyperparameter_cache_ignores_legacy_raw_params(tmp_path, monkeypatch):
    monkeypatch.setattr(lightgbm_model, "TUNED_LIGHTGBM_HYPERPARAMETERS", tmp_path)
    model = _lightgbm_model()
    path = model._hparams_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"boosting_type": "gbdt", "learning_rate": EXPECTED_LEARNING_RATE})
    )

    assert model._maybe_load_cached_hparams() is None


def test_lightgbm_retraining_never_retunes_without_explicit_flag(monkeypatch):
    model = _lightgbm_model()
    monkeypatch.setattr(model, "_maybe_load_cached_hparams", lambda: None)
    monkeypatch.setattr(
        model,
        "_optimize_hyperparameters",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not retune")),
    )

    with pytest.raises(RuntimeError, match="oracle-bets lol retune"):
        model.train_model(
            X_train=pd.DataFrame({"x": [0, 1]}),
            y_train=pd.Series([0, 1]),
            X_val=pd.DataFrame({"x": [0, 1]}),
            y_val=pd.Series([0, 1]),
            categorical_features=None,
        )
