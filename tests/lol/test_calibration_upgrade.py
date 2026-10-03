from __future__ import annotations

import warnings

import numpy as np
import pytest
from lol_bets.prediction_models.gbdt_model import (
    WINNER_TEMPORAL_SPLIT_MODEL_NAMES,
    GradientBoostingModel,
    MetadataAwareProbabilityCalibrator,
    ProbabilityCalibrator,
    ProbabilityUncertaintyModel,
    PropDistributionCalibrator,
    _calibration_metadata,
    conservative_probability_coverage,
)
from oracle_bets_core.paths import (
    GAMELENGTH_PREDICTION_PROP_CALIBRATOR,
    TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR,
    TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR,
)
from oracle_bets_core.pd import pd

SEGMENT_FIT_ROWS = 100
SEGMENT_SELECT_ROWS = 50
PROBABILITY_THRESHOLD = 0.5
UNCERTAINTY_SAMPLE_COUNT = 100
MIN_UNCERTAINTY_WEEK_BLOCKS = 4
MAX_COVERAGE_SHORTFALL = 0.02


class DummyCalibratedModel(GradientBoostingModel):
    def train_model(self, **kwargs):
        assert kwargs is not None

    def _optimize_hyperparameters(self, *args):
        assert args is not None
        return {}


class ProbabilityColumnModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = X["p"].to_numpy(dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


def test_calibration_metadata_preserves_timestamps_without_model_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"], utc=True),
            "league": ["LCK"],
            "model_signal": [42.0],
        }
    )

    metadata = _calibration_metadata(frame)

    assert list(metadata) == ["date", "league"]
    assert metadata["date"].notna().all()


def _calibration_model(tmp_path, monkeypatch) -> DummyCalibratedModel:
    from lol_bets.prediction_models import gbdt_model as gbdt_module

    monkeypatch.setattr(gbdt_module, "MODELS_DIR", tmp_path)
    return DummyCalibratedModel(
        model_name="CalibrationDummy",
        problem_type="classification",
        team_data=pd.DataFrame(),
        player_data=pd.DataFrame(),
        artifact_root=tmp_path,
        report_root=tmp_path / "reports",
    )


def _segmented_probability_data(
    *, segment_rows: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for league in ("LCK", "LEC"):
        for idx in range(segment_rows):
            p = 0.8 if idx % 2 else 0.2
            rows.append({"p": p})
            labels.append(
                int(p < PROBABILITY_THRESHOLD)
                if league == "LCK"
                else int(p > PROBABILITY_THRESHOLD)
            )
    X = pd.DataFrame(rows)
    y = pd.Series(labels, index=X.index)
    meta = pd.DataFrame(
        {
            "league": ["LCK"] * segment_rows + ["LEC"] * segment_rows,
            "strength_pool": ["major"] * len(X),
            "is_bo3": [1] * len(X),
            "patch": ["15.1"] * len(X),
        },
        index=X.index,
    )
    return X, y, meta


def test_temporal_calibration_split_has_no_game_overlap() -> None:
    games = [f"g{i}" for i in range(20)]
    X = pd.DataFrame(
        {
            "gameid": games,
            "date": pd.date_range("2026-01-01", periods=len(games), freq="D"),
            "feature": np.arange(len(games)),
        }
    )
    y = pd.Series([i % 2 for i in range(len(games))], index=X.index)

    splits = GradientBoostingModel.temporal_train_tune_cal_test_split(
        X,
        y,
        tune_size=0.10,
        calibration_size=0.20,
        test_size=0.20,
    )
    x_splits = splits[:6]
    game_sets = [set(frame["gameid"]) for frame in x_splits]

    assert [len(values) for values in game_sets] == [10, 2, 2, 1, 1, 4]
    for idx, left in enumerate(game_sets):
        for right in game_sets[idx + 1 :]:
            assert left.isdisjoint(right)


def test_temporal_calibration_split_keeps_equal_timestamps_together() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-03",
            "2026-01-03",
            "2026-01-04",
            "2026-01-04",
            "2026-01-05",
            "2026-01-05",
            "2026-01-06",
            "2026-01-06",
        ]
    )
    X = pd.DataFrame(
        {
            "gameid": [f"g{i}" for i in range(len(timestamps))],
            "date": timestamps,
            "feature": np.arange(len(timestamps)),
        }
    )
    y = pd.Series([i % 2 for i in range(len(X))], index=X.index)

    splits = GradientBoostingModel.temporal_train_tune_cal_test_split(
        X,
        y,
        tune_size=0.10,
        calibration_size=0.30,
        test_size=0.20,
    )

    timestamp_owners: dict[pd.Timestamp, int] = {}
    for split_index, frame in enumerate(splits[:6]):
        for raw_timestamp in pd.to_datetime(frame["date"]).unique():
            timestamp = pd.Timestamp(raw_timestamp)
            assert timestamp_owners.setdefault(timestamp, split_index) == split_index


def test_winner_v2_uses_predeclared_temporal_partitions() -> None:
    X = pd.DataFrame(
        {
            "gameid": [f"series-{index}" for index in range(100)],
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
    )
    y = pd.Series([index % 2 for index in range(100)])

    splits = GradientBoostingModel.temporal_winner_v2_split(X, y)

    assert [len(frame) for frame in splits[:6]] == [55, 5, 10, 10, 5, 15]


def test_map_winner_uses_the_dedicated_uncertainty_partition() -> None:
    assert "OutcomePrediction_LightGBM" in WINNER_TEMPORAL_SPLIT_MODEL_NAMES


def test_split_integrity_rejects_overlapping_gameids() -> None:
    y = pd.Series([0, 1])
    train = pd.DataFrame({"gameid": ["g1"], "date": [pd.Timestamp("2026-01-01")]})
    test = pd.DataFrame({"gameid": ["g1"], "date": [pd.Timestamp("2026-01-02")]})

    with pytest.raises(ValueError, match="Split gameids must be disjoint"):
        GradientBoostingModel.validate_split_integrity(
            {"train": (train, y.iloc[:1]), "test": (test, y.iloc[1:])},
            require_temporal_order=True,
        )


def test_split_integrity_rejects_non_temporal_order() -> None:
    y = pd.Series([0, 1])
    train = pd.DataFrame({"gameid": ["g1"], "date": [pd.Timestamp("2026-01-03")]})
    test = pd.DataFrame({"gameid": ["g2"], "date": [pd.Timestamp("2026-01-02")]})

    with pytest.raises(ValueError, match="Temporal split order is invalid"):
        GradientBoostingModel.validate_split_integrity(
            {"train": (train, y.iloc[:1]), "test": (test, y.iloc[1:])},
            require_temporal_order=True,
        )


def test_probability_calibrators_share_predict_interface() -> None:
    raw = ProbabilityCalibrator(method="raw", model=None)
    values = np.array([0.25, 0.50, 0.75])

    predicted = raw.predict(values)

    assert np.allclose(predicted, values)
    assert np.all((predicted > 0) & (predicted < 1))


def test_winner_calibration_rejects_better_loss_when_slope_is_unsafe() -> None:
    candidates = [
        {
            "method": "raw",
            "metrics": {
                "log_loss": 0.5000,
                "brier": 0.1900,
                "calibration_slope": 1.21,
                "calibration_intercept": 0.0,
            },
        },
        {
            "method": "sigmoid",
            "metrics": {
                "log_loss": 0.5001,
                "brier": 0.1901,
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
        },
    ]

    selected = GradientBoostingModel._select_probability_calibration_candidate(
        candidates,
        require_safe_calibration=True,
    )

    assert selected["method"] == "sigmoid"
    assert candidates[0]["selection_eligible"] is False
    assert candidates[1]["selection_eligible"] is True


def test_winner_calibration_uses_raw_when_no_fitted_candidate_is_safe() -> None:
    candidates = [
        {
            "method": "raw",
            "metrics": {
                "log_loss": 0.50,
                "brier": 0.19,
                "calibration_slope": 1.0,
                "calibration_intercept": -0.11,
            },
        }
    ]

    selected = GradientBoostingModel._select_probability_calibration_candidate(
        candidates,
        require_safe_calibration=True,
    )

    assert selected["method"] == "raw"
    assert selected["selection_safe"] is False
    assert selected["selection_eligible"] is True
    assert selected["selection_fallback"] is True


def test_winner_calibration_rejects_unsafe_fitted_candidate_without_raw() -> None:
    candidates = [
        {
            "method": "isotonic",
            "metrics": {
                "log_loss": 0.50,
                "brier": 0.19,
                "calibration_slope": 1.3,
                "calibration_intercept": 0.0,
            },
        }
    ]

    with pytest.raises(RuntimeError, match="raw identity fallback"):
        GradientBoostingModel._select_probability_calibration_candidate(
            candidates,
            require_safe_calibration=True,
        )


def test_rating_baseline_can_remain_an_explicitly_unsafe_comparator() -> None:
    candidates = [
        {
            "method": "raw",
            "metrics": {
                "log_loss": 0.61,
                "brier": 0.21,
                "calibration_slope": 0.7,
                "calibration_intercept": 0.12,
            },
        }
    ]

    selected = GradientBoostingModel._select_probability_calibration_candidate(
        candidates,
        require_safe_calibration=False,
    )

    assert selected["method"] == "raw"
    assert selected["selection_eligible"] is False
    assert selected["selection_reasons"]


def test_probability_uncertainty_uses_held_out_residual_intervals() -> None:
    probabilities = np.tile(np.array([0.3, 0.7]), 50)
    actual = np.tile(np.array([0, 1]), 50)

    uncertainty = ProbabilityUncertaintyModel.fit(
        actual,
        probabilities,
        timestamps=pd.date_range("2026-01-01", periods=len(actual), freq="D"),
    )
    lower, upper = uncertainty.interval(np.array([0.3, 0.7]))

    assert uncertainty.fit_split == "uncertainty_fit"
    assert uncertainty.sample_count == UNCERTAINTY_SAMPLE_COUNT
    assert uncertainty.calibration_units > MIN_UNCERTAINTY_WEEK_BLOCKS
    assert np.all(lower <= probabilities[:2])
    assert np.all(upper >= probabilities[:2])
    assert np.all((upper - lower) > 0)


def test_conservative_probability_requires_week_block_coverage() -> None:
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    metadata = pd.DataFrame({"date": dates, "actionable": True, "league": "LCK"})
    actual = np.tile([0, 1], 35)

    passing = conservative_probability_coverage(
        actual, np.full(len(actual), 0.40), metadata
    )
    failing = conservative_probability_coverage(
        actual, np.full(len(actual), 0.90), metadata
    )

    assert passing["passed"] is True
    assert passing["cohorts"]["actionable"]["eligible"] is True
    assert failing["passed"] is False
    assert failing["cohorts"]["aggregate"]["shortfall"] > MAX_COVERAGE_SHORTFALL


def test_conservative_probability_coverage_avoids_downcasting_warning() -> None:
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    metadata = pd.DataFrame(
        {
            "date": dates,
            "actionable": pd.Series(([1.0, None] * 35), dtype=object),
            "league": "LCK",
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        conservative_probability_coverage(
            np.tile([0, 1], 35), np.full(70, 0.40), metadata
        )

    assert not any(isinstance(item.message, FutureWarning) for item in captured)


def test_metadata_probability_calibrator_skips_sparse_segments(
    tmp_path, monkeypatch
) -> None:
    model = _calibration_model(tmp_path, monkeypatch)
    X_fit, y_fit, meta_fit = _segmented_probability_data(segment_rows=20)
    X_select, y_select, meta_select = _segmented_probability_data(segment_rows=20)
    X_full, y_full, meta_full = _segmented_probability_data(segment_rows=20)

    calibrator = model.fit_probability_calibrator(
        ProbabilityColumnModel(),
        X_fit,
        y_fit,
        X_select,
        y_select,
        X_full,
        y_full,
        meta_fit,
        meta_select,
        meta_full,
    )

    assert isinstance(calibrator, MetadataAwareProbabilityCalibrator)
    assert calibrator.segments == {}


def test_metadata_probability_calibrator_keeps_global_fallback_for_dense_segments(
    tmp_path, monkeypatch
) -> None:
    model = _calibration_model(tmp_path, monkeypatch)
    X_fit, y_fit, meta_fit = _segmented_probability_data(segment_rows=SEGMENT_FIT_ROWS)
    X_select, y_select, meta_select = _segmented_probability_data(
        segment_rows=SEGMENT_SELECT_ROWS
    )
    X_full, y_full, meta_full = _segmented_probability_data(
        segment_rows=SEGMENT_FIT_ROWS + SEGMENT_SELECT_ROWS
    )

    calibrator = model.fit_probability_calibrator(
        ProbabilityColumnModel(),
        X_fit,
        y_fit,
        X_select,
        y_select,
        X_full,
        y_full,
        meta_fit,
        meta_select,
        meta_full,
    )

    assert isinstance(calibrator, MetadataAwareProbabilityCalibrator)
    assert calibrator.segments == {}
    assert (
        calibrator.report["segment_status"]
        == "disabled_without_independent_composite_gate"
    )


def test_segment_calibration_does_not_select_isotonic_below_threshold(
    tmp_path, monkeypatch
) -> None:
    model = _calibration_model(tmp_path, monkeypatch)
    X_fit, y_fit, meta_fit = _segmented_probability_data(segment_rows=SEGMENT_FIT_ROWS)
    X_select, y_select, meta_select = _segmented_probability_data(
        segment_rows=SEGMENT_SELECT_ROWS
    )
    X_full, y_full, meta_full = _segmented_probability_data(
        segment_rows=SEGMENT_FIT_ROWS + SEGMENT_SELECT_ROWS
    )

    calibrator = model.fit_probability_calibrator(
        ProbabilityColumnModel(),
        X_fit,
        y_fit,
        X_select,
        y_select,
        X_full,
        y_full,
        meta_fit,
        meta_select,
        meta_full,
    )

    segment_methods = {
        candidate["method"]
        for report in calibrator.report["segment_reports"]
        for candidate in report["candidate_metrics"]
    }
    assert "isotonic" not in segment_methods


def test_prop_distribution_calibrator_prices_over_under() -> None:
    calibrator = PropDistributionCalibrator(
        method="empirical_global",
        global_residuals=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
    )

    signal = calibrator.price(mean=27.0, line=26.5)

    assert signal.over_probability + signal.under_probability == 1
    assert signal.over_probability > 0
    assert signal.under_probability > 0
    assert not hasattr(signal, "over_edge")
    assert not hasattr(signal, "over_half_kelly_fraction")


def test_prop_distribution_calibrator_uses_metadata_segments() -> None:
    calibrator = PropDistributionCalibrator(
        method="metadata_shrunk",
        global_residuals=np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        segment_residuals={
            ("bo_format", "bo3"): np.array([-3.0, -2.0, -1.0, -0.5] * 10)
        },
        min_league_samples=10,
        shrinkage_samples=10,
    )

    global_signal = calibrator.price(mean=27.0, line=27.0)
    segment_signal = calibrator.price(
        mean=27.0,
        line=27.0,
        metadata={"is_bo3": 1, "league": "LCK"},
    )

    assert segment_signal.over_probability < global_signal.over_probability


def test_prop_calibrator_artifact_paths_exist_for_all_prop_targets() -> None:
    assert GAMELENGTH_PREDICTION_PROP_CALIBRATOR.name.endswith("_prop_calibrator.pkl")
    assert TOTAL_KILLS_PREDICTION_PROP_CALIBRATOR.name.endswith("_prop_calibrator.pkl")
    assert TOTAL_TOWERS_PREDICTION_PROP_CALIBRATOR.name.endswith("_prop_calibrator.pkl")
