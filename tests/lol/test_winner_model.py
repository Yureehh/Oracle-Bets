from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from lol_bets.operations.models import CandidateManifest, ModelRegistry
from lol_bets.operations.winner_validation import validate_winner_model
from lol_bets.prediction_models.winner_model import (
    DIRECT_RATING_STATE_COLUMNS,
    WinnerBlendClassifier,
    WinnerLightGBMModel,
)
from oracle_bets_core.pd import pd


class _ProbabilityModel:
    def __init__(self, values: list[float], importances: list[float] | None = None):
        self.values = np.asarray(values, dtype=float)
        self.feature_importances_ = np.asarray(importances or [1.0, 1.0])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.resize(self.values, len(frame))
        return np.column_stack([1.0 - values, values])


def _winner_calibration_trainer() -> WinnerLightGBMModel:
    trainer = object.__new__(WinnerLightGBMModel)
    trainer.calibration = "auto"
    trainer.calibration_method = "raw"
    return trainer


def _balanced_probability_rows() -> tuple[pd.DataFrame, pd.Series]:
    probabilities = np.tile([0.2, 0.8], 40)
    return (
        pd.DataFrame({"probability": probabilities}),
        pd.Series(np.tile([0, 1], 40)),
    )


def test_serving_blend_requires_safe_calibration_selection() -> None:
    trainer = _winner_calibration_trainer()
    X, y = _balanced_probability_rows()
    model = SimpleNamespace(
        blend_weight=1.0,
        predict_proba=lambda frame: np.column_stack(
            [1.0 - frame["probability"], frame["probability"]]
        ),
    )
    observed: list[bool] = []

    def stop_after_selection(_candidates, *, require_safe_calibration):
        observed.append(require_safe_calibration)
        raise RuntimeError("selection observed")

    trainer._select_probability_calibration_candidate = stop_after_selection

    with pytest.raises(RuntimeError, match="selection observed"):
        trainer.fit_probability_calibrator(model, X, y, X, y, X, y)

    assert observed == [True]


def test_rating_baseline_is_a_transparent_comparator_not_a_serving_gate() -> None:
    trainer = _winner_calibration_trainer()
    trainer._store_winner_report = lambda *_args, **_kwargs: None
    X, y = _balanced_probability_rows()
    model = SimpleNamespace(
        baseline=_ProbabilityModel([0.2, 0.8]),
        rating_columns=("probability",),
    )
    observed: list[bool] = []

    def select_first(candidates, *, require_safe_calibration):
        observed.append(require_safe_calibration)
        candidates[0]["selection_eligible"] = False
        candidates[0]["selection_reasons"] = ["comparator_not_in_safe_band"]
        return candidates[0]

    trainer._select_probability_calibration_candidate = select_first

    calibrator = trainer._fit_rating_baseline_calibrator(model, X, y, X, y, X, y)

    assert calibrator is not None
    assert observed == [False]


class _OffsetCalibrator:
    @staticmethod
    def predict(values):
        return np.asarray(values) + 0.02


class _ValidationCalibrator:
    version = 3

    @staticmethod
    def predict(values, metadata=None):
        del metadata
        return np.asarray(values)


class _ValidationUncertainty:
    version = 2
    fit_split = "uncertainty_fit"
    sample_count = 30
    calibration_units = 4
    method = "week_block_q10_plus_one_sided_calibration_bias"
    unit = "iso_week"

    @staticmethod
    def interval(values):
        return values, values


class _WinnerContractModel:
    def __init__(self, columns):
        self.rating_columns = tuple(columns)
        self.members = tuple(
            SimpleNamespace(feature_name_=list(columns)) for _ in range(10)
        )

    @staticmethod
    def component_probabilities(_frame):
        return None

    @staticmethod
    def rating_baseline_probability(_frame):
        return np.asarray([0.55])

    @staticmethod
    def conservative_probability(_frame):
        return None

    @staticmethod
    def predict_proba(_frame):
        return np.asarray([[0.4, 0.6]])


def test_winner_blend_exposes_predeclared_components() -> None:
    frame = pd.DataFrame(
        {
            "delta_elo_win_likelihood": [0.1, -0.1],
            "delta_glicko2_win_likelihood": [0.2, -0.2],
        }
    )
    baseline = _ProbabilityModel([0.55, 0.45])
    members = (_ProbabilityModel([0.70, 0.30]), _ProbabilityModel([0.60, 0.40]))
    model = WinnerBlendClassifier(
        baseline=baseline,
        members=members,
        rating_columns=tuple(frame.columns),
        blend_weight=0.5,
    )

    rating, full = model.component_probabilities(frame)
    point = model.predict_proba(frame)[:, 1]

    assert rating.tolist() == pytest.approx([0.55, 0.45])
    assert full.tolist() == pytest.approx([0.65, 0.35])
    assert point[0] == pytest.approx(1.0 - point[1])
    assert rating[0] < point[0] < full[0]


def test_rating_baseline_has_its_own_calibration() -> None:
    frame = pd.DataFrame({"delta_elo_win_likelihood": [0.1]})
    model = WinnerBlendClassifier(
        baseline=_ProbabilityModel([0.55]),
        members=(_ProbabilityModel([0.60]),),
        rating_columns=tuple(frame.columns),
        baseline_calibrator=_OffsetCalibrator(),
    )

    assert model.rating_baseline_probability(frame)[0] == pytest.approx(0.57)


def test_conservative_probability_uses_member_quantile() -> None:
    frame = pd.DataFrame({"delta_elo_win_likelihood": [0.1]})
    model = WinnerBlendClassifier(
        baseline=_ProbabilityModel([0.60]),
        members=(
            _ProbabilityModel([0.58]),
            _ProbabilityModel([0.62]),
            _ProbabilityModel([0.70]),
        ),
        rating_columns=tuple(frame.columns),
        blend_weight=0.5,
    )

    conservative = model.conservative_probability(frame)

    expected_members = [
        1
        / (
            1
            + np.exp(
                -(
                    0.5 * np.log(0.60 / 0.40)
                    + 0.5 * np.log(probability / (1.0 - probability))
                )
            )
        )
        for probability in (0.58, 0.62, 0.70)
    ]
    assert conservative[0] == pytest.approx(np.quantile(expected_members, 0.1))


def test_compact_schema_wins_when_development_loss_is_noninferior() -> None:
    selected = WinnerLightGBMModel._choose_feature_schema(
        {
            "full": [0.60, 0.62, 0.61],
            "compact": [0.603, 0.621, 0.609],
        }
    )

    assert selected == "compact"


def test_full_schema_wins_when_compact_materially_regresses() -> None:
    selected = WinnerLightGBMModel._choose_feature_schema(
        {
            "full": [0.60, 0.62, 0.61],
            "compact": [0.63, 0.64, 0.62],
        }
    )

    assert selected == "full"


def test_schema_selection_reads_only_development_partitions(tmp_path) -> None:
    trainer = object.__new__(WinnerLightGBMModel)
    trainer.fit_partition_metadata = {
        "train": pd.DataFrame(
            {"date": pd.to_datetime(["2026-01-01", "2026-01-08"], utc=True)},
            index=[0, 1],
        ),
        "tune": pd.DataFrame(
            {"date": pd.to_datetime(["2026-01-15", "2026-01-22"], utc=True)},
            index=[2, 3],
        ),
        "test": pd.DataFrame(
            {"date": pd.to_datetime(["2026-02-01"], utc=True)}, index=[99]
        ),
    }
    trainer.insight_path = lambda name: tmp_path / name
    observed = []

    def schema_losses(schema, features, _target, _weeks, _blocks):
        observed.append((schema, tuple(features.index)))
        return [0.60, 0.61, 0.62]

    trainer._development_schema_losses = schema_losses
    selected = trainer.select_development_feature_schema(
        pd.DataFrame({"signal": [1.0, 2.0]}, index=[0, 1]),
        pd.Series([0, 1], index=[0, 1]),
        pd.DataFrame({"signal": [3.0, 4.0]}, index=[2, 3]),
        pd.Series([0, 1], index=[2, 3]),
    )

    assert selected == "compact"
    assert observed == [("full", (0, 1, 2, 3)), ("compact", (0, 1, 2, 3))]
    assert (
        json.loads((tmp_path / "feature_schema_study.json").read_text())[
            "final_test_used"
        ]
        is False
    )


def test_winner_validation_requires_ratings_parity_and_no_market_features(tmp_path):
    columns = (
        *DIRECT_RATING_STATE_COLUMNS,
        "delta_elo_win_likelihood",
        "delta_glicko2_win_likelihood",
        "delta_pl_win_likelihood",
        "delta_trueskill_win_likelihood",
    )
    model_path = tmp_path / "winner.pkl"
    pipeline_path = tmp_path / "pipeline.pkl"
    schema_path = tmp_path / "schema.pkl"
    lineage_path = tmp_path / "lineage.json"
    calibrator_path = tmp_path / "calibrator.pkl"
    uncertainty_path = tmp_path / "uncertainty.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_WinnerContractModel(columns), handle)
    with pipeline_path.open("wb") as handle:
        pickle.dump(SimpleNamespace(train_columns=list(columns)), handle)
    with schema_path.open("wb") as handle:
        pickle.dump({"version": 1, "canonical_key": "teamid_then_teamname"}, handle)
    with calibrator_path.open("wb") as handle:
        pickle.dump(_ValidationCalibrator(), handle)
    with uncertainty_path.open("wb") as handle:
        pickle.dump(_ValidationUncertainty(), handle)
    lineage_path.write_text(
        json.dumps(
            [
                {
                    "feature": column,
                    "source": column,
                    "availability_timestamp": "strictly_before_fixture_start",
                    "family": "ratings",
                    "swap_behavior": "canonical_team_a_minus_team_b",
                    "model_eligible": True,
                }
                for column in columns
            ]
        )
    )
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"safe")
    registry.register_candidate(
        CandidateManifest(
            model_id="winner-v2",
            sport="lol",
            target="series_winner",
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_version="abc",
            data_manifest="data",
            feature_fingerprint="features",
            config_hash="config",
            dependency_lock_hash="lock",
            random_seed=7,
            metrics={"log_loss": 0.6},
        ),
        {"artifact.bin": artifact},
    )
    registry.promote(
        "winner-v2",
        promoted_at=datetime(2026, 8, 12, tzinfo=UTC),
        reason="test fixture",
    )

    report = validate_winner_model(
        model_path=model_path,
        pipeline_path=pipeline_path,
        schema_path=schema_path,
        lineage_path=lineage_path,
        calibrator_path=calibrator_path,
        uncertainty_path=uncertainty_path,
        registry_root=registry.root,
    )

    assert report.ok
