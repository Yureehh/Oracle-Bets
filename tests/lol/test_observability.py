from __future__ import annotations

import json

import numpy as np
from lol_bets.prediction_models import observability
from lol_bets.prediction_models.observability import MLObservabilityMixin
from oracle_bets_core.pd import pd


class _Observable(MLObservabilityMixin):
    model_name = "ObservableModel"
    problem_type = "classification"
    run_id = "test-run"


SHA256_HEX_LENGTH = 64
EXPECTED_TOP_GAIN = 9.0


def test_calibration_table_serializes_interval_bins_as_text(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "INSIGHTS_DIR", tmp_path)
    model = _Observable()

    table = model.store_calibration_table(
        pd.Series([0, 0, 1, 1, 1, 0]),
        np.array([0.05, 0.20, 0.55, 0.65, 0.85, 0.95]),
        n_bins=3,
    )
    stored = pd.read_parquet(
        tmp_path / "ObservableModel" / "test-run" / "calibration_table.parquet"
    )

    assert table["bin"].dtype == object
    assert stored["bin"].dtype == object
    assert stored["bin"].str.contains(",").all()
    assert {"bin_mean_p", "bin_emp_rate", "n", "ece"}.issubset(stored.columns)


def test_model_card_uses_stable_feature_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "INSIGHTS_DIR", tmp_path)
    model = _Observable()

    model.store_model_card(
        run_id=model.run_id,
        data_window=None,
        n_rows_train=1,
        n_rows_val=1,
        n_rows_test=1,
        features=["b", "a"],
        data_hash="data-sha",
        code_version="code-sha",
    )
    card = json.loads(
        (
            tmp_path / "ObservableModel" / "test-run" / "model_card_test-run.json"
        ).read_text()
    )

    assert len(card["feature_hash"]) == SHA256_HEX_LENGTH
    assert card["data_hash"] == "data-sha"
    assert card["code_version"] == "code-sha"


def test_feature_importance_stores_gain_and_split_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "INSIGHTS_DIR", tmp_path)
    monkeypatch.setattr(observability, "FIGURES_DIR", tmp_path / "figures")

    class _Booster:
        @staticmethod
        def feature_importance(*, importance_type):
            return (
                np.array([1.0, EXPECTED_TOP_GAIN])
                if importance_type == "gain"
                else np.array([4.0, 2.0])
            )

    class _Model:
        booster_ = _Booster()

    model = _Observable()
    model.store_feature_importance(_Model(), ["rating_delta", "form_delta"])
    stored = pd.read_parquet(
        tmp_path / "ObservableModel" / "test-run" / "feature_importances.parquet"
    )

    assert stored.iloc[0]["feature"] == "form_delta"
    assert stored.iloc[0]["gain"] == EXPECTED_TOP_GAIN
    assert {
        "feature",
        "gain",
        "gain_fraction",
        "split",
        "split_fraction",
    } == set(stored.columns)
