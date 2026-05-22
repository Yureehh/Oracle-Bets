import numpy as np
from lol_bets.prediction_models.gbdt_model import GradientBoostingModel
from oracle_bets_core.pd import pd


def test_regression_residual_summary_contains_pricing_fields():
    summary = GradientBoostingModel.build_residual_summary(
        pd.Series([10.0, 12.0, 14.0]),
        np.array([9.0, 12.5, 13.0]),
        model_name="TotalKillsPrediction_LightGBM",
    )

    assert summary["residual_sigma"] > 0
    assert summary["mae"] > 0
    assert "50" in summary["percentiles"]
