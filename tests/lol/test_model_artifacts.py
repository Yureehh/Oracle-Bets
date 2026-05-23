import numpy as np
from lol_bets.prediction_models.gbdt_model import GradientBoostingModel
from oracle_bets_core.pd import pd

PAIRWISE_GAME_COUNT = 2
EXPECTED_ROW_ACCURACY = 0.75
EXPECTED_PAIRWISE_LOG_LOSS_UPPER_BOUND = 0.7


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
