from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from lol_bets.inference.match_predictor import MatchPredictor
from oracle_bets_core.pd import pd

EVEN_PROBABILITY = 0.5
MODEL_PROBABILITY = 0.87654321
FIRST_MAP = 1
SECOND_MAP = 2


def _players(strength: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-24", "2026-05-24"],
            "playername": ["top", "jng"],
            "position": ["top", "jng"],
            "gameid": ["future", "future"],
            "teamname": ["Team", "Team"],
            "side": ["Blue", "Blue"],
            "elo": [1500 + strength, 1510 + strength],
            "glicko2_mu": [1500 + strength, 1510 + strength],
            "glicko2_phi": [60.0, 60.0],
            "pl_mu": [20 + strength / 100, 21 + strength / 100],
            "pl_sigma": [1.5, 1.5],
            "trueskill_mu": [25 + strength / 100, 26 + strength / 100],
            "trueskill_sigma": [1.5, 1.5],
            "ema_kda": [3.0 + strength / 100, 3.1 + strength / 100],
        }
    )


def test_player_lineup_likelihoods_are_available_for_inference() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)

    weaker, stronger = predictor.apply_player_stat_modifications(
        _players(0),
        _players(200),
    )
    stronger_flipped, _ = predictor.apply_player_stat_modifications(
        _players(200),
        _players(0),
    )

    for col in [
        "elo_win_likelihood",
        "glicko2_win_likelihood",
        "pl_win_likelihood",
        "trueskill_win_likelihood",
    ]:
        assert col in weaker.columns
        assert col in stronger_flipped.columns
        assert weaker[col].iloc[0] < EVEN_PROBABILITY
        assert stronger_flipped[col].iloc[0] > EVEN_PROBABILITY

    assert "glicko2_mu" not in weaker.columns
    assert "opp_ema_kda" in stronger.columns


def test_predict_outcomes_preserves_probability_precision() -> None:
    class _OutcomeModel:
        def predict_proba(self, _):
            return np.array([[1.0 - MODEL_PROBABILITY, MODEL_PROBABILITY]])

    def _keep_necessary_columns(X, *, model_name):
        _ = model_name
        return X

    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.outcome_model = _OutcomeModel()
    predictor.outcome_calibrator = None
    predictor.outcome_pipeline = None
    predictor.keep_necessary_columns = _keep_necessary_columns

    out = predictor.predict_outcomes(pd.DataFrame({"feature": [1.0]}))

    assert out[0, 1] == MODEL_PROBABILITY


def test_player_pivot_drops_target_columns_for_inference() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    player_data = pd.DataFrame(
        {
            "gameid": ["future"],
            "side": ["Blue"],
            "position": ["top"],
            "result": [1],
            "gamelength": [31.0],
            "total_kills": [20.0],
            "total_towers": [11.0],
            "ema_kda": [3.0],
        }
    )

    out = predictor.pivot_player_data(player_data)

    assert "top_ema_kda" in out.columns
    assert "top_result" not in out.columns
    assert "top_gamelength" not in out.columns
    assert "top_total_kills" not in out.columns
    assert "top_total_towers" not in out.columns


def test_team_series_context_overrides_stale_flattened_values() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.league_elo_prediction = lambda *_: EVEN_PROBABILITY
    predictor.strength_pool_prediction = lambda *_: EVEN_PROBABILITY
    stale = {
        "teamid": "team-a",
        "league": "LCK",
        "side": "Blue",
        "game_in_series": 3,
        "is_bo1": 1,
        "is_bo3": 1,
        "is_bo5": 0,
        "is_deciding_game": 1,
        "ema_season_win_rate": EVEN_PROBABILITY,
    }
    opponent = {
        "teamid": "team-b",
        "league": "LCK",
        "side": "Red",
        "ema_season_win_rate": EVEN_PROBABILITY,
    }

    out = predictor.apply_stat_modifications(
        pd.Series(stale),
        pd.Series(opponent),
        account_for_side=False,
        match_type="bo5",
    )

    assert out["game_in_series"] == FIRST_MAP
    assert out["is_bo1"] == 0
    assert out["is_bo3"] == 0
    assert out["is_bo5"] == 1
    assert out["is_deciding_game"] == 0


def test_team_matchup_recomputes_and_retains_direct_rating_state() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.league_elo_prediction = lambda *_: EVEN_PROBABILITY
    predictor.strength_pool_prediction = lambda *_: EVEN_PROBABILITY
    base = {
        "league": "LCK",
        "side": "Blue",
        "ema_season_win_rate": EVEN_PROBABILITY,
        "glicko2_phi": 60.0,
        "pl_sigma": 1.5,
        "trueskill_sigma": 1.5,
    }
    stronger = pd.Series(
        {
            **base,
            "teamid": "stronger",
            "elo": 1700.0,
            "glicko2_mu": 1700.0,
            "pl_mu": 27.0,
            "trueskill_mu": 27.0,
        }
    )
    weaker = pd.Series(
        {
            **base,
            "teamid": "weaker",
            "elo": 1500.0,
            "glicko2_mu": 1500.0,
            "pl_mu": 25.0,
            "trueskill_mu": 25.0,
        }
    )

    out = predictor.apply_stat_modifications(
        stronger, weaker, account_for_side=False, match_type="bo3"
    )

    for column in (
        "elo",
        "glicko2_mu",
        "glicko2_phi",
        "pl_mu",
        "pl_sigma",
        "trueskill_mu",
        "trueskill_sigma",
    ):
        assert column in out
    for column in (
        "elo_win_likelihood",
        "glicko2_win_likelihood",
        "pl_win_likelihood",
        "trueskill_win_likelihood",
    ):
        assert out[column] > EVEN_PROBABILITY


def test_team_series_context_defaults_to_unknown_format() -> None:
    assert MatchPredictor.series_context(None) == {
        "game_in_series": FIRST_MAP,
        "is_bo1": 0,
        "is_bo3": 0,
        "is_bo5": 0,
        "is_deciding_game": 0,
    }


def test_match_prediction_is_exactly_complementary_when_team_order_reverses() -> None:
    class _OutcomeModel:
        blend_weight = 0.5

        def predict_proba(self, X):
            probability = 0.3 if X.loc[0, "delta_rating"] < 0 else 0.7
            return np.array([[1.0 - probability, probability]])

        def component_probabilities(self, X):
            probability = self.predict_proba(X)[:, 1]
            return probability, probability

        def rating_baseline_probability(self, X):
            return self.predict_proba(X)[:, 1]

        def conservative_interval(self, X, **_kwargs):
            probability = self.predict_proba(X)[:, 1]
            return probability - 0.05, probability + 0.05

        def predict(self, X, *, pred_contrib: bool):
            assert pred_contrib
            return np.array([[1.5] * len(X.columns) + [0.0]])

    class _IdentityPipeline:
        def transform(self, X):
            return X

    class _Uncertainty:
        method = "held_out_calibration_residual_mean"
        confidence = 0.9
        sample_count = 80

        def interval(self, probabilities):
            return probabilities - 0.05, probabilities + 0.05

    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.series_winner_model = _OutcomeModel()
    predictor.series_winner_calibrator = None
    predictor.series_winner_uncertainty = _Uncertainty()
    predictor.series_winner_pipeline = _IdentityPipeline()
    predictor._outcome_team_features = lambda team, _opponent, **_kwargs: pd.DataFrame(
        {"rating": [team.rating], "first_pick": [0.5]}
    )
    alpha = SimpleNamespace(name="Alpha", rating=1.0, team_stats={"teamid": "a"})
    zulu = SimpleNamespace(name="Zulu", rating=2.0, team_stats={"teamid": "z"})

    forward = predictor.predict_match(alpha, zulu)
    reverse = predictor.predict_match(zulu, alpha)

    assert forward["team1_win_probability"] == reverse["team2_win_probability"]
    assert forward["team2_win_probability"] == reverse["team1_win_probability"]
    assert forward["team1_win_probability"] + forward["team2_win_probability"] == 1.0
    assert forward["team1_probability_lower"] == reverse["team2_probability_lower"]
    assert forward["team1_probability_upper"] == reverse["team2_probability_upper"]
    assert forward["uncertainty_method"] == "week_block_member_quantile_with_bias_bound"
    assert forward["drivers"]
    assert "not causal proof" in forward["drivers"][0]


def test_next_map_prediction_is_shadow_only_and_exactly_complementary() -> None:
    class _NextMapModel:
        @staticmethod
        def predict_proba(frame):
            probability = 0.7 if frame.loc[0, "delta_rating"] > 0 else 0.3
            return np.array([[1.0 - probability, probability]])

    class _NextMapPipeline:
        @staticmethod
        def transform(frame):
            return frame

    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.next_map_winner_model = _NextMapModel()
    predictor.next_map_winner_calibrator = None
    predictor.next_map_winner_pipeline = _NextMapPipeline()
    predictor._outcome_team_features = lambda team, _opponent, **_kwargs: pd.DataFrame(
        {"rating": [team.rating]}
    )
    alpha = SimpleNamespace(name="Alpha", rating=1.0, team_stats={"teamid": "a"})
    zulu = SimpleNamespace(name="Zulu", rating=2.0, team_stats={"teamid": "z"})

    forward = predictor.predict_next_map(
        alpha, zulu, best_of=3, team1_maps=1, team2_maps=0
    )
    reverse = predictor.predict_next_map(
        zulu, alpha, best_of=3, team1_maps=0, team2_maps=1
    )

    assert forward["team1_win_probability"] == reverse["team2_win_probability"]
    assert forward["team2_win_probability"] == reverse["team1_win_probability"]
    assert forward["next_map_number"] == SECOND_MAP
    assert forward["research_mode"] == "shadow_only"
