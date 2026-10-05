from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from lol_bets.data_generation.feature_engineering.performance_features.season_win_rate import (
    EPSILON as SEASON_WIN_EPSILON,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    DEFAULT_POSITION_WEIGHT,
    POSITION_WEIGHTS,
    expected_outcome,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    DEFAULT_MU,
    DEFAULT_PHI,
    DEFAULT_SIGMA,
    Glicko2,
    Rating,
)
from lol_bets.inference.match_predictor import MatchPredictor, _glicko2_prob
from lol_bets.prediction_models.gbdt_model import GradientBoostingModel
from lol_bets.prediction_models.prop_features import build_game_level_outcome_features
from oracle_bets_core.pd import pd
from pandas.api.types import is_numeric_dtype

EVEN_PROBABILITY = 0.5
MODEL_PROBABILITY = 0.87654321
FIRST_MAP = 1
SECOND_MAP = 2
OWN_EMA_KDA = 3.0
OPP_EMA_KDA = 2.0
OWN_SEASON_WIN_RATE = 0.8
OPP_SEASON_WIN_RATE = 0.2
OWN_GOLD_EMA = 10.0
OPP_GOLD_EMA = 8.0


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


def _full_players(strength: float) -> pd.DataFrame:
    frames = []
    for index, role in enumerate(("top", "jng", "mid", "bot", "sup")):
        row = _players(strength + index).iloc[[0]].copy()
        row["playername"] = role
        row["position"] = role
        frames.append(row)
    return pd.concat(frames, ignore_index=True)


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


def test_player_elo_likelihood_matches_weighted_training_total() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    own = _full_players(0)
    opponent = _full_players(200)
    actual, _ = predictor.apply_player_stat_modifications(own, opponent)

    def rating_total(rows: pd.DataFrame) -> float:
        weights = rows["position"].map(
            lambda role: POSITION_WEIGHTS.get(role, DEFAULT_POSITION_WEIGHT)
        )
        return float((rows["elo"] * weights).sum() / weights.sum() * len(rows))

    expected = expected_outcome(rating_total(own), rating_total(opponent), 400.0)
    assert actual["elo_win_likelihood"].iloc[0] == pytest.approx(expected, abs=1e-12)


def test_glicko_likelihood_uses_opponent_mean_impact() -> None:
    model = Glicko2(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)
    blue = Rating(1600.0, 50.0)
    red = Rating(1500.0, 150.0)
    expected = model.expect_score(blue, red, model.reduce_impact(red))

    assert _glicko2_prob(1600.0, 50.0, 1500.0, 150.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_red_player_likelihood_is_blue_complement() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    blue, _ = predictor.apply_player_stat_modifications(
        _players(0), _players(200), side="Blue"
    )
    red, _ = predictor.apply_player_stat_modifications(
        _players(200), _players(0), side="Red"
    )

    for name in (
        "elo_win_likelihood",
        "glicko2_win_likelihood",
        "pl_win_likelihood",
        "trueskill_win_likelihood",
    ):
        assert red[name].iloc[0] == pytest.approx(1.0 - blue[name].iloc[0])


def test_prop_features_include_player_and_rating_consensus() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    blue = SimpleNamespace(side="Blue", team_stats={"league": "LCK"})
    red = SimpleNamespace(side="Red", team_stats={"league": "LCK"})

    def side_features(team, _opponent, _account_for_side):
        _ = _opponent, _account_for_side
        strength = 0.7 if team.side == "Blue" else 0.3
        return pd.DataFrame(
            {
                "gameid": ["live"],
                "side": [team.side],
                "elo_win_likelihood": [strength],
                "glicko2_win_likelihood": [strength],
                "top_elo_win_likelihood": [strength],
                "jng_elo_win_likelihood": [strength],
            }
        )

    predictor.calculate_team_and_player_stats = side_features
    features = predictor.calculate_prop_features(blue, red, account_for_side=False)

    assert features.loc[0, "blue_players_elo_win_likelihood"] == pytest.approx(0.7)
    assert features.loc[0, "red_players_elo_win_likelihood"] == pytest.approx(0.3)
    assert features.loc[0, "blue_rating_consensus"] == pytest.approx(0.7)


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


def test_player_ema_opponent_uses_training_feature_names() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    players = pd.DataFrame(
        {
            "gameid": ["future"],
            "side": ["Blue"],
            "position": ["top"],
            "ema_kda": [OWN_EMA_KDA],
            "opp_ema_kda": [OPP_EMA_KDA],
        }
    )
    teams = pd.DataFrame({"gameid": ["future"], "side": ["Blue"]})

    out = predictor.preprocess_data(teams, players)

    assert out.loc[0, "top_opp_ema_kda"] == OPP_EMA_KDA
    assert out.loc[0, "top_diff_ema_kda"] == OWN_EMA_KDA - OPP_EMA_KDA


def test_player_pivot_preserves_missing_opponent_ema() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    roles = ("top", "jng", "mid", "bot", "sup")
    players = pd.DataFrame(
        {
            "gameid": ["future"] * len(roles),
            "side": ["Blue"] * len(roles),
            "position": roles,
            "ema_kda": [OWN_EMA_KDA] * len(roles),
            "opp_ema_kda": [np.nan, *([OPP_EMA_KDA] * (len(roles) - 1))],
        }
    )
    teams = pd.DataFrame({"gameid": ["future"], "side": ["Blue"]})

    out = predictor.preprocess_data(teams, players)

    assert np.isnan(out.loc[0, "top_opp_ema_kda"])
    assert np.isnan(out.loc[0, "top_diff_ema_kda"])
    assert out.loc[0, "jng_diff_ema_kda"] == OWN_EMA_KDA - OPP_EMA_KDA


def test_team_likelihood_remains_numeric_and_is_not_fused_with_opponent() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.league_elo_prediction = lambda *_: EVEN_PROBABILITY
    predictor.strength_pool_prediction = lambda *_: EVEN_PROBABILITY
    own = SimpleNamespace(
        name="Alpha",
        side="Blue",
        team_stats=pd.Series(
            {
                "teamid": "alpha",
                "gameid": "future",
                "side": "Blue",
                "league": "LCK",
                "ema_season_win_rate": OWN_SEASON_WIN_RATE,
                "ema_gold": OWN_GOLD_EMA,
            }
        ),
    )
    opponent = SimpleNamespace(
        name="Beta",
        side="Red",
        team_stats=pd.Series(
            {
                "teamid": "beta",
                "gameid": "future",
                "side": "Red",
                "league": "LCK",
                "ema_season_win_rate": OPP_SEASON_WIN_RATE,
                "ema_gold": OPP_GOLD_EMA,
            }
        ),
    )

    team_row = predictor.calculate_team_stats(own, opponent, False, "bo3")
    fused = GradientBoostingModel.fuse_opposing_team_features(team_row)

    assert is_numeric_dtype(team_row["season_win_likelihood"])
    assert "opp_season_win_likelihood" not in team_row
    assert fused.loc[0, "season_win_likelihood"] == pytest.approx(
        OWN_SEASON_WIN_RATE
        / (OWN_SEASON_WIN_RATE + OPP_SEASON_WIN_RATE + SEASON_WIN_EPSILON),
        abs=1e-12,
    )
    assert fused.loc[0, "diff_ema_gold"] == OWN_GOLD_EMA - OPP_GOLD_EMA
    opposite_row = predictor.calculate_team_stats(opponent, own, False, "bo3")
    sides = pd.concat(
        [fused, GradientBoostingModel.fuse_opposing_team_features(opposite_row)],
        ignore_index=True,
    )
    sides = sides.drop(columns=GradientBoostingModel._meta_columns(), errors="ignore")
    metadata = pd.DataFrame(
        {
            "gameid": ["future", "future"],
            "teamid": ["alpha", "beta"],
            "teamname": ["Alpha", "Beta"],
        }
    )
    matchup, _, _ = build_game_level_outcome_features(sides, metadata)
    assert matchup.loc[0, "delta_season_win_likelihood"] == pytest.approx(
        (OWN_SEASON_WIN_RATE - OPP_SEASON_WIN_RATE)
        / (OWN_SEASON_WIN_RATE + OPP_SEASON_WIN_RATE + SEASON_WIN_EPSILON),
        abs=1e-12,
    )


def test_player_feature_assembly_rejects_duplicate_or_missing_roles() -> None:
    predictor = MatchPredictor.__new__(MatchPredictor)
    duplicate = _full_players(0)
    duplicate.loc[duplicate["position"] == "sup", "position"] = "mid"
    team_a = SimpleNamespace(
        name="Alpha", player_stats=duplicate, team_stats={}, side="Blue"
    )
    team_b = SimpleNamespace(
        name="Beta", player_stats=_full_players(10), team_stats={}, side="Red"
    )

    with pytest.raises(ValueError, match="exactly one row per role"):
        predictor.calculate_player_stats(team_a, team_b)


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
    assert (
        forward["uncertainty_method"]
        == "week_block_members_plus_held_out_calibration_bias"
    )
    assert forward["team1_probability_lower"] == pytest.approx(
        forward["team1_win_probability"] - 0.10
    )
    assert forward["drivers"]
    assert "not causal proof" in forward["drivers"][0]


def test_map_prediction_is_research_only_and_exactly_complementary() -> None:
    class _MapModel:
        @staticmethod
        def predict_proba(frame):
            probability = 0.7 if frame.loc[0, "delta_rating"] > 0 else 0.3
            return np.array([[1.0 - probability, probability]])

    class _IdentityPipeline:
        train_columns = ("delta_rating",)

        @staticmethod
        def transform(frame):
            return frame

    class _Uncertainty:
        method = "held_out_calibration_residual_mean"
        confidence = 0.9
        sample_count = 80

        @staticmethod
        def interval(probabilities):
            return probabilities - 0.05, probabilities + 0.05

    predictor = MatchPredictor.__new__(MatchPredictor)
    predictor.outcome_model = _MapModel()
    predictor.outcome_calibrator = SimpleNamespace(
        predict=lambda values, **_kwargs: np.asarray(values) + 0.1
    )
    predictor.outcome_uncertainty = _Uncertainty()
    predictor.outcome_pipeline = _IdentityPipeline()
    predictor._outcome_team_features = lambda team, _opponent, **_kwargs: pd.DataFrame(
        {"rating": [team.rating]}
    )
    alpha = SimpleNamespace(name="Alpha", rating=1.0, team_stats={"teamid": "a"})
    zulu = SimpleNamespace(name="Zulu", rating=2.0, team_stats={"teamid": "z"})

    forward = predictor.predict_map(alpha, zulu)
    reverse = predictor.predict_map(zulu, alpha)

    assert forward["team1_win_probability"] == reverse["team2_win_probability"]
    assert forward["team2_win_probability"] == reverse["team1_win_probability"]
    assert forward["team1_probability_lower"] == reverse["team2_probability_lower"]
    assert forward["model_target"] == "map_winner"
    assert forward["research_mode"] == "shadow_only"
    assert forward["raw_model_probability"] == pytest.approx(0.3)
    assert forward["team1_win_probability"] == pytest.approx(0.4)
    assert reverse["raw_model_probability"] == pytest.approx(0.7)


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
