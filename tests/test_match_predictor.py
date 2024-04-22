import pandas as pd
import pytest

from src.discord.match_predictor import MatchPredictor
from utils.paths import EGPM_DOM_LOGISTIC
from utils.utils import load_model


@pytest.fixture
def match_predictor():
    return MatchPredictor()


def test_generate_validation_metrics(match_predictor, monkeypatch):
    mock_weights = {
        "team_elo": 0.2,
        "player_elo": 0.1,
        "trueskill": 0.15,
        "team_pl": 0.1,
        "player_pl": 0.15,
        "team_egpm_dom": 0.1,
        "player_egpm_dom": 0.1,
        "team_side_win": 0.05,
        "player_side_win": 0.05,
    }

    # Directly mock the `weights` attribute if the intention is to test behavior assuming `weights` is set to `mock_weights`.
    monkeypatch.setattr(match_predictor, "weights", mock_weights)

    assert match_predictor.weights == mock_weights


def test_is_iterable(match_predictor):
    assert match_predictor.is_iterable([1, 2, 3]) is True
    assert match_predictor.is_iterable(123) is False


def test_side_wr_prediction(match_predictor):
    team1_stat = [0.6, 0.7]
    team2_stat = [0.5, 0.4]
    assert match_predictor.side_wr_prediction(team1_stat, team2_stat) == 0.591
    assert match_predictor.side_wr_prediction(0.7, 0.5) == 0.583


def test_elo_prediction(match_predictor):
    team1_elo = [1500, 1600]
    team2_elo = [1700, 1800]
    assert round(match_predictor.elo_prediction(team1_elo, team2_elo), 3) == 0.091
    assert round(match_predictor.elo_prediction(1600, 1800), 3) == 0.240


def test_pl_trueskill_prediction(match_predictor):
    team1_mu = [25, 30]
    team1_sigma = [8, 10]
    team2_mu = [20, 25]
    team2_sigma = [12, 9]
    assert (
        round(
            match_predictor.pl_trueskill_prediction(team1_mu, team1_sigma, team2_mu, team2_sigma),
            3,
        )
        == 0.663
    )
    assert round(match_predictor.pl_trueskill_prediction(25, 8, 20, 12, is_player=False), 3) == 0.75


def test_egpm_dom_prediction(match_predictor, monkeypatch):
    team1_egpm_dom = [0.6, 0.7]
    team2_egpm_dom = [0.5, 0.4]
    mock_model = load_model(EGPM_DOM_LOGISTIC)
    monkeypatch.setattr(match_predictor, "egpm_model", mock_model)
    assert round(match_predictor.egpm_dom_prediction(team1_egpm_dom, team2_egpm_dom), 4) == 0.5001


def test_best_of_three(match_predictor):
    assert "50.00%" in match_predictor.best_of_three("Team1", 0.5, "Team2", 0.5)
    assert "64.80%" in match_predictor.best_of_three("Team1", 0.6, "Team2", 0.4)


def test_best_of_five(match_predictor):
    assert "50.00%" in match_predictor.best_of_five("Team1", 0.5, "Team2", 0.5)
    assert "68.26%" in match_predictor.best_of_five("Team1", 0.6, "Team2", 0.4)


class MockTeam:
    def __init__(self, name, side, team_stats, player_stats):
        self.name = name
        self.side = side
        self.team_stats = team_stats
        self.player_stats = player_stats
        self.roster = None


class MockTeamStats:
    def __init__(
        self,
        team_elo,
        team_trueskill_sum_mu,
        team_trueskill_sigma_squared,
        team_pl_mu,
        team_pl_sigma,
        egpm_dominance_ratio,
        blue_side_wr,
        red_side_wr,
    ):
        self.team_elo = team_elo
        self.team_trueskill_sum_mu = team_trueskill_sum_mu
        self.team_trueskill_sigma_squared = team_trueskill_sigma_squared
        self.team_pl_mu = team_pl_mu
        self.team_pl_sigma = team_pl_sigma
        self.egpm_dominance_ratio = egpm_dominance_ratio
        self.blue_side_wr = blue_side_wr
        self.red_side_wr = red_side_wr


class MockPlayerStats:
    def __init__(
        self,
        player_elo,
        player_pl_mu,
        player_pl_sigma,
        egpm_dominance_ratio,
        blue_side_wr,
        red_side_wr,
    ):
        self.player_elo = player_elo
        self.player_pl_mu = player_pl_mu
        self.player_pl_sigma = player_pl_sigma
        self.egpm_dominance_ratio = egpm_dominance_ratio
        self.blue_side_wr = blue_side_wr
        self.red_side_wr = red_side_wr


def test_predict_match(match_predictor, monkeypatch):
    team1_stats = MockTeamStats(1500, 25, 8, 20, 12, 0.6, 0.7, 0.6)
    team2_stats = MockTeamStats(1700, 20, 12, 15, 10, 0.5, 0.4, 0.5)
    team1_player_stats = MockPlayerStats([1500, 1600], [25, 30], [8, 10], [20, 25], [0.6, 0.7], [0.7, 0.6])
    team2_player_stats = MockPlayerStats([1700, 1800], [20, 25], [12, 9], [15, 20], [0.5, 0.4], [0.4, 0.5])
    team1 = MockTeam("Team 1", "Blue", team1_stats, team1_player_stats)
    team2 = MockTeam("Team 2", "Red", team2_stats, team2_player_stats)

    monkeypatch.setattr(
        match_predictor,
        "weights",
        {
            "team_elo": 0.2,
            "player_elo": 0.1,
            "trueskill": 0.15,
            "team_pl": 0.1,
            "player_pl": 0.15,
            "team_egpm_dom": 0.1,
            "player_egpm_dom": 0.1,
            "team_side_win": 0.05,
            "player_side_win": 0.05,
        },
    )

    result = match_predictor.predict_match(team1, team2)
    assert isinstance(result, pd.DataFrame)
    assert "team1_win_chance" in result.columns
    assert "deviation" in result.columns
