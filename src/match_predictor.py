"""
Oracle's Elixir Match Predictor

This script is intended to query the data lake and predict upcoming games using an ensemble of models.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.logger import logger
from utils.paths import EGPM_DOM_LOGISTIC, MIXED_VALIDATOR_WEIGHTS
from utils.team import Team
from utils.utils import load_model, setup_pandas


@dataclass
class MatchPredictor:

    egpm_model: object = field(default=None, init=False)
    weights: dict = field(default=None, init=False)

    def __post_init__(self):
        self.egpm_model = load_model(EGPM_DOM_LOGISTIC)
        self.weights = load_model(MIXED_VALIDATOR_WEIGHTS)

    @staticmethod
    def is_iterable(obj):
        """Check if the object is iterable."""
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    @staticmethod
    def aggregate_stats(stats):
        """Aggregate statistics, sum if iterable, else pass through."""
        return sum(stats) if MatchPredictor.is_iterable(stats) else stats

    @staticmethod
    def side_wr_prediction(team1_side_wr, team2_side_wr) -> float:
        """Predict winning likelihood based on side win rates."""
        return round(
            MatchPredictor.aggregate_stats(team1_side_wr)
            / (MatchPredictor.aggregate_stats(team1_side_wr) + MatchPredictor.aggregate_stats(team2_side_wr) + 1e-8),
            3,
        )

    @staticmethod
    def elo_prediction(team1_elo, team2_elo) -> float:
        """Predict winning likelihood based on ELO ratings."""
        return 1 / (
            1 + 10 ** ((MatchPredictor.aggregate_stats(team2_elo) - MatchPredictor.aggregate_stats(team1_elo)) / 400)
        )

    @staticmethod
    def pl_trueskill_prediction(
        team1_mu: List[float],
        team1_sigma: List[float],
        team2_mu: List[float],
        team2_sigma: List[float],
        is_player: bool = True,
        sigma_value: float = 25 / 3,
    ) -> float:
        """Predict winning likelihood using TrueSkill ratings."""
        delta_mu = MatchPredictor.aggregate_stats(team1_mu) - MatchPredictor.aggregate_stats(team2_mu)
        sum_sigma = sum(r**2 for r in team1_sigma + team2_sigma) if is_player else team1_sigma + team2_sigma
        size = 10 if is_player else 2
        return norm.cdf(delta_mu / np.sqrt(size * (sigma_value / 2) ** 2 + sum_sigma))

    def egpm_dom_prediction(self, team1_egpm_dom: List[float], team2_egpm_dom: List[float]) -> float:
        """Predict winning likelihood based on EGPM dominance."""
        X = np.array([[MatchPredictor.aggregate_stats(team1_egpm_dom), MatchPredictor.aggregate_stats(team2_egpm_dom)]])
        X = pd.DataFrame(
            X,
            columns=[
                "egpm_dominance_ratio_ema_before",
                "egpm_opp_dominance_ratio_ema_before",
            ],
        )
        return self.egpm_model.predict_proba(X)[:, 1][0]

    def predict_match(self, team1: Team, team2: Team) -> str:
        """
        Predict the outcome of a match between two teams.
        """
        match = pd.DataFrame(
            {
                "team1": [team1.name],
                "team2": [team2.name],
                "team_elo": [
                    self.elo_prediction(
                        team1.team_stats.team_elo,
                        team2.team_stats.team_elo,
                    )
                ],
                "player_elo": [
                    self.elo_prediction(
                        team1.player_stats.player_elo,
                        team2.player_stats.player_elo,
                    )
                ],
                "trueskill": [
                    self.pl_trueskill_prediction(
                        team1.team_stats.team_trueskill_sum_mu,
                        team1.team_stats.team_trueskill_sigma_squared,
                        team2.team_stats.team_trueskill_sum_mu,
                        team2.team_stats.team_trueskill_sigma_squared,
                        is_player=False,
                    )
                ],
                "team_pl": [
                    self.pl_trueskill_prediction(
                        team1.team_stats.team_pl_mu,
                        team1.team_stats.team_pl_sigma,
                        team2.team_stats.team_pl_mu,
                        team2.team_stats.team_pl_sigma,
                        is_player=False,
                    )
                ],
                "player_pl": [
                    self.pl_trueskill_prediction(
                        team1.player_stats.player_pl_mu,
                        team1.player_stats.player_pl_sigma,
                        team2.player_stats.player_pl_mu,
                        team2.player_stats.player_pl_sigma,
                    )
                ],
                "team_egpm_dom": [
                    self.egpm_dom_prediction(
                        team1.team_stats.egpm_dominance_ratio,
                        team2.team_stats.egpm_dominance_ratio,
                    )
                ],
                "player_egpm_dom": [
                    self.egpm_dom_prediction(
                        team1.player_stats.egpm_dominance_ratio,
                        team2.player_stats.egpm_dominance_ratio,
                    )
                ],
                "team_side_win": [
                    self.side_wr_prediction(
                        (team1.team_stats.blue_side_wr if team1.side == "Blue" else team1.team_stats.red_side_wr),
                        (team2.team_stats.blue_side_wr if team2.side == "Blue" else team2.team_stats.red_side_wr),
                    )
                ],
                "player_side_win": [
                    self.side_wr_prediction(
                        (team1.player_stats.blue_side_wr if team1.side == "Blue" else team1.player_stats.red_side_wr),
                        (team2.player_stats.blue_side_wr if team2.side == "Blue" else team2.player_stats.red_side_wr),
                    )
                ],
            }
        )

        # Calculate the sum of accuracy weights
        sum_accuracy = sum(self.weights.values())
        predictive_attributes = [col for col in match.columns if col not in ["team1", "team2"]]

        # Calculate win chance and deviation
        match["team1_win_chance"] = sum(
            (match[attr] * (self.weights[f"{attr}"] / sum_accuracy)).iloc[0] for attr in predictive_attributes
        )
        match["deviation"] = match[predictive_attributes].std(axis=1)

        return match

    def best_of_three(self, t1name, t1odds, t2name, t2odds):
        """
        Calculate the likelihood of each team winning a best of three series.
        """
        # Team 1 2/0:
        t1_20 = t1odds**2

        # Team 1 2/1
        t1_21 = 2 * t1odds**2 * t2odds

        # Team 2 2/0
        t2_20 = t2odds**2

        # Team 2 2/1
        t2_21 = 2 * t2odds**2 * t1odds

        # Final Outputs
        doublecheck = t1_20 + t1_21 + t2_20 + t2_21
        assert round(doublecheck, 5) == 1.0, "Probabilities do not sum to 1"

        output = (
            f"```Overall Likelihood Of {t1name} To Win Series: {((t1_20 + t1_21) * 100):.2f}% \n"
            f"Probability {t1name} wins 2/0: {(t1_20 * 100):.2f}% \n"
            f"Probability {t1name} wins 2/1: {(t1_21 * 100):.2f}% \n \n"
            f"Overall Likelihood Of {t2name} To Win Series: {((t2_20 + t2_21) * 100):.2f}% \n"
            f"Probability {t2name} wins 2/0: {(t2_20 * 100):.2f}% \n"
            f"Probability {t2name} wins 2/1: {(t2_21 * 100):.2f}%```"
        )
        return output

    def best_of_five(self, t1name, t1odds, t2name, t2odds):
        """
        Calculate the likelihood of each team winning a best of five series.
        """
        # Team 1 3/0:
        t1_30 = t1odds**3

        # Team 1 3/1
        t1_31 = 3 * t1odds**3 * t2odds

        # Team 1 3/2
        t1_32 = 6 * t1odds**3 * t2odds**2

        # Team 2 3/0
        t2_30 = t2odds**3

        # Team 2 3/1
        t2_31 = 3 * t2odds**3 * t1odds

        # Team 2 3/2
        t2_32 = 6 * t2odds**3 * t1odds**2

        # Final Outputs
        doublecheck = t1_30 + t1_31 + t1_32 + t2_30 + t2_31 + t2_32
        assert round(doublecheck, 5) == 1.0, "Probabilities do not sum to 1"

        # Final Outputs
        doublecheck = t1_30 + t1_31 + t1_32 + t2_30 + t2_31 + t2_32
        assert round(doublecheck, 5) == 1.0
        t1likelihood = (t1_30 + t1_31 + t1_32) * 100
        t2likelihood = (t2_30 + t2_31 + t2_32) * 100

        output = (
            f"```Overall Likelihood Of {t1name} To Win Series: {t1likelihood:.2f}% \n"
            f"Probability {t1name} wins 3/0: {(t1_30 * 100):.2f}% \n"
            f"Probability {t1name} wins 3/1: {(t1_31 * 100):.2f}% \n"
            f"Probability {t1name} wins 3/2: {(t1_32 * 100):.2f}% \n \n"
            f"Overall Likelihood Of {t2name} To Win Series: {t2likelihood:.2f}% \n"
            f"Probability {t2name} wins 3/0: {(t2_30 * 100):.2f}% \n"
            f"Probability {t2name} wins 3/1: {(t2_31 * 100):.2f}% \n"
            f"Probability {t2name} wins 3/2: {(t2_32 * 100):.2f}%```"
        )
        return output


def main():
    setup_pandas(pd)

    # Example usage of the refactored MatchPredictor class
    predictor = MatchPredictor()

    print()
    logger.info("Match Predictor initialized successfully.\n")

    # Teams creation and prediction example
    # Example team names and player names should be replaced with real ones
    team1 = Team(name="G2 Esports")
    team2 = Team(name="T1", side="Red")

    team1.display_team_info()
    team2.display_team_info()
    print()

    # Predict a match
    logger.info(predictor.predict_match(team1, team2))


if __name__ == "__main__":
    main()
