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
        X = np.array(
            [[MatchPredictor.aggregate_stats(team1_egpm_dom) - MatchPredictor.aggregate_stats(team2_egpm_dom)]]
        )
        X = pd.DataFrame(
            X,
            columns=[
                "egpm_dominance_diff",
            ],
        )
        return self.egpm_model.predict_proba(X)[:, 1][0]

    def _adjust_weights_for_side_exclusion(self, predictive_attributes):
        # Calculate total weights to be remapped
        weights_to_remap = self.weights.get("team_side_win", 0) + self.weights.get("player_side_win", 0)

        # Evenly distribute the remapped weights among remaining predictive attributes
        weight_addition_per_attribute = weights_to_remap / len(predictive_attributes)
        for attr in predictive_attributes:
            self.weights[attr] += weight_addition_per_attribute

    def predict_match(self, team1: Team, team2: Team, account_for_side: bool = True) -> pd.DataFrame:
        """
        Predict the outcome of a match between two teams.
        """
        match = pd.DataFrame(
            {
                "team1": [team1.name],
                "team2": [team2.name],
                "team1_roster": [team1.roster],
                "team2_roster": [team2.roster],
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
        # Drop the index column
        match.reset_index(drop=True, inplace=True)
        match = match.round(3)

        # Define columns that do not contribute to predictive attributes
        non_predictive_columns = ["team1", "team2", "team1_roster", "team2_roster"]

        # Identify predictive attributes by excluding non-predictive columns
        predictive_attributes = [col for col in match.columns if col not in non_predictive_columns]

        # Adjust weights and predictive attributes based on account_for_side flag
        if not account_for_side:
            match.drop(columns=["team_side_win", "player_side_win"], inplace=True)
            predictive_attributes = [
                attr for attr in predictive_attributes if attr not in ["team_side_win", "player_side_win"]
            ]
            self._adjust_weights_for_side_exclusion(predictive_attributes)

        # Calculate normalized weights
        total_weight = sum(self.weights[attr] for attr in predictive_attributes)
        normalized_weights = {attr: self.weights[attr] / total_weight for attr in predictive_attributes}

        # Calculate weighted sum for win chance
        match["team1_win_chance"] = (
            match[predictive_attributes].mul(list(normalized_weights.values()), axis=1).sum(axis=1)
        )

        # Calculate standard deviation of win chance
        match["deviation"] = np.sqrt(
            match[predictive_attributes].mul(list(normalized_weights.values()), axis=1).std(axis=1)
        )

        match["team2_win_chance"] = 1 - match["team1_win_chance"]
        return match

    @staticmethod
    def best_of_three(t1name, t1odds, t2name, t2odds):
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
            f"```Overall Likelihood Of {t1name} To Win Series: {((t1_20 + t1_21) * 100):.2f}% \n \n"
            f"\tProbability {t1name} wins 2/0: {(t1_20 * 100):.2f}% \n"
            f"\tProbability {t1name} wins 2/1: {(t1_21 * 100):.2f}% \n \n \n"
            f"Overall Likelihood Of {t2name} To Win Series: {((t2_20 + t2_21) * 100):.2f}% \n \n"
            f"\tProbability {t2name} wins 2/0: {(t2_20 * 100):.2f}% \n"
            f"\tProbability {t2name} wins 2/1: {(t2_21 * 100):.2f}% \n \n \n"
            f"Overall likelihoods of each team winning at least 1 game: \n \n"
            f"\tProbability {t1name} wins at least 1 game: {((t1_21 + t1_20 + t2_21) * 100):.2f}% \n"
            f"\tProbability {t2name} wins at least 1 game: {((t2_21 + t2_20 + t1_21) * 100):.2f}% \n \n \n"
            f"Overall Likelihood Of Exactly 3 Games: {((t1_21 + t2_21) * 100):.2f}%```"
        )
        return output

    @staticmethod
    def best_of_five(t1name, t1odds, t2name, t2odds):
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
            f"```Overall Likelihood Of {t1name} To Win Series: {t1likelihood:.2f}% \n\n"
            f"\tProbability {t1name} wins 3/0: {(t1_30 * 100):.2f}% \n"
            f"\tProbability {t1name} wins 3/1: {(t1_31 * 100):.2f}% \n"
            f"\tProbability {t1name} wins 3/2: {(t1_32 * 100):.2f}% \n \n \n"
            f"Overall Likelihood Of {t2name} To Win Series: {t2likelihood:.2f}% \n\n"
            f"\tProbability {t2name} wins 3/0: {(t2_30 * 100):.2f}% \n"
            f"\tProbability {t2name} wins 3/1: {(t2_31 * 100):.2f}% \n"
            f"\tProbability {t2name} wins 3/2: {(t2_32 * 100):.2f}% \n\n\n"
            f"Overall likelihoods of each team winning at least 1 game: \n\n"
            f"\tProbability {t1name} wins at least 1 game: {((t1_31 + t1_32 + t1_30 + t2_31 + t2_32) * 100):.2f}% \n"
            f"\tProbability {t2name} wins at least 1 game: {((t2_31 + t2_32 + t2_30 + t1_31 + t1_32) * 100):.2f}% \n\n\n"
            f"Overall Likelihood Of At Least 4 Games: {((t1_31 + t1_32 + t2_31 + t2_32) * 100):.2f}% \n\n\n"
            f"Overall Likelihood Of Exactly 5 Games: {((t1_32 + t2_32) * 100):.2f}%```"
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

    print()

    # Predict a match
    prediction = predictor.predict_match(team1, team2, account_for_side=True)
    logger.info(prediction)


if __name__ == "__main__":
    main()
