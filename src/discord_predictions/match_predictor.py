"""
Lol Oracle Match Predictor

This script is intended to query the data lake and predict upcoming games using GDBT prediction model.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from feature_engineering.ratings_features.glicko import DEFAULT_MU, DEFAULT_PHI, DEFAULT_SIGMA, Glicko2
from feature_engineering.ratings_features.glicko import Rating as GlickoRating
from feature_engineering.ratings_features.glicko import calculate_mean_rating
from feature_engineering.ratings_features.plackett_luce import PlackettLuce
from feature_engineering.ratings_features.plackett_luce import predict_win_probability as pl_win_probability
from feature_engineering.ratings_features.trueskill import Rating as TrueskillRating
from feature_engineering.ratings_features.trueskill import win_probability as trueskill_win_probability
from prediction_models.gbdt_model import GradientBoostingModel
from src.utils.paths import (
    LEAGUE_ELO,
    OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
    OUTCOME_PREDICTION_FINAL_FEATURES,
    OUTCOME_PREDICTION_MODEL_PATH,
    TEAM_LEAGUES_MAPPING,
)
from src.utils.team import Team
from src.utils.utils import load_model

# Constants
PREDICTION_PRECISION = 3
ELO_FACTOR = 400
RATING_DECIMALS = 3


@dataclass
class MatchPredictor:
    outcome_prediction_model: Optional[dict] = field(default=None, init=False)
    team_to_league: Optional[pd.DataFrame] = field(default=None, init=False)
    league_to_elo: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self):
        """Initializes the models from predefined paths."""
        self.load_models_and_data()

    def load_models_and_data(self):
        """Lazy loading of models and dataframes."""
        if self.outcome_prediction_model is None:
            self.outcome_prediction_model = load_model(OUTCOME_PREDICTION_MODEL_PATH)
        if self.team_to_league is None:
            self.team_to_league = pd.read_parquet(TEAM_LEAGUES_MAPPING)
        if self.league_to_elo is None:
            self.league_to_elo = pd.read_parquet(LEAGUE_ELO)

    @staticmethod
    def is_iterable(obj: Any) -> bool:
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
    def elo_prediction(team1_elo: List[float], team2_elo: List[float]) -> float:
        """Predict based on ELO rating differences."""
        team1_elo_sum = MatchPredictor.aggregate_stats(team1_elo)
        team2_elo_sum = MatchPredictor.aggregate_stats(team2_elo)
        prediction = 1 / (1 + 10 ** ((team2_elo_sum - team1_elo_sum) / ELO_FACTOR))
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def gl2_prediction(
        team1_mus: List[float], team1_phis: List[float], team2_mus: List[float], team2_phis: List[float]
    ) -> float:
        """Predict based on GL2 (Glicko-2) ratings."""
        blue_ratings = [GlickoRating(mu, phi) for mu, phi in zip(team1_mus, team1_phis)]
        red_ratings = [GlickoRating(mu, phi) for mu, phi in zip(team2_mus, team2_phis)]
        model = Glicko2(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)

        mean_blue_rating = calculate_mean_rating(blue_ratings)
        mean_red_rating = calculate_mean_rating(red_ratings)
        mean_blue_impact = sum(model.reduce_impact(rating) for rating in blue_ratings) / len(blue_ratings)
        prediction = model.expect_score(mean_blue_rating, mean_red_rating, mean_blue_impact)
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def pl_prediction(
        team1_mus: List[float], team1_sigmas: List[float], team2_mus: List[float], team2_sigmas: List[float]
    ) -> float:
        """Predict based on Plackett-Luce model."""
        model = PlackettLuce()
        team1_ratings = [model.rating(mu, sigma) for mu, sigma in zip(team1_mus, team1_sigmas)]
        team2_ratings = [model.rating(mu, sigma) for mu, sigma in zip(team2_mus, team2_sigmas)]
        prediction = pl_win_probability(model, team1_ratings, team2_ratings)[0]
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def trueskill_prediction(
        team1_mus: List[float], team1_sigmas: List[float], team2_mus: List[float], team2_sigmas: List[float]
    ) -> float:
        """Predict based on TrueSkill ratings."""
        team1_ratings = [TrueskillRating(mu, sigma) for mu, sigma in zip(team1_mus, team1_sigmas)]
        team2_ratings = [TrueskillRating(mu, sigma) for mu, sigma in zip(team2_mus, team2_sigmas)]
        prediction = trueskill_win_probability(team1_ratings, team2_ratings)
        return round(prediction, RATING_DECIMALS)

    def whr_prediction(self, blue_name: int, red_name: int) -> float:
        """Predict winning likelihood based on Whole History Rating (WHR)."""
        win_likelihood = self.whr_model.probability_future_match(blue_name, red_name)
        return round(win_likelihood[0], RATING_DECIMALS)

    def league_elo_prediction(self, team1_id: float, team2_id: float) -> float:
        """Predict winning likelihood based on league ELO ratings."""
        try:
            team1_league = self.team_to_league.loc[self.team_to_league["teamid"] == team1_id, "league"].values[0]
            team2_league = self.team_to_league.loc[self.team_to_league["teamid"] == team2_id, "league"].values[0]
        except IndexError as e:
            raise ValueError(f"Team ID not found: {e}") from e

        try:
            team1_league_elo = self.league_to_elo.loc[self.league_to_elo["league"] == team1_league, "elo"].values[0]
            team2_league_elo = self.league_to_elo.loc[self.league_to_elo["league"] == team2_league, "elo"].values[0]
        except IndexError as e:
            raise ValueError(f"League not found in ELO ratings: {e}") from e

        prediction = 1 / (1 + 10 ** ((team2_league_elo - team1_league_elo) / ELO_FACTOR))
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def side_wr_prediction(team1_side_wr: float, team2_side_wr: float) -> float:
        """Predict winning likelihood based on side win rates."""
        prediction = team1_side_wr / (team1_side_wr + team2_side_wr + 1e-8)
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def patch_season_wr_prediction(team1_patch_wr: float, team2_patch_wr: float) -> float:
        """Predict winning likelihood based on patch win rates."""
        prediction = team1_patch_wr / (team1_patch_wr + team2_patch_wr + 1e-8)
        return round(prediction, RATING_DECIMALS)

    def apply_stat_modifications(
        self, team1_stats: pd.Series, team2_stats: pd.Series, team1_side: str, team2_side: str, account_for_side: bool
    ) -> None:
        """Applies statistical predictions to modify team statistics based on game side and other predictive metrics."""
        team1_stats["elo_win_likelihood"] = self.elo_prediction(team1_stats["elo"], team2_stats["elo"])
        team1_stats["gl2_win_likelihood"] = self.gl2_prediction(
            [team1_stats["gl2_mu"]], [team1_stats["gl2_phi"]], [team2_stats["gl2_mu"]], [team2_stats["gl2_phi"]]
        )
        team1_stats["pl_win_likelihood"] = self.pl_prediction(
            [team1_stats["pl_mu"]], [team1_stats["pl_sigma"]], [team2_stats["pl_mu"]], [team2_stats["pl_sigma"]]
        )
        team1_stats["trueskill_win_likelihood"] = self.trueskill_prediction(
            [team1_stats["trueskill_mu"]],
            [team1_stats["trueskill_sigma"]],
            [team2_stats["trueskill_mu"]],
            [team2_stats["trueskill_sigma"]],
        )
        team1_stats["league_elo_win_likelihood"] = self.league_elo_prediction(
            team1_stats["teamid"], team2_stats["teamid"]
        )
        team1_stats["side_win_likelihood"] = (
            self.side_wr_prediction(
                team1_stats[f"ema_{team1_side.lower()}_side"], team2_stats[f"ema_{team2_side.lower()}_side"]
            )
            if account_for_side
            else 0.5
        )
        team1_stats["patch_win_likelihood"] = self.patch_season_wr_prediction(
            team1_stats["ema_patch_win_rate"], team2_stats["ema_patch_win_rate"]
        )
        team1_stats["season_win_likelihood"] = self.patch_season_wr_prediction(
            team1_stats["ema_season_win_rate"], team2_stats["ema_season_win_rate"]
        )

        drop_columns = [
            "teamid",
            "date",
            "ema_red_side",
            "ema_blue_side",
            "league_elo",
            "elo",
            "gl2_mu",
            "gl2_phi",
            "pl_mu",
            "pl_sigma",
            "trueskill_mu",
            "trueskill_sigma",
        ]
        team1_stats.drop(labels=drop_columns, errors="ignore", inplace=True)
        team2_stats.drop(labels=drop_columns + ["teamname", "gameid"], errors="ignore", inplace=True)

    def calculate_team_stats(self, team1: Team, team2: Team, account_for_side: bool) -> pd.DataFrame:
        """Combines and modifies team stats for further processing and analysis."""
        team1_stats, team2_stats = team1.team_stats.copy(), team2.team_stats.copy()
        self.apply_stat_modifications(team1_stats, team2_stats, team1.side, team2.side, account_for_side)
        team2_stats.index = [f"opp_{col}" for col in team2_stats.index]
        return pd.concat([team1_stats.to_frame().T, team2_stats.to_frame().T], axis=1)

    def apply_player_stat_modifications(self, player1_stats: pd.DataFrame, player2_stats: pd.DataFrame) -> None:
        """Applies statistical predictions to modify player statistics based on game side and other predictive metrics."""
        player1_stats["elo_win_likelihood"] = self.elo_prediction(player1_stats["elo"], player2_stats["elo"])
        player1_stats["gl2_win_likelihood"] = self.gl2_prediction(
            player1_stats["gl2_mu"], player1_stats["gl2_phi"], player2_stats["gl2_mu"], player2_stats["gl2_phi"]
        )
        player1_stats["pl_win_likelihood"] = self.pl_prediction(
            player1_stats["pl_mu"], player1_stats["pl_sigma"], player2_stats["pl_mu"], player2_stats["pl_sigma"]
        )
        player1_stats["trueskill_win_likelihood"] = self.trueskill_prediction(
            player1_stats["trueskill_mu"],
            player1_stats["trueskill_sigma"],
            player2_stats["trueskill_mu"],
            player2_stats["trueskill_sigma"],
        )

        player_drop_columns = [
            "date",
            "playername",
            "elo",
            "gl2_mu",
            "gl2_phi",
            "pl_mu",
            "pl_sigma",
            "trueskill_mu",
            "trueskill_sigma",
        ]
        player1_stats.drop(columns=player_drop_columns, inplace=True)
        player2_stats.drop(columns=player_drop_columns + ["gameid", "teamname"], inplace=True)

    def calculate_player_stats(self, team1: Team, team2: Team) -> pd.DataFrame:
        """Handles player statistics similar to team statistics with modifications based on game side."""
        player1_stats, player2_stats = team1.player_stats.copy(), team2.player_stats.copy()
        self.apply_player_stat_modifications(player1_stats, player2_stats)
        player2_stats.columns = [f"opp_{col}" if col != "position" else col for col in player2_stats.columns]
        return pd.merge(player1_stats, player2_stats, on="position", how="inner", validate="many_to_many")

    def pivot_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Pivots player data to prepare for merging with team data by grouping numeric and non-numeric columns."""
        numeric_cols = player_data.select_dtypes(include=["number"]).columns
        non_numeric_cols = player_data.columns.difference(numeric_cols)

        agg_funcs = {col: "mean" for col in numeric_cols}
        agg_funcs.update({col: "first" for col in non_numeric_cols if col not in ["position", "side"]})

        player_data_pivoted = player_data.pivot_table(
            index=["gameid", "teamname"], columns="position", aggfunc=agg_funcs, fill_value=0
        )
        player_data_pivoted.columns = [f"{col[1]}_{col[0]}" for col in player_data_pivoted.columns]

        return player_data_pivoted.reset_index()

    def merge_datasets(self, team_data: pd.DataFrame, player_data: pd.DataFrame) -> pd.DataFrame:
        """Merges team data with pivoted player data by game ID and team name, removing redundant columns."""
        prediction_data = pd.merge(
            team_data, player_data, on=["gameid", "teamname"], how="inner", validate="many_to_many"
        )

        columns_to_drop = ["gameid", "teamname"] + [
            f"{pos}_{field}" for pos in ["top", "jng", "mid", "bot", "sup"] for field in ["gameid", "teamname"]
        ]

        return prediction_data.drop(columns=columns_to_drop, errors="ignore")

    def preprocess_data(self, team_data: pd.DataFrame, player_data: pd.DataFrame) -> pd.DataFrame:
        """Processes team and player data by pivoting player data and merging it with team data."""
        player_data_processed = self.pivot_player_data(player_data)
        return self.merge_datasets(team_data, player_data_processed)

    def keep_necessary_columns(self, final_stats: pd.DataFrame) -> pd.DataFrame:
        """Keeps only the necessary columns for the prediction model."""
        with open(OUTCOME_PREDICTION_FINAL_FEATURES, "rb") as f:
            final_features = pickle.load(f)
        final_stats = final_stats[final_features]
        return self.convert_data_types(final_stats)

    def convert_data_types(self, final_stats: pd.DataFrame) -> pd.DataFrame:
        """Converts columns to numeric where possible, otherwise converts to category type."""
        with open(OUTCOME_PREDICTION_CATEGORICAL_FEATURES, "rb") as f:
            categorical_features = pickle.load(f)

        final_stats = final_stats.copy()
        for col in final_stats.columns:
            if col in categorical_features:
                final_stats[col] = final_stats[col].astype("category")
            else:
                final_stats[col] = pd.to_numeric(final_stats[col], errors="coerce")

        return final_stats

    def predict_outcomes(self, final_stats: pd.DataFrame) -> np.ndarray:
        """Uses the predictive model to estimate outcomes based on preprocessed final stats."""
        final_stats = GradientBoostingModel.process_players_likelihood_columns(final_stats)
        final_stats = GradientBoostingModel.fuse_opposing_team_features(final_stats)
        final_stats = self.keep_necessary_columns(final_stats)
        return self.outcome_prediction_model.predict_proba(final_stats).round(PREDICTION_PRECISION)

    def predict_match(self, team1: Team, team2: Team, account_for_side: bool = True) -> dict:
        """Main method to predict the outcome of a match between two teams."""
        team_stats = self.calculate_team_stats(team1, team2, account_for_side)
        player_stats = self.calculate_player_stats(team1, team2)
        final_stats = self.preprocess_data(team_stats, player_stats)
        return self.predict_outcomes(final_stats)
