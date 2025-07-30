"""
Lol Oracle Match Predictor

This module predicts the outcomes of upcoming games using various rating
models and a Gradient Boosting Decision Tree (GBDT) prediction model.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from entities.team import Team
from feature_engineering.ratings_features.glicko import (
    DEFAULT_MU,
    DEFAULT_PHI,
    DEFAULT_SIGMA,
    Glicko2,
    calculate_mean_rating,
)
from feature_engineering.ratings_features.glicko import Rating as GlickoRating
from feature_engineering.ratings_features.plackett_luce import PlackettLuce
from feature_engineering.ratings_features.plackett_luce import (
    predict_win_probability as pl_win_probability,
)
from feature_engineering.ratings_features.trueskill import Rating as TrueskillRating
from feature_engineering.ratings_features.trueskill import (
    win_probability as trueskill_win_probability,
)
from prediction_models.gbdt_model import GradientBoostingModel
from utils.io_utils import load_model
from utils.paths import (
    LEAGUE_ELO,
    OUTCOME_PREDICTION_CATEGORICAL_FEATURES,
    OUTCOME_PREDICTION_FINAL_FEATURES,
    OUTCOME_PREDICTION_MODEL_PATH,
    TEAM_LEAGUES_MAPPING,
)

# Constants
PREDICTION_PRECISION = 3
ELO_FACTOR = 400
RATING_DECIMALS = 3


@dataclass
class MatchPredictor:
    """Predicts the outcome of matches between two teams using various rating models and a GBDT prediction model."""

    outcome_prediction_model: Any = field(default=None, init=False)
    team_to_league: pd.DataFrame = field(default=None, init=False)
    league_to_elo: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initializes the MatchPredictor by loading necessary models and data."""
        try:
            self.load_models_and_data()
        except Exception as e:
            msg = f"Failed to initialize MatchPredictor: {e}"
            raise RuntimeError(msg) from e

    def load_models_and_data(self) -> None:
        """Loads the outcome prediction model and necessary dataframes from predefined paths."""
        try:
            self.outcome_prediction_model = load_model(OUTCOME_PREDICTION_MODEL_PATH)
            self.team_to_league = pd.read_parquet(
                TEAM_LEAGUES_MAPPING, engine="fastparquet"
            )
            self.league_to_elo = pd.read_parquet(LEAGUE_ELO, engine="fastparquet")
        except FileNotFoundError as e:
            msg = f"Required file not found: {e}"
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Error loading models and data: {e}"
            raise RuntimeError(msg) from e

    @staticmethod
    def is_iterable(obj: Any) -> bool:
        """
        Checks if the provided object is iterable.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if iterable, False otherwise.

        """
        return isinstance(obj, list | tuple | np.ndarray)

    @staticmethod
    def aggregate_stats(stats: float | list[float]) -> float:
        """
        Aggregates statistics by summing if iterable, else returns the value as is.

        Args:
            stats (Union[float, List[float]]): The statistics to aggregate.

        Returns:
            float: Aggregated statistics.

        """
        if MatchPredictor.is_iterable(stats):
            return sum(stats)
        return stats

    @staticmethod
    def elo_prediction(
        team1_elo: float | list[float], team2_elo: float | list[float]
    ) -> float:
        """
        Predicts the win probability based on ELO rating differences.

        Args:
            team1_elo (Union[float, List[float]]): ELO ratings for Team 1.
            team2_elo (Union[float, List[float]]): ELO ratings for Team 2.

        Returns:
            float: Predicted win probability for Team 1.

        """
        team1_elo_sum = MatchPredictor.aggregate_stats(team1_elo)
        team2_elo_sum = MatchPredictor.aggregate_stats(team2_elo)
        prediction = 1 / (1 + 10 ** ((team2_elo_sum - team1_elo_sum) / ELO_FACTOR))
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def gl2_prediction(
        team1_mus: float | list[float],
        team1_phis: float | list[float],
        team2_mus: float | list[float],
        team2_phis: float | list[float],
    ) -> float:
        """
        Predicts the win probability based on Glicko-2 ratings.

        Args:
            team1_mus (Union[float, List[float]]): Glicko-2 mu values for Team 1.
            team1_phis (Union[float, List[float]]): Glicko-2 phi values for Team 1.
            team2_mus (Union[float, List[float]]): Glicko-2 mu values for Team 2.
            team2_phis (Union[float, List[float]]): Glicko-2 phi values for Team 2.

        Returns:
            float: Predicted win probability for Team 1.

        """
        if not MatchPredictor.is_iterable(team1_mus):
            team1_mus = [team1_mus]
            team1_phis = [team1_phis]
            team2_mus = [team2_mus]
            team2_phis = [team2_phis]

        blue_ratings = [
            GlickoRating(mu, phi)
            for mu, phi in zip(team1_mus, team1_phis, strict=False)
        ]
        red_ratings = [
            GlickoRating(mu, phi)
            for mu, phi in zip(team2_mus, team2_phis, strict=False)
        ]
        model = Glicko2(mu=DEFAULT_MU, phi=DEFAULT_PHI, sigma=DEFAULT_SIGMA)

        mean_blue_rating = calculate_mean_rating(blue_ratings)
        mean_red_rating = calculate_mean_rating(red_ratings)
        mean_blue_impact = sum(
            model.reduce_impact(rating) for rating in blue_ratings
        ) / len(blue_ratings)
        prediction = model.expect_score(
            mean_blue_rating, mean_red_rating, mean_blue_impact
        )
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def pl_prediction(
        team1_mus: float | list[float],
        team1_sigmas: float | list[float],
        team2_mus: float | list[float],
        team2_sigmas: float | list[float],
    ) -> float:
        """
        Predicts the win probability based on Plackett-Luce model.

        Args:
            team1_mus (Union[float, List[float]]): Plackett-Luce mu values for Team 1.
            team1_sigmas (Union[float, List[float]]): Plackett-Luce sigma values for Team 1.
            team2_mus (Union[float, List[float]]): Plackett-Luce mu values for Team 2.
            team2_sigmas (Union[float, List[float]]): Plackett-Luce sigma values for Team 2.

        Returns:
            float: Predicted win probability for Team 1.

        """
        if not MatchPredictor.is_iterable(team1_mus):
            team1_mus = [team1_mus]
            team1_sigmas = [team1_sigmas]
            team2_mus = [team2_mus]
            team2_sigmas = [team2_sigmas]

        model = PlackettLuce()
        team1_ratings = [
            model.rating(mu, sigma)
            for mu, sigma in zip(team1_mus, team1_sigmas, strict=False)
        ]
        team2_ratings = [
            model.rating(mu, sigma)
            for mu, sigma in zip(team2_mus, team2_sigmas, strict=False)
        ]
        prediction = pl_win_probability(model, team1_ratings, team2_ratings)[0]
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def trueskill_prediction(
        team1_mus: float | list[float],
        team1_sigmas: float | list[float],
        team2_mus: float | list[float],
        team2_sigmas: float | list[float],
    ) -> float:
        """
        Predicts the win probability based on TrueSkill ratings.

        Args:
            team1_mus (Union[float, List[float]]): TrueSkill mu values for Team 1.
            team1_sigmas (Union[float, List[float]]): TrueSkill sigma values for Team 1.
            team2_mus (Union[float, List[float]]): TrueSkill mu values for Team 2.
            team2_sigmas (Union[float, List[float]]): TrueSkill sigma values for Team 2.

        Returns:
            float: Predicted win probability for Team 1.

        """
        if not MatchPredictor.is_iterable(team1_mus):
            team1_mus = [team1_mus]
            team1_sigmas = [team1_sigmas]
            team2_mus = [team2_mus]
            team2_sigmas = [team2_sigmas]

        team1_ratings = [
            TrueskillRating(mu, sigma)
            for mu, sigma in zip(team1_mus, team1_sigmas, strict=False)
        ]
        team2_ratings = [
            TrueskillRating(mu, sigma)
            for mu, sigma in zip(team2_mus, team2_sigmas, strict=False)
        ]
        prediction = trueskill_win_probability(team1_ratings, team2_ratings)
        return round(prediction, RATING_DECIMALS)

    def whr_prediction(self, blue_team_id: int, red_team_id: int) -> float:
        """
        Predicts the win probability based on Whole History Rating (WHR).

        Args:
            blue_team_id (int): Team ID for the Blue team.
            red_team_id (int): Team ID for the Red team.

        Returns:
            float: Predicted win probability for the Blue team.

        """
        try:
            win_likelihood = self.outcome_prediction_model.probability_future_match(
                blue_team_id, red_team_id
            )
            return round(win_likelihood[0], RATING_DECIMALS)
        except AttributeError as e:
            msg = "WHR model is not loaded or not available."
            raise AttributeError(msg) from e
        except Exception as e:
            msg = f"Error during WHR prediction: {e}"
            raise RuntimeError(msg) from e

    def league_elo_prediction(self, team1_id: float, team2_id: float) -> float:
        """
        Predicts the win probability based on league ELO ratings.

        Args:
            team1_id (float): Team ID for Team 1.
            team2_id (float): Team ID for Team 2.

        Returns:
            float: Predicted win probability for Team 1.

        """
        team1_league_row = self.team_to_league[
            self.team_to_league["teamid"] == team1_id
        ]
        team2_league_row = self.team_to_league[
            self.team_to_league["teamid"] == team2_id
        ]

        if team1_league_row.empty or team2_league_row.empty:
            msg = "Team ID not found in team-to-league mapping."
            raise ValueError(msg)

        team1_league = team1_league_row["league"].values[0]
        team2_league = team2_league_row["league"].values[0]

        team1_league_elo_row = self.league_to_elo[
            self.league_to_elo["league"] == team1_league
        ]
        team2_league_elo_row = self.league_to_elo[
            self.league_to_elo["league"] == team2_league
        ]

        if team1_league_elo_row.empty or team2_league_elo_row.empty:
            msg = "League not found in ELO ratings."
            raise ValueError(msg)

        team1_league_elo = team1_league_elo_row["elo"].values[0]
        team2_league_elo = team2_league_elo_row["elo"].values[0]

        prediction = 1 / (
            1 + 10 ** ((team2_league_elo - team1_league_elo) / ELO_FACTOR)
        )
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def side_wr_prediction(team1_side_wr: float, team2_side_wr: float) -> float:
        """
        Predicts the win probability based on side win rates.

        Args:
            team1_side_wr (float): Team 1 side win rate.
            team2_side_wr (float): Team 2 side win rate.

        Returns:
            float: Predicted win probability for Team 1.

        """
        total_wr = team1_side_wr + team2_side_wr
        if total_wr == 0:
            return 0.5  # Default to 50% if no data is available
        prediction = team1_side_wr / total_wr
        return round(prediction, RATING_DECIMALS)

    @staticmethod
    def patch_season_wr_prediction(team1_wr: float, team2_wr: float) -> float:
        """
        Predicts the win probability based on patch or season win rates.

        Args:
            team1_wr (float): Team 1 win rate.
            team2_wr (float): Team 2 win rate.

        Returns:
            float: Predicted win probability for Team 1.

        """
        total_wr = team1_wr + team2_wr
        if total_wr == 0:
            return 0.5  # Default to 50% if no data is available
        prediction = team1_wr / total_wr
        return round(prediction, RATING_DECIMALS)

    def apply_stat_modifications(
        self,
        team1_stats: pd.Series,
        team2_stats: pd.Series,
        account_for_side: bool,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Applies statistical predictions to modify team statistics based on game side and other predictive metrics.

        Args:
            team1_stats (pd.Series): Statistics for Team 1.
            team2_stats (pd.Series): Statistics for Team 2.
            account_for_side (bool): Whether to account for side win rates.

        Returns:
            Tuple[pd.Series, pd.Series]: Modified statistics for Team 1 and Team 2.

        """
        team1_stats = team1_stats.copy()
        team2_stats = team2_stats.copy()

        # Apply different prediction models
        team1_stats["elo_win_likelihood"] = self.elo_prediction(
            team1_stats["elo"], team2_stats["elo"]
        )
        team1_stats["gl2_win_likelihood"] = self.gl2_prediction(
            team1_stats["gl2_mu"],
            team1_stats["gl2_phi"],
            team2_stats["gl2_mu"],
            team2_stats["gl2_phi"],
        )
        team1_stats["pl_win_likelihood"] = self.pl_prediction(
            team1_stats["pl_mu"],
            team1_stats["pl_sigma"],
            team2_stats["pl_mu"],
            team2_stats["pl_sigma"],
        )
        team1_stats["trueskill_win_likelihood"] = self.trueskill_prediction(
            team1_stats["trueskill_mu"],
            team1_stats["trueskill_sigma"],
            team2_stats["trueskill_mu"],
            team2_stats["trueskill_sigma"],
        )
        team1_stats["league_elo_win_likelihood"] = self.league_elo_prediction(
            team1_stats["teamid"], team2_stats["teamid"]
        )
        if account_for_side:
            side_key_team1 = f"ema_{team1_stats['side'].lower()}_side"
            side_key_team2 = f"ema_{team2_stats['side'].lower()}_side"
            team1_side_wr = team1_stats.get(side_key_team1, 0.0)
            team2_side_wr = team2_stats.get(side_key_team2, 0.0)
            team1_stats["side_win_likelihood"] = self.side_wr_prediction(
                team1_side_wr, team2_side_wr
            )
        else:
            team1_stats["side_win_likelihood"] = 0.5
        team1_stats["patch_win_likelihood"] = self.patch_season_wr_prediction(
            team1_stats["ema_patch_win_rate"], team2_stats["ema_patch_win_rate"]
        )
        team1_stats["season_win_likelihood"] = self.patch_season_wr_prediction(
            team1_stats["ema_season_win_rate"], team2_stats["ema_season_win_rate"]
        )

        # Drop unnecessary columns
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
        team1_stats = self.drop_unnecessary_columns(team1_stats, drop_columns)
        team2_stats = self.drop_unnecessary_columns(
            team2_stats, [*drop_columns, "teamname", "gameid"]
        )

        return team1_stats, team2_stats

    @staticmethod
    def drop_unnecessary_columns(stats: pd.Series, columns_to_drop: list) -> pd.Series:
        """
        Drops unnecessary columns from the statistics.

        Args:
            stats (pd.Series): The statistics to clean.
            columns_to_drop (list): List of columns to drop.

        Returns:
            pd.Series: Cleaned statistics.

        """
        return stats.drop(labels=columns_to_drop, errors="ignore")

    def calculate_team_stats(
        self, team1: Team, team2: Team, account_for_side: bool
    ) -> pd.DataFrame:
        """
        Combines and modifies team stats for further processing and analysis.

        Args:
            team1 (Team): Team 1 data.
            team2 (Team): Team 2 data.
            account_for_side (bool): Whether to account for side win rates.

        Returns:
            pd.DataFrame: Combined team statistics.

        """
        team1_stats, team2_stats = self.apply_stat_modifications(
            team1.team_stats, team2.team_stats, account_for_side
        )
        team2_stats = team2_stats.add_prefix("opp_")
        return pd.concat([team1_stats.to_frame().T, team2_stats.to_frame().T], axis=1)

    def apply_player_stat_modifications(
        self, player1_stats: pd.DataFrame, player2_stats: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies statistical predictions to modify player statistics based on game side and other predictive metrics.

        Args:
            player1_stats (pd.DataFrame): Player 1 statistics.
            player2_stats (pd.DataFrame): Player 2 statistics.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Modified player statistics for Player 1 and Player 2.

        """
        player1_stats = player1_stats.copy()
        player2_stats = player2_stats.copy()

        # Apply different prediction models
        player1_stats["elo_win_likelihood"] = self.elo_prediction(
            player1_stats["elo"], player2_stats["elo"]
        )
        player1_stats["gl2_win_likelihood"] = self.gl2_prediction(
            player1_stats["gl2_mu"],
            player1_stats["gl2_phi"],
            player2_stats["gl2_mu"],
            player2_stats["gl2_phi"],
        )
        player1_stats["pl_win_likelihood"] = self.pl_prediction(
            player1_stats["pl_mu"],
            player1_stats["pl_sigma"],
            player2_stats["pl_mu"],
            player2_stats["pl_sigma"],
        )
        player1_stats["trueskill_win_likelihood"] = self.trueskill_prediction(
            player1_stats["trueskill_mu"],
            player1_stats["trueskill_sigma"],
            player2_stats["trueskill_mu"],
            player2_stats["trueskill_sigma"],
        )

        # Drop unnecessary columns
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
        player1_stats = player1_stats.drop(columns=player_drop_columns, errors="ignore")
        player2_stats = player2_stats.drop(
            columns=[*player_drop_columns, "gameid", "teamname"], errors="ignore"
        )

        return player1_stats, player2_stats

    def calculate_player_stats(self, team1: Team, team2: Team) -> pd.DataFrame:
        """
        Handles player statistics similar to team statistics with modifications based on game side.

        Args:
            team1 (Team): Team 1 data.
            team2 (Team): Team 2 data.

        Returns:
            pd.DataFrame: Combined player statistics.

        """
        player1_stats, player2_stats = self.apply_player_stat_modifications(
            team1.player_stats, team2.player_stats
        )
        player2_stats = player2_stats.rename(
            columns=lambda x: f"opp_{x}" if x != "position" else x
        )
        return pd.merge(
            player1_stats,
            player2_stats,
            on="position",
            how="inner",
            validate="many_to_many",
        )

    @staticmethod
    def pivot_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots player data to prepare for merging with team data by grouping numeric and non-numeric columns.

        Args:
            player_data (pd.DataFrame): Raw player data.

        Returns:
            pd.DataFrame: Pivoted player data.

        """
        numeric_cols = player_data.select_dtypes(include=["number"]).columns
        non_numeric_cols = player_data.columns.difference(numeric_cols)

        agg_funcs = {col: "mean" for col in numeric_cols}
        agg_funcs.update(
            {
                col: "first"
                for col in non_numeric_cols
                if col not in ["position", "side"]
            }
        )

        player_data_pivoted = player_data.pivot_table(
            index=["gameid", "teamname"],
            columns="position",
            aggfunc=agg_funcs,
            fill_value=0,
        )
        player_data_pivoted.columns = [
            f"{pos}_{field}" for pos, field in player_data_pivoted.columns
        ]
        return player_data_pivoted.reset_index()

    @staticmethod
    def merge_datasets(
        team_data: pd.DataFrame, player_data_pivoted: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges team data with pivoted player data by game ID and team name, removing redundant columns.

        Args:
            team_data (pd.DataFrame): Team data.
            player_data_pivoted (pd.DataFrame): Pivoted player data.

        Returns:
            pd.DataFrame: Merged dataset.

        """
        prediction_data = pd.merge(
            team_data,
            player_data_pivoted,
            on=["gameid", "teamname"],
            how="inner",
            validate="many_to_many",
        )

        columns_to_drop = (
            ["gameid", "teamname"]
            + [f"{pos}_gameid" for pos in ["top", "jng", "mid", "bot", "sup"]]
            + [f"{pos}_teamname" for pos in ["top", "jng", "mid", "bot", "sup"]]
        )

        return prediction_data.drop(columns=columns_to_drop, errors="ignore")

    def preprocess_data(
        self, team_data: pd.DataFrame, player_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes team and player data by pivoting player data and merging it with team data.

        Args:
            team_data (pd.DataFrame): Team data.
            player_data (pd.DataFrame): Player data.

        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.

        """
        player_data_pivoted = self.pivot_player_data(player_data)
        return self.merge_datasets(team_data, player_data_pivoted)

    def keep_necessary_columns(self, final_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps only the necessary columns for the prediction model.

        Args:
            final_stats (pd.DataFrame): DataFrame containing all features.

        Returns:
            pd.DataFrame: DataFrame with only the required features.

        """
        try:
            with open(OUTCOME_PREDICTION_FINAL_FEATURES, "rb") as f:
                final_features = pickle.load(f)
            final_stats = final_stats.reindex(columns=final_features)
            return self.convert_data_types(final_stats)
        except FileNotFoundError as e:
            msg = f"Feature file not found: {e}"
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Error keeping necessary columns: {e}"
            raise RuntimeError(msg) from e

    def convert_data_types(self, final_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Converts columns to numeric where possible, otherwise converts to category type.

        Args:
            final_stats (pd.DataFrame): DataFrame to convert.

        Returns:
            pd.DataFrame: DataFrame with converted data types.

        """
        try:
            with open(OUTCOME_PREDICTION_CATEGORICAL_FEATURES, "rb") as f:
                categorical_features = pickle.load(f)

            final_stats = final_stats.copy()
            for col in final_stats.columns:
                if col in categorical_features:
                    final_stats[col] = final_stats[col].astype("category")
                else:
                    final_stats[col] = pd.to_numeric(final_stats[col], errors="coerce")

            return final_stats
        except FileNotFoundError as e:
            msg = f"Categorical features file not found: {e}"
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Error converting data types: {e}"
            raise RuntimeError(msg) from e

    def predict_outcomes(self, final_stats: pd.DataFrame) -> np.ndarray:
        """
        Uses the predictive model to estimate outcomes based on preprocessed final stats.

        Args:
            final_stats (pd.DataFrame): Preprocessed data.

        Returns:
            np.ndarray: Prediction probabilities.

        """
        try:
            final_stats = GradientBoostingModel.process_players_likelihood_columns(
                final_stats
            )
            final_stats = GradientBoostingModel.fuse_opposing_team_features(final_stats)
            final_stats = self.keep_necessary_columns(final_stats)
            return round(
                self.outcome_prediction_model.predict_proba(final_stats),
                PREDICTION_PRECISION,
            )
        except AttributeError as e:
            msg = f"Prediction model not properly loaded: {e}"
            raise AttributeError(msg) from e
        except Exception as e:
            msg = f"Error during prediction: {e}"
            raise RuntimeError(msg) from e

    def calculate_team_and_player_stats(
        self, team1: Team, team2: Team, account_for_side: bool
    ) -> pd.DataFrame:
        """
        Calculates combined team and player statistics for prediction.

        Args:
            team1 (Team): Team 1 data.
            team2 (Team): Team 2 data.
            account_for_side (bool): Whether to account for side win rates.

        Returns:
            pd.DataFrame: Combined statistics ready for preprocessing.

        """
        team_stats = self.calculate_team_stats(team1, team2, account_for_side)
        player_stats = self.calculate_player_stats(team1, team2)
        return self.preprocess_data(team_stats, player_stats)

    def predict_match(
        self, team1: Team, team2: Team, account_for_side: bool = True
    ) -> dict[str, float]:
        """
        Predicts the outcome of a match between two teams.

        Args:
            team1 (Team): Team 1 data.
            team2 (Team): Team 2 data.
            account_for_side (bool, optional): Whether to account for side win rates. Defaults to True.

        Returns:
            Dict[str, float]: Prediction probabilities for Team 1 and Team 2.

        """
        try:
            final_stats = self.calculate_team_and_player_stats(
                team1, team2, account_for_side
            )
            predictions = self.predict_outcomes(final_stats)
            return {
                "team1_win_probability": float(predictions[0][1]),
                "team2_win_probability": float(predictions[0][0]),
            }
        except Exception as e:
            msg = f"Error predicting match outcome: {e}"
            raise RuntimeError(msg) from e
