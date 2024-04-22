"""
Model Validators

This module contains classes that validate the performance of the models used in the project.
The classes are designed to calculate the accuracy, log loss, and Brier score of the models.

"""

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss

from utils.logger import logger, models_logger
from utils.paths import CONSIDERED_LEAGUES, FIGURES_DIR, METRICS_DIR, MIXED_VALIDATOR_WEIGHTS, PROCESSED_DIR
from utils.utils import json_loader

sns.set_style("darkgrid")

# Silence seaborn warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelValidator(ABC):
    teams: pd.DataFrame
    players: pd.DataFrame
    directory: Path = field(default=FIGURES_DIR)
    graph: bool = field(default=True)
    metrics: dict = field(default_factory=lambda: {"accuracy": 0.0, "logloss": 0.0, "brier": 0.0})

    @abstractmethod
    def preprocess_data(self):
        pass

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        predictions: str,
        actuals: str,
        likelihood: str = None,
        is_trueskill: bool = False,
    ):
        """
        Calculate the accuracy, log loss, and Brier score of the model using the given data.
        """
        if likelihood is None:
            likelihood = predictions

        likelihoods = df[likelihood].fillna(0.5)
        if is_trueskill:
            likelihoods = likelihoods + 0.5

        df["correct"] = (df[predictions] == df[actuals]).astype(int)
        df["tie"] = (df[predictions] == 0.5) * 0.5

        accuracy = df[["correct", "tie"]].sum(axis=1).mean()
        logloss = log_loss(df[actuals], likelihoods)
        brier = brier_score_loss(df[actuals], likelihoods)
        self.metrics.update({"accuracy": accuracy, "logloss": logloss, "brier": brier})

    def generate_graph(self, df: pd.DataFrame, x: str, y: str, hue: str, title: str, filename: str):
        """
        Generate a jointplot with the given data and save it to a file.
        """
        if not self.graph:
            return

        # Format the text for annotations
        metrics_text = (
            f"Accuracy: {self.metrics['accuracy']:.4f}\n"
            f"Log Loss: {self.metrics['logloss']:.4f}\n"
            f"Brier: {self.metrics['brier']:.4f}"
        )

        grf = sns.jointplot(data=df, x=x, y=y, hue=hue)
        grf.ax_joint.text(
            (df[x].mean() - (df[x].mean() * 0.125)),
            (df[y].max() - (df[y].max() * 0.01)),
            metrics_text,
            bbox=dict(facecolor="grey", edgecolor="black", boxstyle="round"),
        )
        grf.set_axis_labels(x, y)
        plt.title(title, loc="right", y=1.1)
        grf.savefig(self.directory.joinpath(filename), dpi=300, format="png")
        plt.clf()
        plt.close()

    def plot_historical_accuracy(self, model_name, df):
        """
        Plot the historical accuracy of a model over time, with a trend line.

        Args:
            model_name (str): The name of the model.
            df (pd.DataFrame): DataFrame containing 'date' and 'correct' columns.
        """

        df = df.copy()

        # Ensure 'date' is in datetime format and sort
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.sort_values(by="date", inplace=True)

        # Compute day by day accuracy using vectorized operations
        df_grouped = df.groupby(df["date"])["correct"].mean().reset_index(name="accuracy")
        df_grouped["date"] = pd.to_datetime(df_grouped["date"])

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_grouped,
            x="date",
            y="accuracy",
            marker="o",
            linestyle="-",
            ax=ax,
            label="Daily Accuracy",
        )

        # Plot a trend line
        z = np.polyfit(mdates.date2num(df_grouped["date"]), df_grouped["accuracy"], 1)
        p = np.poly1d(z)
        plt.plot(
            df_grouped["date"],
            p(mdates.date2num(df_grouped["date"])),
            "r--",
            label="Trend Line",
        )

        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Historical Accuracy of {model_name}")
        ax.grid(True)
        plt.xticks(df_grouped["date"][::14], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Saving the plot
        plt.savefig(self.directory.joinpath(f"{model_name}_Historical_Accuracy.png"), dpi=300)
        plt.close()

    @abstractmethod
    def validate(self):
        pass

    def log_metrics(self, model_name: str):
        """Log the calculated metrics for the model."""
        metrics_format = (
            f"{model_name}\t"
            f"Accuracy: {self.metrics['accuracy']:.4f}, "
            f"Log Loss: {self.metrics['logloss']:.4f}, "
            f"Brier: {self.metrics['brier']:.4f}"
        )

        # Logging the formatted metrics
        models_logger.info(metrics_format)

        # Store metrics in a json file
        metrics_file = METRICS_DIR / f"{model_name.strip()}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)


@dataclass
class TeamEloValidator(ModelValidator):
    def preprocess_data(self):
        conditions = [
            self.teams["elo_win_likelihood"] > 0.5,
            self.teams["elo_win_likelihood"] < 0.5,
        ]
        choices = [1, 0]
        self.teams["team_elo_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["result"] = self.teams["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(self.teams, "team_elo_expected_result", "result", "elo_win_likelihood")
        self.generate_graph(
            df=self.teams,
            x="elo_pre_match",
            y="elo_pre_match_opponent",
            hue="result",
            title="Team Elo Validation",
            filename="Team_Elo_Validation.png",
        )
        self.log_metrics(model_name="Team_Elo")
        self.plot_historical_accuracy("Team_Elo", self.teams)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class PlayerEloValidator(ModelValidator):
    def preprocess_data(self):
        elo_win_likelihood_mean = self.players.groupby(["gameid", "side"])["elo_win_likelihood"].transform("sum") / 5
        self.players["elo_win_likelihood_mean"] = elo_win_likelihood_mean
        conditions = [
            self.players["elo_win_likelihood_mean"] > 0.5,
            self.players["elo_win_likelihood_mean"] < 0.5,
        ]
        choices = [1, 0]
        self.players["player_elo_expected_result"] = np.select(conditions, choices, default=0.5)
        self.players["result"] = self.players["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(
            self.players,
            "player_elo_expected_result",
            "result",
            "elo_win_likelihood_mean",
        )
        self.generate_graph(
            df=self.players,
            x="elo_pre_match",
            y="elo_pre_match_opponent",
            hue="result",
            title="Player Elo Validation",
            filename="Player_Elo_Validation.png",
        )
        self.log_metrics(model_name="Player_Elo")
        self.plot_historical_accuracy("Player_Elo", self.players)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class TrueSkillValidator(ModelValidator):
    def preprocess_data(self):
        conditions = [
            self.teams["trueskill_diff"] > 0,
            self.teams["trueskill_diff"] < 0,
        ]
        choices = [1, 0]
        self.teams["trueskill_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["result"] = self.teams["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(
            self.teams,
            "trueskill_expected_result",
            "result",
            "trueskill_diff",
            is_trueskill=True,
        )
        self.generate_graph(
            df=self.teams,
            x="trueskill_sum_mu",
            y="trueskill_opponent_sum_mu",
            hue="result",
            title="TrueSkill Validation",
            filename="TrueSkill_Validation.png",
        )
        self.log_metrics(model_name="TrueSkill")
        self.plot_historical_accuracy("TrueSkill", self.teams)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class TeamPlackettLuceValidator(ModelValidator):
    def preprocess_data(self):
        conditions = [
            self.teams["pl_win_likelihood"] > 0.5,
            self.teams["pl_win_likelihood"] < 0.5,
        ]
        choices = [1, 0]
        self.teams["team_pl_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["result"] = self.teams["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(self.teams, "team_pl_expected_result", "result", "pl_win_likelihood")
        self.generate_graph(
            df=self.teams,
            x="pl_pre_match_mu",
            y="pl_pre_match_mu_opponent",
            hue="result",
            title="Team Plackett-Luce Validation",
            filename="Team_PlackettLuce_Validation.png",
        )
        self.log_metrics(model_name="Team_PL ")
        self.plot_historical_accuracy("Team_PlackettLuce", self.teams)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class PlayerPlackettLuceValidator(ModelValidator):
    def preprocess_data(self):
        pl_win_likelihood_mean = self.players.groupby(["gameid", "side"])["pl_win_likelihood"].transform("sum") / 5
        self.players["pl_win_likelihood_mean"] = pl_win_likelihood_mean
        conditions = [
            self.players["pl_win_likelihood_mean"] > 0.5,
            self.players["pl_win_likelihood_mean"] < 0.5,
        ]
        choices = [1, 0]
        self.players["player_pl_expected_result"] = np.select(conditions, choices, default=0.5)
        self.players["result"] = self.players["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(
            self.players,
            "player_pl_expected_result",
            "result",
            "pl_win_likelihood_mean",
        )
        self.generate_graph(
            df=self.players,
            x="pl_pre_match_mu",
            y="pl_pre_match_mu_opponent",
            hue="result",
            title="Player Plackett-Luce Validation",
            filename="Player_PlackettLuce_Validation.png",
        )
        self.log_metrics(model_name="Player_PL")
        self.plot_historical_accuracy("Player_PlackettLuce", self.players)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class TeamEgpmDominanceValidator(ModelValidator):
    def preprocess_data(self):
        conditions = [
            self.teams["egpm_dominance_log_win_perc"] > 0.5,
            self.teams["egpm_dominance_log_win_perc"] < 0.5,
        ]
        choices = [1, 0]
        self.teams["team_egpm_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["result"] = self.teams["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(
            self.teams,
            "team_egpm_expected_result",
            "result",
            "egpm_dominance_log_win_perc",
        )
        self.generate_graph(
            df=self.teams,
            x="egpm_dominance_ratio_ema_after",
            y="egpm_opp_dominance_ratio_ema_after",
            hue="result",
            title="Team EGPM Validation",
            filename="Team_EGPM_Validation.png",
        )
        self.log_metrics(model_name="Team_EGPM")
        self.plot_historical_accuracy("Team_EGPM", self.teams)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class PlayerEgpmDominanceValidator(ModelValidator):
    def preprocess_data(self):
        egpm_dominance_diff_mean = (
            self.players.groupby(["gameid", "side"])["egpm_dominance_log_win_perc"].transform("sum") / 5
        )
        self.players["egpm_dominance_log_win_perc_mean"] = egpm_dominance_diff_mean
        conditions = [
            self.players["egpm_dominance_log_win_perc_mean"] > 0.5,
            self.players["egpm_dominance_log_win_perc_mean"] < 0.5,
        ]
        choices = [1, 0]
        self.players["player_egpm_expected_result"] = np.select(conditions, choices, default=0.5)
        self.players["result"] = self.players["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(
            self.players,
            "player_egpm_expected_result",
            "result",
            "egpm_dominance_log_win_perc_mean",
        )
        self.generate_graph(
            df=self.players,
            x="egpm_dominance_ratio_ema_after",
            y="egpm_opp_dominance_ratio_ema_after",
            hue="result",
            title="Player EGPM Validation",
            filename="Player_EGPM_Validation.png",
        )
        self.log_metrics(model_name="Player_EGPM")
        self.plot_historical_accuracy("Player_EGPM", self.players)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class TeamSideEmaValidator(ModelValidator):
    def preprocess_data(self):
        conditions = [
            self.teams["ema_side_win_perc"] > 0.5,
            self.teams["ema_side_win_perc"] < 0.5,
        ]
        choices = [1, 0]
        self.teams["team_egpm_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["result"] = self.teams["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(self.teams, "team_egpm_expected_result", "result", "ema_side_win_perc")
        self.generate_graph(
            df=self.teams,
            x="ema_blue_side_before",
            y="ema_opp_blue_side_before",
            hue="result",
            title="Team Side EMA Validation",
            filename="Team_Side_EMA_Validation.png",
        )
        self.log_metrics(model_name="Team_Side_EMA")
        self.plot_historical_accuracy("Team_Side_EMA", self.teams)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class PlayerSideEmaValidator(ModelValidator):
    def preprocess_data(self):
        side_ema_mean = self.players.groupby(["gameid", "side"])["ema_side_win_perc"].transform("sum") / 5
        self.players["side_ema_mean"] = side_ema_mean
        conditions = [
            self.players["side_ema_mean"] > 0.5,
            self.players["side_ema_mean"] < 0.5,
        ]
        choices = [1, 0]
        self.players["player_side_ema_expected_result"] = np.select(conditions, choices, default=0.5)
        self.players["result"] = self.players["result"].astype("int32")

    def validate(self):
        self.preprocess_data()
        self.calculate_metrics(self.players, "player_side_ema_expected_result", "result", "side_ema_mean")
        self.generate_graph(
            df=self.players,
            x="ema_blue_side_before",
            y="ema_opp_blue_side_before",
            hue="result",
            title="Player Side EMA Validation",
            filename="Player_Side_EMA_Validation.png",
        )
        self.log_metrics(model_name="Player_Side_EMA")
        self.plot_historical_accuracy("Player_Side_EMA", self.players)
        return self.metrics["accuracy"], self.metrics["logloss"], self.metrics["brier"]


@dataclass
class TeamEnsembleValidator(ModelValidator):
    def __post_init__(self):
        self.preprocess_data()

    def preprocess_data(self):
        self.validators = [
            TeamEloValidator(self.teams, self.players, self.directory, self.graph).validate(),
            TrueSkillValidator(self.teams, self.players, self.directory, self.graph).validate(),
            TeamPlackettLuceValidator(self.teams, self.players, self.directory, self.graph).validate(),
            TeamEgpmDominanceValidator(self.teams, self.players, self.directory, self.graph).validate(),
            TeamSideEmaValidator(self.teams, self.players, self.directory, self.graph).validate(),
        ]
        accuracies = [validator[0] for validator in self.validators]
        total_accuracy = sum(accuracies)
        weights = [accuracy / total_accuracy for accuracy in accuracies]
        self.teams["ensemble_win_perc"] = (
            (self.teams["elo_win_likelihood"] * weights[0])
            + ((self.teams["trueskill_diff"] + 0.5) * weights[1])
            + (self.teams["pl_win_likelihood"] * weights[2])
            + (self.teams["egpm_dominance_log_win_perc"] * weights[3])
            + (self.teams["ema_side_win_perc"] * weights[4])
        )
        self.teams["ensemble_opp_win_perc"] = 1 - self.teams["ensemble_win_perc"]
        conditions = [
            self.teams["ensemble_win_perc"] > 0.5,
            self.teams["ensemble_win_perc"] < 0.5,
        ]
        choices = [1, 0]
        self.teams["ensemble_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams["majority_voting"] = self.teams[
            [
                "team_elo_expected_result",
                "trueskill_expected_result",
                "team_pl_expected_result",
                "team_egpm_expected_result",
                "team_egpm_expected_result",
            ]
        ].mode(axis=1)[0]

    def validate(self):
        self.calculate_metrics(self.teams, "ensemble_expected_result", "result", "ensemble_win_perc")
        self.generate_graph(
            df=self.teams,
            x="ensemble_win_perc",
            y="ensemble_opp_win_perc",
            hue="result",
            title="Team Ensemble Validation",
            filename="Team_Ensemble_Validation.png",
        )
        self.log_metrics(model_name="Team_Ensemble")
        self.plot_historical_accuracy("Team_Ensemble", self.teams)
        ensemble_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]
        self.calculate_metrics(self.teams, "majority_voting", "result", "ensemble_win_perc")
        self.generate_graph(
            df=self.teams,
            x="elo_pre_match",
            y="elo_pre_match_opponent",
            hue="result",
            title="Team Majority Voting Validation",
            filename="Team_Majority_Voting_Validation.png",
        )
        self.log_metrics(model_name="Team MVoting")
        self.plot_historical_accuracy("Team_Majority_Voting", self.teams)
        majority_voting_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]
        return ensemble_metrics, majority_voting_metrics


@dataclass
class PlayerEnsembleValidator(ModelValidator):
    def __post_init__(self):
        self.preprocess_data()

    def preprocess_data(self):
        validators = [
            PlayerEloValidator(self.teams, self.players, self.directory, self.graph).validate(),
            PlayerPlackettLuceValidator(self.teams, self.players, self.directory, self.graph).validate(),
            PlayerEgpmDominanceValidator(self.teams, self.players, self.directory, self.graph).validate(),
            PlayerSideEmaValidator(self.teams, self.players, self.directory, self.graph).validate(),
        ]
        accuracies = [validator[0] for validator in validators]
        total_accuracy = sum(accuracies)
        weights = [accuracy / total_accuracy for accuracy in accuracies]
        self.players["ensemble_win_perc"] = (
            (self.players["elo_win_likelihood"] * weights[0])
            + (self.players["pl_win_likelihood"] * weights[1])
            + (self.players["egpm_dominance_log_win_perc"] * weights[2])
            + (self.players["ema_side_win_perc"] * weights[3])
        )
        self.players["ensemble_opp_win_perc"] = 1 - self.players["ensemble_win_perc"]
        conditions = [
            self.players["ensemble_win_perc"] > 0.5,
            self.players["ensemble_win_perc"] < 0.5,
        ]
        choices = [1, 0]
        self.players["ensemble_expected_result"] = np.select(conditions, choices, default=0.5)
        self.players["majority_voting"] = self.players[
            [
                "player_elo_expected_result",
                "player_pl_expected_result",
                "player_egpm_expected_result",
                "player_side_ema_expected_result",
            ]
        ].mode(axis=1)[0]

    def validate(self):
        self.calculate_metrics(self.players, "ensemble_expected_result", "result", "ensemble_win_perc")
        self.generate_graph(
            df=self.players,
            x="ensemble_win_perc",
            y="ensemble_opp_win_perc",
            hue="result",
            title="Player Ensemble Validation",
            filename="Player_Ensemble_Validation.png",
        )
        self.log_metrics(model_name="Player_Ensemble")
        self.plot_historical_accuracy("Player_Ensemble", self.players)
        ensemble_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]
        self.calculate_metrics(self.players, "majority_voting", "result")
        self.generate_graph(
            df=self.players,
            x="elo_pre_match",
            y="elo_pre_match_opponent",
            hue="result",
            title="Player Majority Voting Validation",
            filename="Player_Majority_Voting_Validation.png",
        )
        self.log_metrics(model_name="Player_MVoting")
        self.plot_historical_accuracy("Player_Majority_Voting", self.players)
        majority_voting_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]
        return ensemble_metrics, majority_voting_metrics


@dataclass
class MixedValidator(ModelValidator):
    """
    A class to validate the Mixed Ensemble and Majority Voting models.
    They combine the predictions of the individual models to make a final prediction.
    """

    def __post_init__(self):
        self.preprocess_data()

    def preprocess_data(self):
        grouped_players = self.players.groupby(["gameid", "side"])
        player_groups = pd.DataFrame(
            {
                "gameid": [k[0] for k in grouped_players.groups.keys()],
                "side": [k[1] for k in grouped_players.groups.keys()],
                "elo_win_likelihood_mean": grouped_players["elo_win_likelihood"].mean(),
                "pl_win_likelihood_mean": grouped_players["pl_win_likelihood"].mean(),
                "egpm_dominance_log_win_perc_mean": grouped_players["egpm_dominance_log_win_perc"].mean(),
                "side_ema_mean": grouped_players["ema_side_win_perc"].mean(),
            },
            index=grouped_players.groups.keys(),
        )
        merged_df = pd.merge(self.teams, player_groups, on=["gameid", "side"], how="left")
        merged_df = merged_df.fillna(self.teams)
        self.validators = [
            TeamEloValidator(merged_df, self.players, self.directory, self.graph),
            PlayerEloValidator(merged_df, self.players, self.directory, self.graph),
            TrueSkillValidator(merged_df, self.players, self.directory, self.graph),
            TeamPlackettLuceValidator(merged_df, self.players, self.directory, self.graph),
            PlayerPlackettLuceValidator(merged_df, self.players, self.directory, self.graph),
            TeamEgpmDominanceValidator(merged_df, self.players, self.directory, self.graph),
            PlayerEgpmDominanceValidator(merged_df, self.players, self.directory, self.graph),
            TeamSideEmaValidator(merged_df, self.players, self.directory, self.graph),
            PlayerSideEmaValidator(merged_df, self.players, self.directory, self.graph),
        ]
        self.calculate_ensemble(merged_df)
        self.calculate_majority_voting(merged_df)

    def calculate_ensemble(self, df):
        accuracies = [validator.validate()[0] for validator in self.validators]
        total_accuracy = sum(accuracies)
        weights = [accuracy / total_accuracy for accuracy in accuracies]
        df["ensemble_win_perc"] = (
            (df["elo_win_likelihood"] * weights[0])
            + (df["elo_win_likelihood_mean"] * weights[1])
            + ((df["trueskill_diff"] + 0.5) * weights[2])
            + (df["pl_win_likelihood"] * weights[3])
            + (df["pl_win_likelihood_mean"] * weights[4])
            + (df["egpm_dominance_log_win_perc"] * weights[5])
            + (df["egpm_dominance_log_win_perc_mean"] * weights[6])
            + (df["ema_side_win_perc"] * weights[7])
            + (df["side_ema_mean"] * weights[8])
        )
        df["ensemble_opp_win_perc"] = 1 - df["ensemble_win_perc"]
        conditions = [
            df["ensemble_win_perc"] > 0.5,
            df["ensemble_win_perc"] < 0.5,
        ]
        choices = [1, 0]
        df["ensemble_expected_result"] = np.select(conditions, choices, default=0.5)
        self.teams = df

        self.weights = {
            "team_elo": weights[0],
            "player_elo": weights[1],
            "trueskill": weights[2],
            "team_pl": weights[3],
            "player_pl": weights[4],
            "team_egpm_dom": weights[5],
            "player_egpm_dom": weights[6],
            "team_side_win": weights[7],
            "player_side_win": weights[8],
        }

        # Store the weights in a pickle file
        with open(MIXED_VALIDATOR_WEIGHTS, "wb") as f:
            pickle.dump(self.weights, f)

    def calculate_majority_voting(self, df):
        player_aggregates = (
            self.players.groupby(["gameid", "side"])
            .agg(
                {
                    "player_elo_expected_result": "mean",
                    "player_pl_expected_result": "mean",
                    "player_egpm_expected_result": "mean",
                    "player_side_ema_expected_result": "mean",
                }
            )
            .reset_index()
        )
        df = pd.merge(df, player_aggregates, on=["gameid", "side"], how="left")
        df["majority_voting"] = df[
            [
                "team_elo_expected_result",
                "player_elo_expected_result",
                "trueskill_expected_result",
                "team_pl_expected_result",
                "player_pl_expected_result",
                "team_egpm_expected_result",
                "player_egpm_expected_result",
                "team_egpm_expected_result",
                "player_side_ema_expected_result",
            ]
        ].mode(axis=1)[0]
        self.teams = df

    def validate(self):
        self.calculate_metrics(self.teams, "ensemble_expected_result", "result", "ensemble_win_perc")
        self.generate_graph(
            df=self.teams,
            x="ensemble_win_perc",
            y="ensemble_opp_win_perc",
            hue="result",
            title="Mixed Ensemble Validation",
            filename="Mixed_Ensemble_Validation.png",
        )
        self.log_metrics(model_name="Mixed_Ensemble")
        self.plot_historical_accuracy("Mixed_Ensemble", self.teams)
        ensemble_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]

        self.calculate_metrics(self.teams, "majority_voting", "result")
        self.generate_graph(
            df=self.teams,
            x="elo_pre_match",
            y="elo_pre_match_opponent",
            hue="result",
            title="Mixed Majority Voting Validation",
            filename="Mixed_Majority_Voting_Validation.png",
        )
        self.log_metrics(model_name="Mixed_MVoting")
        self.plot_historical_accuracy("Mixed_Majority_Voting", self.teams)
        models_logger.info("\n")
        majority_voting_metrics = [
            self.metrics["accuracy"],
            self.metrics["logloss"],
            self.metrics["brier"],
        ]
        return ensemble_metrics, majority_voting_metrics

    def get_weights(self):
        return self.weights


if __name__ == "__main__":
    team_data = pd.read_csv(PROCESSED_DIR / "team_data.csv")
    player_data = pd.read_csv(PROCESSED_DIR / "player_data.csv")
    if considered_leagues := json_loader(CONSIDERED_LEAGUES)["considered_leagues"]:
        logger.info(f"Considering only a subset of {len(considered_leagues)} leagues.")
        team_data = team_data[team_data["league"].isin(considered_leagues)]
        player_data = player_data[player_data["league"].isin(considered_leagues)]

    logger.info("Validating Ensemble models...")
    team_ensemble_validator = TeamEnsembleValidator(
        teams=team_data,
        players=player_data,
        directory=FIGURES_DIR,
        graph=True,
    ).validate()

    player_ensemble_validator = PlayerEnsembleValidator(
        teams=team_data,
        players=player_data,
        directory=FIGURES_DIR,
        graph=True,
    ).validate()

    logger.info("Validation complete.")
    print()

    logger.info("Validating Mixed Ensemble and Majority Voting models...")

    mixed_validator = MixedValidator(
        teams=team_data,
        players=player_data,
        directory=FIGURES_DIR,
        graph=True,
    ).validate()

    logger.info("Validation complete.")

# HALF LIFE TUNING
# 20 - 0.6483104817561999
# 19 - 0.6484603281636323
# 18 - 0.6484603281636323
# 17 - 0.648385404959916
# 16 - 0.648760020978497
# 15 - 0.6486850977747809
# 11 - 0.6486476361729228
# 10 - 0.6486101745710646
# 9  - 0.648385404959916
# 8  - 0.647935865737619
# 5  - 0.6474114033116056
