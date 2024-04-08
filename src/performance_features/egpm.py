"""
Earned Gold Per Minute Model

This scripts computes Earned Gold Per Minute (EGPM) stats for players and teams.
It uses TrueSkill ratings to adjust the EGPM stats based on the strength of the opponent.
I explored other methods to adjust EGPM stats, but TrueSkill was the most effective so far.
I also tried to include sigma in the dominance ratio, but it didn't improve the model.
A more refined formula could be used to adjust the dominance ratio, but it would be more complex.
"""

import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.paths import DEFAULT_PARAMETERS, MODELS_DIR
from utils.utils import json_loader

HALF_LIFE = json_loader(DEFAULT_PARAMETERS)["half_life"]


def calculate_egpm_dominance_ratios(data, mu, opp_mu):
    """
    Calculate dominance ratios using weighted mus.
    """

    # Compute dominance ratios using weighted mus
    data["egpm_dominance_ratio"] = data["egpm"] / (opp_mu / mu)
    data["egpm_opp_dominance_ratio"] = data["opponent_egpm"] / (mu / opp_mu)

    return data


def calculate_ema_features(data, identity, feature_name):
    """
    Calculate Exponential Moving Average (EMA) features.
    """
    # We fill nans with another forward fill and then backfill just to avoid losing the temporal progression
    ema_before = data.groupby([identity])[feature_name].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean().shift().bfill()
    )
    ema_after = data.groupby([identity])[feature_name].transform(
        lambda x: x.ewm(halflife=HALF_LIFE, ignore_na=True).mean()
    )

    ema_before = ema_before.ffill().bfill()
    ema_after = ema_after.ffill().bfill()

    ema_before.fillna(0, inplace=True)
    ema_after.fillna(0, inplace=True)

    return ema_before, ema_after


def calculate_dominance_metrics(data, identity):
    """
    Calculate dominance metrics using Exponential Moving Averages (EMA).
    """
    for feature in ["egpm_dominance_ratio", "egpm_opp_dominance_ratio"]:
        (
            data[f"{feature}_ema_before"],
            data[f"{feature}_ema_after"],
        ) = calculate_ema_features(data, identity, feature)

    data["egpm_dominance_diff"] = data["egpm_dominance_ratio_ema_before"] - data["egpm_opp_dominance_ratio_ema_before"]

    return data


def train_logistic_regression_model(data):
    features = [
        "egpm_dominance_ratio_ema_before",
        "egpm_opp_dominance_ratio_ema_before",
    ]
    X = data[features]
    y = data["result"]

    clf = LogisticRegression()
    clf.fit(X, y)

    # Store the win probabilities in the dataframe
    data["egpm_dominance_log_win_perc"] = clf.predict_proba(X)[:, 1]
    return clf, data


def save_model(clf, filename):
    filepath = MODELS_DIR / filename
    pickle.dump(clf, open(filepath, "wb"))


def egpm_model(data: pd.DataFrame, entity: str, store_model: bool = True) -> pd.DataFrame:
    """
    Calculate a model based on Earned GPM with TrueSkill factored in.

    Parameters
    ----------
    data : DataFrame
        Oracle's Elixir data as provided by the output of the Team TrueSkill function.
    entity : str
        Entity to compute on, either "Team" or "Player"

    Returns
    -------
    DataFrame with Earned Gold Per Minute vs Opponent Team Strength model.
    """
    if entity.lower() == "team":
        identity, mu, opp_mu = (
            "teamid",
            data["trueskill_sum_mu"],
            data["trueskill_opponent_sum_mu"],
        )
    elif entity.lower() == "player":
        identity, mu, opp_mu = (
            "playerid",
            data["trueskill_mu"],
            data["trueskill_opponent_mu"],
        )
    else:
        raise ValueError("Entity must be either 'player' or 'team'.")

    data = calculate_egpm_dominance_ratios(data, mu, opp_mu)
    data = calculate_dominance_metrics(data, identity)
    clf, data = train_logistic_regression_model(data)
    if store_model:
        save_model(clf, "egpm_dom_logistic_regression.pkl")

    return data
