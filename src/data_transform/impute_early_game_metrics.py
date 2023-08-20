import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

"""
Impute metrics for the 15m mark, `csat15`, `xpat15`, `goldat15`. 
By generating an ensemble model composed of L2 Regression, k-NN, and a decision tree.
It will try to use end game values to predict/impute values at 15m, which we need for
other models elsewhere.
"""

# Initialize Logger
date_format = "%m/%d/%Y %I:%M:%S %p"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt=date_format
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Feature Engineering
def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate required additional features for imputing the given data.

    Args:
    - data (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with additional features.
    """
    # Calculate teamkills before generating other features
    teamkills = data.groupby(["gameid", "teamid"])["kills"].transform("sum")

    # KDA ratio
    data["KDA"] = (data["kills"] + data["assists"]) / np.where(
        data["deaths"] == 0, 1, data["deaths"]
    )

    # Gold Efficiency
    data["gold_efficiency"] = np.where(
        data["gamelength"] == 0, 0, data["totalgold"] / data["gamelength"]
    )

    # XP Efficiency
    data["xp_efficiency"] = np.where(
        data["gamelength"] == 0, 0, data["total cs"] / data["gamelength"]
    )

    # Kill Participation
    data["kill_participation"] = np.where(
        teamkills == 0, 0, (data["kills"] + data["assists"]) / teamkills
    )

    # Sort and group the data once
    data = data.sort_values(by=["playerid", "date"], ascending=True)
    grouped_data = data.groupby("playerid")

    # Calculate volatility metrics
    data["kills_volatility"] = (
        grouped_data["kills"].rolling(window=5).std().reset_index(0, drop=True)
    )
    data["deaths_volatility"] = (
        grouped_data["deaths"].rolling(window=5).std().reset_index(0, drop=True)
    )

    # Calculate growth metrics
    data["kills_growth"] = (
        grouped_data["kills"].rolling(window=5).mean().reset_index(0, drop=True)
    )
    data["deaths_growth"] = (
        grouped_data["deaths"].rolling(window=5).mean().reset_index(0, drop=True)
    )

    return data


# Training Stacked Models for Imputation
def train_stacked_model(
    train_data: pd.DataFrame, val_data: pd.DataFrame, features: list, target: str
) -> tuple[dict[str, KNeighborsRegressor | Ridge | DecisionTreeRegressor], object]:
    """
    Train stacked models for the given target using the specified features.

    Args:
    - train_data (pd.DataFrame): Training data.
    - val_data (pd.DataFrame): Validation data.
    - features (list): List of feature columns.
    - target (str): Target column.

    Returns:
    - tuple: Tuple containing dictionary of base models and the meta-model.
    """
    knn = KNeighborsRegressor(n_neighbors=5)
    ridge = Ridge(alpha=1.0)
    tree = DecisionTreeRegressor(max_depth=5)
    base_models = {"KNN": knn, "Ridge": ridge, "Decision Tree": tree}

    for model in base_models.values():
        model.fit(train_data[features], train_data[target])
    val_predictions_base = {
        name: model.predict(val_data[features]) for name, model in base_models.items()
    }
    val_predictions_df = pd.DataFrame(val_predictions_base)
    meta_model = LinearRegression().fit(val_predictions_df, val_data[target])

    return base_models, meta_model


def impute_missing_values(
    data: pd.DataFrame,
    features: list,
    stacked_models: Dict[
        str,
        Dict[
            str,
            Union[KNeighborsRegressor, Ridge, DecisionTreeRegressor, LinearRegression],
        ],
    ],
) -> pd.DataFrame:
    """
    Impute missing values using the trained stacked models.

    Args:
    - data (pd.DataFrame): Data with missing values.
    - features (list): List of feature columns.
    - stacked_models (dict): Dictionary containing trained base models
                                and meta-model for each target.

    Returns:
    - pd.DataFrame: Data with imputed values.
    """
    missing_data = data[
        data["goldat15"].isnull() | data["xpat15"].isnull() | data["csat15"].isnull()
    ]
    for target, models in stacked_models.items():
        base_predictions = {
            name: model.predict(missing_data[features])
            for name, model in models["base_models"].items()
        }
        base_predictions_df = pd.DataFrame(base_predictions)
        final_predictions = models["meta_model"].predict(base_predictions_df)
        data.loc[missing_data.index, target] = final_predictions

    return data


def process_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> pd.DataFrame:
    """
    Process the data by generating features, training stacked models,
        and imputing missing values.

    Args:
    - data (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: Processed dataframe.
    """
    data = generate_features(data)

    # One-hot encode the 'position' column
    position_dummies = pd.get_dummies(data["position"], prefix="position")
    data = pd.concat([data, position_dummies], axis=1)

    # Drop rows with NaN target values for training and validation
    training_data = data.dropna(subset=["goldat15", "xpat15", "csat15"])

    train_data, val_data = train_test_split(
        training_data, test_size=test_size, random_state=random_state
    )

    # Extend the feature list to include one-hot encoded position columns
    features_extended = [
        "egpm",
        "gamelength",
        "totalgold",
        "ckpm",
        "monsterkills",
        "minionkills",
        "kills",
        "deaths",
        "assists",
    ] + position_dummies.columns.tolist()

    # Train stacked models
    stacked_models = {}
    for target in ["goldat15", "xpat15", "csat15"]:
        base_models, meta_model = train_stacked_model(
            train_data, val_data, features_extended, target
        )
        stacked_models[target] = {"base_models": base_models, "meta_model": meta_model}

        # Print validation metrics
        logger.info(f"\nPerformance Metrics for Target: {target}")
        logger.info("=" * 40)

        # Base models
        for model_name, model in base_models.items():
            predictions = model.predict(val_data[features_extended])
            mae = mean_absolute_error(val_data[target], predictions)
            r2 = r2_score(val_data[target], predictions)
            logger.info(f"{model_name} - MAE: {mae:.2f}, R2: {r2:.2f}")

        # Ensemble model
        val_predictions_base = {
            name: model.predict(val_data[features_extended])
            for name, model in base_models.items()
        }
        val_predictions_df = pd.DataFrame(val_predictions_base)
        ensemble_predictions = meta_model.predict(val_predictions_df)
        mae = mean_absolute_error(val_data[target], ensemble_predictions)
        r2 = r2_score(val_data[target], ensemble_predictions)
        logger.info(f"Ensemble - MAE: {mae:.2f}, R2: {r2:.2f}")
        logger.info("=" * 40)

    # Impute missing values
    data = impute_missing_values(data, features_extended, stacked_models)

    return data
