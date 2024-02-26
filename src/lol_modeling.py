# Housekeeping
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from openskill.models import PlackettLuce


@dataclass
class FeatureEngineering:
    raw_data: pd.DataFrame
    column_names: List[str]
    target_col: str
    player_mapping: dict = field(default_factory=dict)
    team_mapping: dict = field(default_factory=dict)

    def player_map(self):
        """Create a player mapping from player names to unique IDs."""
        # Implementation ...

    def team_map(self):
        """Create a team mapping from team names to unique IDs."""
        # Implementation ...

    def apply_maps(self):
        """Apply the player and team mappings to the raw data."""
        # Implementation ...

    def feature_engineer(self):
        """Engineer additional features for the dataset."""
        # Implementation ...

    def finalize_features(self):
        """Finalize the feature set for modeling."""
        # Implementation ...

    def preprocess(self):
        """Complete preprocessing for the dataset."""
        self.player_map()
        self.team_map()
        self.apply_maps()
        self.feature_engineer()
        self.finalize_features()
        return self.raw_data


@dataclass
class LoLModeling:
    # feature_engineering: FeatureEngineering
    model: Optional[Any] = None

    @staticmethod
    def calculate_elo(
        df: pd.DataFrame, entity: str, initial_elo: int = 1200, k: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Elo ratings_features for entities in a DataFrame.

        Args:
        df (pd.DataFrame): The DataFrame containing match results.
        entity (str): The name of the entity for which to calculate Elo ratings_features.
        initial_elo (int, optional): The initial Elo rating for entities. Defaults to 1200.
        k (int, optional): The K-factor used in Elo rating updates. Defaults to 20.

        Returns:
        pd.DataFrame: The DataFrame with Elo ratings_features added.
        """

        def _expected(elo_a: float, elo_b: float) -> float:
            """
            Calculate the expected outcome of a match between two entities.

            Args:
            elo_a (float): The Elo rating of the first entity.
            elo_b (float): The Elo rating of the second entity.

            Returns:
            float: The expected outcome of the match for the first entity.
            """
            return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        def _elo(old: float, exp: float, score: float) -> float:
            """
            Update an Elo rating based on the outcome of a match.

            Args:
            old (float): The old Elo rating.
            exp (float): The expected outcome of the match.
            score (float): The actual outcome of the match.

            Returns:
            float: The updated Elo rating.
            """
            if old is None:
                old = initial_elo
            return old + k * (score - exp)

        if entity.lower() == "team":
            sort_keys = ["date", "league", "gameid", "result"]
        elif entity.lower() == "player":
            sort_keys = ["date", "league", "gameid", "teamid", "position", "result"]
        else:
            raise ValueError(f"Unsupported entity name: {entity}")

        df = df.sort_values(sort_keys).reset_index(drop=True)

        elo_dict: Dict[str, float] = defaultdict(lambda: initial_elo)
        opponent_entity = "opponent" + entity
        entity_column = entity + "name"

        elos, opponent_elos, elo_expected_values = [], [], []

        for _, row in df.iterrows():
            entity_elo = elo_dict[row[entity_column]]
            opponent_elo = elo_dict[row[opponent_entity]]

            expected_outcome = _expected(entity_elo, opponent_elo)

            entity_new_elo = _elo(entity_elo, expected_outcome, row["result"])
            opponent_new_elo = _elo(
                opponent_elo, 1 - expected_outcome, 1 - row["result"]
            )

            elos.append(entity_new_elo)
            opponent_elos.append(opponent_new_elo)
            elo_expected_values.append(expected_outcome)

            elo_dict[row[entity_column]] = entity_new_elo
            elo_dict[row[opponent_entity]] = opponent_new_elo

        df["elo"] = elos
        df["opponent_elo"] = opponent_elos
        df["elo_win_likelihood"] = elo_expected_values

        return df

    @staticmethod
    def calculate_plackett_luce(
        df: pd.DataFrame,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333333333333334,
    ) -> pd.DataFrame:
        """
        Calculate Plackett-Luce ratings_features for players in a DataFrame.
        Why: https://janzert.com/halite/rating-report/

        Args:
        df (pd.DataFrame): The DataFrame containing match results.
        entity: :Entity to split the data on (team or player).
        initial_mu (float, optional): The initial rating for teams. Defaults to 25.0.
        initial_sigma (float, optional): The initial uncertainty
                                         about the rating for teams. Defaults to 8.3333.

        Returns:
        pd.DataFrame: The DataFrame with Plackett-Luce ratings_features added.
        """

        model = PlackettLuce()

        ratings_dict = {}

        # Initialize team ratings_features
        def update_ratings(row):
            team1 = [(ratings_dict.get(row[primary_entity], model.rating()),)]
            team2 = [(ratings_dict.get(row[opponent_entity], model.rating()),)]

            if row["result"] == 1:
                match = [team1, team2]
            else:
                match = [team2, team1]

            [team1, team2] = model.rate(match)

            ratings_dict[row[primary_entity]] = team1[0]
            ratings_dict[row[opponent_entity]] = team2[0]

            row[f"{primary_entity}_rating_mu"] = team1[0].mu
            row[f"{primary_entity}_rating_sigma"] = team1[0].sigma
            row[f"{opponent_entity}_rating_mu"] = team2[0].mu
            row[f"{opponent_entity}_rating_sigma"] = team2[0].sigma

            return row

        df = df.apply(update_ratings, axis=1)

        return df

        return df

    def calculate_egpm(self):
        """Description..."""
        # Implementation ...

    def train_model(self):
        """Train the model using the preprocessed data."""
        # Implementation ...

    def predict(self):
        """Predict outcomes based on the trained model."""
        # Implementation ...
