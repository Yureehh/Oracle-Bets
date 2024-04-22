import numpy as np
import pandas as pd
import pytest
import trueskill

from src.feature_engineering.ratings_features.trueskill import (
    calculate_and_merge_team_statistics,
    compute_win_probability,
    generate_match_array,
    initialize_player_ratings,
    merge_player_stats,
    preprocess_data,
    trueskill_model,
    update_player_ratings,
)
from utils.utils import get_sorting_keys


class TestTrueSkillRating:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup common test variables"""
        self.ts_env = trueskill.TrueSkill()
        self.initial_mu = 25
        self.default_sigma = 8.333
        self.initial_sigma = 42
        self.match_col_names = [
            "gameid",
            "date",
            "league",
            "ckpm",
            "blue_teamname",
            "blue_teamid",
            "team_egpm",
            "team_kpm",
            "blue_player4",
            "blue_player2",
            "blue_player3",
            "blue_player5",
            "blue_player1",
            "blue_result",
            "red_teamname",
            "red_teamid",
            "red_team_egpm",
            "red_team_kpm",
            "red_player4",
            "red_player2",
            "red_player3",
            "red_player5",
            "red_player1",
            "red_result",
        ]
        self.true_skill_col_names = [
            "blue_win_probability",
            "blue_player1_mu",
            "blue_player2_mu",
            "blue_player3_mu",
            "blue_player4_mu",
            "blue_player5_mu",
            "red_player1_mu",
            "red_player2_mu",
            "red_player3_mu",
            "red_player4_mu",
            "red_player5_mu",
            "blue_player1_sigma",
            "blue_player2_sigma",
            "blue_player3_sigma",
            "blue_player4_sigma",
            "blue_player5_sigma",
            "red_player1_sigma",
            "red_player2_sigma",
            "red_player3_sigma",
            "red_player4_sigma",
            "red_player5_sigma",
        ]
        self.sample_player_data = pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                ],
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                ],
                "gameid": [11, 11, 11, 11, 12, 12, 12, 12],
                "teamname": ["team1", "team1", "team2", "team2"] * 2,
                "teamid": [1, 1, 2, 2, 1, 1, 2, 2],
                "playerid": [
                    "player1",
                    "player2",
                    "playerA",
                    "playerB",
                    "player1",
                    "player2",
                    "playerA",
                    "playerB",
                ],
                "position": ["top", "mid", "top", "mid", "top", "mid", "top", "mid"],
                "side": ["Blue", "Blue", "Red", "Red", "Red", "Red", "Blue", "Blue"],
                "egpm": [300, 200, 100, 300, 250, 150, 350, 50],
                "team_kpm": [2, 2, 1, 1, 1, 1, 1, 1],
                "ckpm": [3, 3, 3, 3, 2, 2, 2, 2],
                "result": [1, 1, 0, 0, 1, 1, 0, 0],
            }
        )
        self.sample_team_data = pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-03",
                ],
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "La Liga",
                    "La Liga",
                ],
                "gameid": [11, 11, 12, 12, 13, 13],
                "teamid": [1, 2, 2, 1, 3, 4],
                "teamname": ["team1", "team2", "team2", "team1", "team3", "team4"],
                "position": ["team", "team", "team", "team", "team", "team"],
                "side": ["Blue", "Red", "Blue", "Red", "Blue", "Red"],
                "result": [1, 0, 0, 1, 1, 0],
            }
        )
        self.extended_sample_player_data = pd.DataFrame(
            {
                "date": [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                    "2021-01-02",
                ],
                "league": [
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                    "Premier League",
                ],
                "gameid": [
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    11,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                ],
                "teamname": [
                    "team1",
                    "team1",
                    "team1",
                    "team1",
                    "team1",
                    "team2",
                    "team2",
                    "team2",
                    "team2",
                    "team2",
                ]
                * 2,
                "teamid": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "playerid": [
                    "player1",
                    "player2",
                    "player3",
                    "player4",
                    "player5",
                    "playerA",
                    "playerB",
                    "playerC",
                    "playerD",
                    "playerE",
                    "player1",
                    "player2",
                    "player3",
                    "player4",
                    "player5",
                    "playerA",
                    "playerB",
                    "playerC",
                    "playerD",
                    "playerE",
                ],
                "position": [
                    "top",
                    "jungle",
                    "mid",
                    "adc",
                    "supp",
                    "top",
                    "jungle",
                    "mid",
                    "adc",
                    "supp",
                    "top",
                    "jungle",
                    "mid",
                    "adc",
                    "supp",
                    "top",
                    "jungle",
                    "mid",
                    "adc",
                    "supp",
                ],
                "side": [
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Red",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                    "Blue",
                ],
                "egpm": [
                    300,
                    200,
                    300,
                    200,
                    300,
                    100,
                    300,
                    100,
                    300,
                    100,
                    250,
                    150,
                    250,
                    150,
                    250,
                    350,
                    50,
                    350,
                    50,
                    350,
                ],
                "team_kpm": [
                    2,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
                "ckpm": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                "result": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )

    def test_initialize_player_ratings(self):
        """Test initializing player ratings with default and custom TrueSkill environments"""
        default_ratings = initialize_player_ratings(self.sample_player_data)
        custom_ratings = initialize_player_ratings(
            self.sample_player_data, self.initial_mu, self.initial_sigma, self.ts_env
        )

        # Assert that ratings are initialized and have the correct sigma
        for rating in default_ratings.values():
            assert rating.sigma == self.default_sigma

        for rating in custom_ratings.values():
            assert rating.sigma == self.initial_sigma
            assert isinstance(rating, trueskill.Rating), "Rating should be an instance of trueskill.Rating"

    def test_preprocess_data_creates_team_egpm(self):
        """Test that the preprocess_data function correctly calculates and includes 'team_egpm'"""

        processed_data = preprocess_data(self.sample_player_data)
        assert (
            "team_egpm" in processed_data.columns
        ), "'team_egpm' should be calculated and included in the processed data."

        # Check processed-data is sorted
        assert processed_data.sort_values(by=get_sorting_keys("team")).equals(processed_data)

    def test_generate_match_array(self):
        match_array = generate_match_array(preprocess_data(self.sample_player_data))
        expected_length = 2  # Based on the unique game IDs in the sample data
        assert len(match_array) == expected_length, "The match array length should match the number of unique games."
        assert all(isinstance(item, list) for item in match_array), "All items in the match array should be lists."
        # This should be 24 elements long, 12 for each team, but I just inserted 2 players per team and not 5
        assert all(len(item) == 18 for item in match_array), "Each item in the match array should contain 24 elements."

        # Flatten the match_array to a single list to check for player IDs
        flattened_match_array = [item for sublist in match_array for item in sublist]
        assert all(
            player in flattened_match_array for player in self.sample_player_data["playerid"].unique()
        ), "All playerids in the player_data should be included in the match_array."

    def test_compute_win_probability(self):
        team1 = [
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
        ]
        team2 = [
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
        ]
        win_prob = compute_win_probability(team1, team2, self.ts_env)
        assert 0 <= win_prob <= 1, "Win probability should be between 0 and 1."

        team3 = [
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
        ]
        team4 = [
            self.ts_env.create_rating(mu=1),
            self.ts_env.create_rating(mu=1),
            self.ts_env.create_rating(mu=1),
            self.ts_env.create_rating(mu=1),
            self.ts_env.create_rating(mu=1),
        ]
        win_prob_team3_team4 = compute_win_probability(team3, team4, self.ts_env)
        assert win_prob_team3_team4 > 0.9, "Team3 should have a higher win probability than Team4."

        team3 = [
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
            self.ts_env.create_rating(),
        ]
        team4 = [
            self.ts_env.create_rating(mu=50),
            self.ts_env.create_rating(mu=50),
            self.ts_env.create_rating(mu=50),
            self.ts_env.create_rating(mu=50),
            self.ts_env.create_rating(mu=50),
        ]
        win_prob_team3_team4 = compute_win_probability(team3, team4, self.ts_env)
        assert win_prob_team3_team4 < 0.1, "Team4 should have a higher win probability than Team3."

    def test_update_player_ratings(self):
        gameid_dict = {}
        match_array = generate_match_array(preprocess_data(self.extended_sample_player_data))
        match_df = pd.DataFrame(match_array, columns=self.match_col_names)

        first_match = match_df.iloc[0]
        rating_dict = initialize_player_ratings(
            self.extended_sample_player_data, self.initial_mu, self.default_sigma, self.ts_env
        )
        ts_preview = update_player_ratings(first_match, rating_dict, gameid_dict, self.ts_env)

        assert (
            len(ts_preview) == 21
        ), "The returned Series should contain 21 elements (1 win probability + 10 mu + 10 sigma values)."
        assert gameid_dict, "The gameid_dict should be updated with the processed game."

        full_ts = match_df.apply(
            lambda x: update_player_ratings(x, rating_dict, gameid_dict, self.ts_env),
            axis=1,
            result_type="expand",
        )

        assert len(full_ts) == len(
            match_df
        ), "The returned DataFrame should have the same number of rows as the input match_df."

        # Asser full_ts has the shape of match_df with the TrueSkill columns
        assert full_ts.shape == (
            len(match_df),
            21,
        ), "The returned DataFrame should have the same shape as the input match_df with the TrueSkill columns."

    def test_calculate_and_merge_team_statistics(self):
        gameid_dict = {}
        rating_dict = initialize_player_ratings(
            self.extended_sample_player_data, self.initial_mu, self.default_sigma, self.ts_env
        )
        match_df = pd.DataFrame(
            generate_match_array(preprocess_data(self.extended_sample_player_data)),
            columns=self.match_col_names,
        )

        updates = match_df.apply(
            lambda x: update_player_ratings(x, rating_dict, gameid_dict, self.ts_env),
            axis=1,
            result_type="expand",
        )
        updates.columns = self.true_skill_col_names
        match_df[self.true_skill_col_names] = updates

        match_df["red_win_probability"] = 1 - match_df["blue_win_probability"]
        # Add the expected result column based on the win probability
        match_df["blue_expected_result"] = np.where(match_df["blue_win_probability"] > 0.5, 1, 0)

        team_data_before = self.sample_team_data
        team_data_after = calculate_and_merge_team_statistics(match_df, team_data_before)

        # Verify the team_data DataFrame has been updated with additional columns for TrueSkill statistics
        expected_columns = [
            "trueskill_sum_mu",
            "trueskill_sigma_squared",
            "trueskill_opponent_sum_mu",
            "trueskill_opponent_sigma_squared",
            "trueskill_diff",
        ]
        for col in expected_columns:
            assert (
                col in team_data_after.columns
            ), f"'{col}' should be in the columns of the updated team_data DataFrame."

        # Verify the total number of rows in the team_data DataFrame has not changed
        assert len(team_data_after) == len(
            team_data_before
        ), "The number of rows in the team_data DataFrame should not change."

        # Verify the total number of columns in the team_data DataFrame has increased by the number of expected columns
        assert len(team_data_after.columns) == len(team_data_before.columns) + len(
            expected_columns
        ), "The number of columns in the team_data DataFrame should increase by the number of expected columns."

    def test_merge_player_stats(self):
        # Prepare the match and player data
        gameid_dict = {}
        rating_dict = initialize_player_ratings(
            self.extended_sample_player_data, self.initial_mu, self.default_sigma, self.ts_env
        )
        match_df = pd.DataFrame(
            generate_match_array(preprocess_data(self.extended_sample_player_data)),
            columns=self.match_col_names,
        )
        updates = match_df.apply(
            lambda x: update_player_ratings(x, rating_dict, gameid_dict, self.ts_env),
            axis=1,
            result_type="expand",
        )
        updates.columns = self.true_skill_col_names
        match_df[self.true_skill_col_names] = updates

        # Merge player stats
        player_data_before = self.extended_sample_player_data.copy()
        player_data_after = merge_player_stats(player_data_before, match_df)

        # Verify new columns for TrueSkill ratings are added to player_data DataFrame
        expected_columns = [
            "trueskill_mu",
            "trueskill_sigma",
            "trueskill_opponent_mu",
            "trueskill_opponent_sigma",
        ]
        for col in expected_columns:
            assert (
                col in player_data_after.columns
            ), f"'{col}' should be in the columns of the updated player_data DataFrame."

        # Verify that the TrueSkill ratings are non-null and correctly merged for at least one player
        sample_player_id = self.extended_sample_player_data["playerid"].iloc[0]
        sample_player_data = player_data_after[player_data_after["playerid"] == sample_player_id]
        for col in expected_columns:
            assert (
                not sample_player_data[col].isnull().any()
            ), f"'{col}' should have non-null values for the sample player."

    def test_trueskill_model(self):
        player_data, team_data, player_ratings_dict = trueskill_model(
            self.extended_sample_player_data, self.sample_team_data
        )

        # Verify that the returned player_data DataFrame contains TrueSkill rating columns
        expected_player_columns = [
            "trueskill_mu",
            "trueskill_sigma",
            "trueskill_opponent_mu",
            "trueskill_opponent_sigma",
        ]
        for col in expected_player_columns:
            assert (
                col in player_data.columns
            ), f"'{col}' should be in the columns of the returned player_data DataFrame."

        # Verify that the returned team_data DataFrame contains aggregated TrueSkill statistics
        expected_team_columns = [
            "trueskill_sum_mu",
            "trueskill_sigma_squared",
            "trueskill_opponent_sum_mu",
            "trueskill_opponent_sigma_squared",
            "trueskill_diff",
        ]
        for col in expected_team_columns:
            assert col in team_data.columns, f"'{col}' should be in the columns of the returned team_data DataFrame."

        # Verify that player_ratings_dict is updated with new TrueSkill ratings
        sample_player_id = list(player_ratings_dict.keys())[0]
        sample_rating = player_ratings_dict[sample_player_id]
        assert isinstance(
            sample_rating, trueskill.Rating
        ), "The player_ratings_dict should contain updated trueskill.Rating instances."
        assert (
            sample_rating.mu != self.ts_env.mu or sample_rating.sigma != self.default_sigma
        ), "The sample player's rating should be updated from the default values."
