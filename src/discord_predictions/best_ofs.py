"""
BestOfs Module

This module contains the BestOfs class, which is used to calculate the results of different best-of series.
"""


class BestOfs:
    """Calculate the likelihood of each team winning a best-of series given their odds."""

    PROB_SUM_ERROR_MESSAGE = "Probabilities do not sum to 1."
    TOLERANCE = 1e-5

    @staticmethod
    def validate_probabilities(*probs: float) -> None:
        """
        Validate that the sum of provided probabilities is equal to 1.

        Args:
            *probs (float): Probabilities to validate.

        Raises:
            ValueError: If the sum of probabilities does not equal 1 within the defined tolerance.
        """
        total = sum(probs)
        if abs(total - 1.0) > BestOfs.TOLERANCE:
            raise ValueError(BestOfs.PROB_SUM_ERROR_MESSAGE)

    @staticmethod
    def _format_percentage(prob: float) -> str:
        """
        Format a probability as a percentage string.

        Args:
            prob (float): Probability value between 0 and 1.

        Returns:
            str: Formatted percentage string.
        """
        return f"{prob * 100:.2f}%"

    @staticmethod
    def best_of_one(t1_name: str, t1_odds: float, t2_name: str, t2_odds: float) -> str:
        """
        Calculate the likelihood of each team winning a best-of-one series.

        Args:
            t1_name (str): Name of Team 1.
            t1_odds (float): Probability of Team 1 winning a single game.
            t2_name (str): Name of Team 2.
            t2_odds (float): Probability of Team 2 winning a single game.

        Returns:
            str: Formatted string with likelihoods.
        """
        BestOfs.validate_probabilities(t1_odds, t2_odds)

        return (
            f"Overall Likelihood Of {t1_name} To Win Game: {BestOfs._format_percentage(t1_odds)}\n\n"
            f"Overall Likelihood Of {t2_name} To Win Game: {BestOfs._format_percentage(t2_odds)}"
        )

    @staticmethod
    def best_of_two(t1_name: str, t1_odds: float, t2_name: str, t2_odds: float) -> str:
        """
        Calculate the likelihood of each team winning a best-of-two series (allowing for tie results).

        Args:
            t1_name (str): Name of Team 1.
            t1_odds (float): Probability of Team 1 winning a single game.
            t2_name (str): Name of Team 2.
            t2_odds (float): Probability of Team 2 winning a single game.

        Returns:
            str: Formatted string with likelihoods.
        """
        BestOfs.validate_probabilities(t1_odds, t2_odds)

        # Possible outcomes: Team1 wins 2-0, Team2 wins 2-0, or a tie (1-1)
        t1_win_2 = t1_odds**2
        t2_win_2 = t2_odds**2
        tie = 2 * t1_odds * t2_odds  # Two ways the series can tie: T1 wins one game, T2 wins the other

        BestOfs.validate_probabilities(t1_win_2 + t2_win_2 + tie)

        return (
            f"Likelihood Of {t1_name} To Win a single game: {BestOfs._format_percentage(t1_odds)}\n"
            f"Likelihood Of {t2_name} To Win a single game: {BestOfs._format_percentage(t2_odds)}\n\n"
            f"\tProbability {t1_name} wins 2-0: {BestOfs._format_percentage(t1_win_2)}\n"
            f"\tProbability {t2_name} wins 2-0: {BestOfs._format_percentage(t2_win_2)}\n\n"
            f"\tProbability of a Tie (1-1): {BestOfs._format_percentage(tie)}"
        )

    @staticmethod
    def best_of_three(t1_name: str, t1_odds: float, t2_name: str, t2_odds: float) -> str:
        """
        Calculate the likelihood of each team winning a best-of-three series.

        Args:
            t1_name (str): Name of Team 1.
            t1_odds (float): Probability of Team 1 winning a single game.
            t2_name (str): Name of Team 2.
            t2_odds (float): Probability of Team 2 winning a single game.

        Returns:
            str: Formatted string with likelihoods.
        """
        BestOfs.validate_probabilities(t1_odds, t2_odds)

        # Team 1 outcomes
        t1_win_2_0 = t1_odds**2  # Wins first two games
        t1_win_2_1 = 2 * t1_odds**2 * t2_odds  # Wins in three games

        # Team 2 outcomes
        t2_win_2_0 = t2_odds**2
        t2_win_2_1 = 2 * t2_odds**2 * t1_odds

        # Overall series win probabilities
        t1_win_series = t1_win_2_0 + t1_win_2_1
        t2_win_series = t2_win_2_0 + t2_win_2_1

        # Validate probabilities
        BestOfs.validate_probabilities(t1_win_series + t2_win_series)

        # At least one win probabilities
        t1_win_at_least_one = t1_win_series + t2_win_2_1
        t2_win_at_least_one = t2_win_series + t1_win_2_1

        # Exactly three games
        exactly_three_games = t1_win_2_1 + t2_win_2_1

        return (
            f"Likelihood Of {t1_name} To Win a single game: {BestOfs._format_percentage(t1_odds)}\n"
            f"Likelihood Of {t2_name} To Win a single game: {BestOfs._format_percentage(t2_odds)}\n\n"
            f"Overall Likelihood Of {t1_name} To Win Series: {BestOfs._format_percentage(t1_win_series)}\n\n"
            f"\tProbability {t1_name} wins 2-0: {BestOfs._format_percentage(t1_win_2_0)}\n"
            f"\tProbability {t1_name} wins 2-1: {BestOfs._format_percentage(t1_win_2_1)}\n\n"
            f"Overall Likelihood Of {t2_name} To Win Series: {BestOfs._format_percentage(t2_win_series)}\n\n"
            f"\tProbability {t2_name} wins 2-0: {BestOfs._format_percentage(t2_win_2_0)}\n"
            f"\tProbability {t2_name} wins 2-1: {BestOfs._format_percentage(t2_win_2_1)}\n\n"
            f"Overall likelihoods of each team winning at least 1 game:\n\n"
            f"\tProbability {t1_name} wins at least 1 game: {BestOfs._format_percentage(t1_win_at_least_one)}\n"
            f"\tProbability {t2_name} wins at least 1 game: {BestOfs._format_percentage(t2_win_at_least_one)}\n\n"
            f"Overall Likelihood Of Exactly 3 Games: {BestOfs._format_percentage(exactly_three_games)}"
        )

    @staticmethod
    def best_of_five(t1_name: str, t1_odds: float, t2_name: str, t2_odds: float) -> str:
        """
        Calculate the likelihood of each team winning a best-of-five series.

        Args:
            t1_name (str): Name of Team 1.
            t1_odds (float): Probability of Team 1 winning a single game.
            t2_name (str): Name of Team 2.
            t2_odds (float): Probability of Team 2 winning a single game.

        Returns:
            str: Formatted string with likelihoods.
        """
        BestOfs.validate_probabilities(t1_odds, t2_odds)

        # Team 1 outcomes
        t1_win_3_0 = t1_odds**3
        t1_win_3_1 = 3 * t1_odds**3 * t2_odds
        t1_win_3_2 = 6 * t1_odds**3 * t2_odds**2

        # Team 2 outcomes
        t2_win_3_0 = t2_odds**3
        t2_win_3_1 = 3 * t2_odds**3 * t1_odds
        t2_win_3_2 = 6 * t2_odds**3 * t1_odds**2

        # Overall series win probabilities
        t1_win_series = t1_win_3_0 + t1_win_3_1 + t1_win_3_2
        t2_win_series = t2_win_3_0 + t2_win_3_1 + t2_win_3_2

        # Validate probabilities
        BestOfs.validate_probabilities(t1_win_series + t2_win_series)

        # At least one win probabilities
        t1_win_at_least_one = t1_win_series + t2_win_3_1 + t2_win_3_2
        t2_win_at_least_one = t2_win_series + t1_win_3_1 + t1_win_3_2

        # Game counts
        exactly_three_games = t1_win_3_0 + t2_win_3_0
        at_least_four_games = t1_win_3_1 + t1_win_3_2 + t2_win_3_1 + t2_win_3_2
        exactly_five_games = t1_win_3_2 + t2_win_3_2

        return (
            f"Likelihood Of {t1_name} To Win a single game: {BestOfs._format_percentage(t1_odds)}\n"
            f"Likelihood Of {t2_name} To Win a single game: {BestOfs._format_percentage(t2_odds)}\n\n"
            f"Overall Likelihood Of {t1_name} To Win Series: {BestOfs._format_percentage(t1_win_series)}\n\n"
            f"\tProbability {t1_name} wins 3-0: {BestOfs._format_percentage(t1_win_3_0)}\n"
            f"\tProbability {t1_name} wins 3-1: {BestOfs._format_percentage(t1_win_3_1)}\n"
            f"\tProbability {t1_name} wins 3-2: {BestOfs._format_percentage(t1_win_3_2)}\n\n"
            f"Overall Likelihood Of {t2_name} To Win Series: {BestOfs._format_percentage(t2_win_series)}\n\n"
            f"\tProbability {t2_name} wins 3-0: {BestOfs._format_percentage(t2_win_3_0)}\n"
            f"\tProbability {t2_name} wins 3-1: {BestOfs._format_percentage(t2_win_3_1)}\n"
            f"\tProbability {t2_name} wins 3-2: {BestOfs._format_percentage(t2_win_3_2)}\n\n"
            f"Overall likelihoods of each team winning at least 1 game:\n\n"
            f"\tProbability {t1_name} wins at least 1 game: {BestOfs._format_percentage(t1_win_at_least_one)}\n"
            f"\tProbability {t2_name} wins at least 1 game: {BestOfs._format_percentage(t2_win_at_least_one)}\n\n"
            f"Overall Likelihood Of Exactly 3 Games: {BestOfs._format_percentage(exactly_three_games)}\n\n"
            f"Overall Likelihood Of At Least 4 Games: {BestOfs._format_percentage(at_least_four_games)}\n\n"
            f"Overall Likelihood Of Exactly 5 Games: {BestOfs._format_percentage(exactly_five_games)}"
        )
