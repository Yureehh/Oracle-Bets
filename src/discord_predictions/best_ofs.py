"""
BestOfs Class

This module contains the BestOfs class, which is used to calculate the results of different best-of series.
"""


class BestOfs:
    """
    Calculate the likelihood of each team winning a best of series given their odds.
    """

    PROB_SUM_ERROR_MESSAGE = "Probabilities do not sum to 1"

    @staticmethod
    def validate_probabilities(t1odds, t2odds):
        """Validate that the sum of probabilities is equal to 1."""
        if not round(t1odds + t2odds, 5) == 1.0:
            raise ValueError(BestOfs.PROB_SUM_ERROR_MESSAGE)

    @staticmethod
    def best_of_one(t1name, t1odds, t2name, t2odds):
        """Calculate the likelihood of each team winning a best of one series."""
        BestOfs.validate_probabilities(t1odds, t2odds)

        output = (
            f"```Overall Likelihood Of {t1name} To Win Game: {(t1odds * 100):.2f}%\n\n"
            f"Overall Likelihood Of {t2name} To Win Game: {(t2odds * 100):.2f}%```"
        )
        return output

    @staticmethod
    def best_of_three(t1name, t1odds, t2name, t2odds):
        """Calculate the likelihood of each team winning a best of three series."""
        BestOfs.validate_probabilities(t1odds, t2odds)

        t1_20, t1_21 = t1odds**2, 2 * t1odds**2 * t2odds
        t2_20, t2_21 = t2odds**2, 2 * t2odds**2 * t1odds

        BestOfs.validate_probabilities(t1_20 + t1_21, t2_20 + t2_21)

        t1_win_series = t1_20 + t1_21
        t2_win_series = t2_20 + t2_21
        t1_win_at_least_one = t1_win_series + t2_21
        t2_win_at_least_one = t2_win_series + t1_21
        exactly_three_games = t1_21 + t2_21

        output = (
            f"```Overall Likelihood Of {t1name} To Win Series: {(t1_win_series * 100):.2f}%\n\n"
            f"\tProbability {t1name} wins 2/0: {(t1_20 * 100):.2f}%\n"
            f"\tProbability {t1name} wins 2/1: {(t1_21 * 100):.2f}%\n\n"
            f"Overall Likelihood Of {t2name} To Win Series: {(t2_win_series * 100):.2f}%\n\n"
            f"\tProbability {t2name} wins 2/0: {(t2_20 * 100):.2f}%\n"
            f"\tProbability {t2name} wins 2/1: {(t2_21 * 100):.2f}%\n\n"
            f"Overall likelihoods of each team winning at least 1 game:\n\n"
            f"\tProbability {t1name} wins at least 1 game: {(t1_win_at_least_one * 100):.2f}%\n"
            f"\tProbability {t2name} wins at least 1 game: {(t2_win_at_least_one * 100):.2f}%\n\n"
            f"Overall Likelihood Of Exactly 3 Games: {(exactly_three_games * 100):.2f}%```"
        )
        return output

    @staticmethod
    def best_of_five(t1name, t1odds, t2name, t2odds):
        """Calculate the likelihood of each team winning a best of five series."""
        BestOfs.validate_probabilities(t1odds, t2odds)

        t1_30, t1_31, t1_32 = t1odds**3, 3 * t1odds**3 * t2odds, 6 * t1odds**3 * t2odds**2
        t2_30, t2_31, t2_32 = t2odds**3, 3 * t2odds**3 * t1odds, 6 * t2odds**3 * t1odds**2

        BestOfs.validate_probabilities(t1_30 + t1_31 + t1_32, t2_30 + t2_31 + t2_32)

        t1_win_series = t1_30 + t1_31 + t1_32
        t2_win_series = t2_30 + t2_31 + t2_32
        t1_win_at_least_one = t1_30 + t1_31 + t1_32 + t2_31 + t2_32
        t2_win_at_least_one = t2_30 + t2_31 + t2_32 + t1_31 + t1_32
        at_least_four_games = t1_31 + t1_32 + t2_31 + t2_32
        exactly_five_games = t1_32 + t2_32

        output = (
            f"```Overall Likelihood Of {t1name} To Win Series: {(t1_win_series * 100):.2f}%\n\n"
            f"\tProbability {t1name} wins 3/0: {(t1_30 * 100):.2f}%\n"
            f"\tProbability {t1name} wins 3/1: {(t1_31 * 100):.2f}%\n"
            f"\tProbability {t1name} wins 3/2: {(t1_32 * 100):.2f}%\n\n"
            f"Overall Likelihood Of {t2name} To Win Series: {(t2_win_series * 100):.2f}%\n\n"
            f"\tProbability {t2name} wins 3/0: {(t2_30 * 100):.2f}%\n"
            f"\tProbability {t2name} wins 3/1: {(t2_31 * 100):.2f}%\n"
            f"\tProbability {t2name} wins 3/2: {(t2_32 * 100):.2f}%\n\n"
            f"Overall likelihoods of each team winning at least 1 game:\n\n"
            f"\tProbability {t1name} wins at least 1 game: {(t1_win_at_least_one * 100):.2f}%\n"
            f"\tProbability {t2name} wins at least 1 game: {(t2_win_at_least_one * 100):.2f}%\n\n"
            f"Overall Likelihood Of At Least 4 Games: {(at_least_four_games * 100):.2f}%\n\n"
            f"Overall Likelihood Of Exactly 5 Games: {(exactly_five_games * 100):.2f}%```"
        )
        return output
