"""Pure prematch LoL market probability transformations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesPathDistribution:
    """Legal terminal BO-series outcomes derived from one map probability."""

    team_a_win: float
    exact_score: dict[tuple[int, int], float]
    total_maps: dict[int, float]
    map_differential: dict[int, float]


def enumerate_series_paths(
    team_a_map_probability: float,
    *,
    best_of: int,
) -> SeriesPathDistribution:
    """Enumerate paths that stop immediately when either team clinches."""
    probability = float(team_a_map_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("map probability must be between zero and one")
    if best_of not in {1, 3, 5}:
        raise ValueError("best_of must be 1, 3, or 5")
    wins_needed = best_of // 2 + 1
    exact_score: dict[tuple[int, int], float] = {}

    def visit(a_wins: int, b_wins: int, path_probability: float) -> None:
        if wins_needed in {a_wins, b_wins}:
            score = (a_wins, b_wins)
            exact_score[score] = exact_score.get(score, 0.0) + path_probability
            return
        visit(a_wins + 1, b_wins, path_probability * probability)
        visit(a_wins, b_wins + 1, path_probability * (1.0 - probability))

    visit(0, 0, 1.0)
    totals: dict[int, float] = {}
    differentials: dict[int, float] = {}
    for (a_wins, b_wins), value in exact_score.items():
        totals[a_wins + b_wins] = totals.get(a_wins + b_wins, 0.0) + value
        difference = a_wins - b_wins
        differentials[difference] = differentials.get(difference, 0.0) + value
    return SeriesPathDistribution(
        team_a_win=sum(
            value for (a_wins, b_wins), value in exact_score.items() if a_wins > b_wins
        ),
        exact_score=exact_score,
        total_maps=totals,
        map_differential=differentials,
    )
