"""Best-of series probabilities derived from map-win probabilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

DRAW_FORMAT = 2
PROBABILITY_TOLERANCE = 1e-12
ROOT_IMAGINARY_TOLERANCE = 1e-10


class SeriesStateError(ValueError):
    """Raised when a series format, score, or probability path is impossible."""


@dataclass(frozen=True)
class SeriesDistribution:
    best_of: int
    starting_score: tuple[int, int]
    team_a_win: float
    team_b_win: float
    draw: float
    score_probabilities: Mapping[str, float]
    total_maps_probabilities: Mapping[int, float]

    def __post_init__(self) -> None:
        total = self.team_a_win + self.team_b_win + self.draw
        if not isfinite(total) or abs(total - 1.0) > PROBABILITY_TOLERANCE:
            msg = f"Series outcome probabilities must sum to one, got {total}."
            raise SeriesStateError(msg)


def _validate(
    best_of: int,
    probabilities: tuple[float, ...],
    team_a_maps: int,
    team_b_maps: int,
) -> None:
    if best_of not in {1, 2, 3, 5}:
        msg = "best_of must be 1, 2, 3, or 5."
        raise SeriesStateError(msg)
    if team_a_maps < 0 or team_b_maps < 0:
        msg = "Completed map scores must be non-negative."
        raise SeriesStateError(msg)
    maps_played = team_a_maps + team_b_maps
    if maps_played >= best_of:
        msg = "Series state is already terminal."
        raise SeriesStateError(msg)
    target_wins = best_of // 2 + 1
    if best_of != DRAW_FORMAT and max(team_a_maps, team_b_maps) >= target_wins:
        msg = "Series state is already terminal."
        raise SeriesStateError(msg)
    remaining_maps = best_of - maps_played
    if len(probabilities) != remaining_maps:
        msg = (
            f"Expected {remaining_maps} remaining map probabilities, "
            f"received {len(probabilities)}."
        )
        raise SeriesStateError(msg)
    if not all(isfinite(value) and 0 <= value <= 1 for value in probabilities):
        msg = "Map probabilities must be finite values in [0, 1]."
        raise SeriesStateError(msg)


def derive_series_distribution(
    best_of: int,
    map_probabilities: list[float] | tuple[float, ...],
    *,
    team_a_maps: int = 0,
    team_b_maps: int = 0,
) -> SeriesDistribution:
    """Enumerate all remaining legal score paths from one map probability engine."""
    probabilities = tuple(float(value) for value in map_probabilities)
    _validate(best_of, probabilities, team_a_maps, team_b_maps)
    target_wins = best_of // 2 + 1
    states: dict[tuple[int, int], float] = {(team_a_maps, team_b_maps): 1.0}
    scores: defaultdict[str, float] = defaultdict(float)
    totals: defaultdict[int, float] = defaultdict(float)

    for probability in probabilities:
        next_states: defaultdict[tuple[int, int], float] = defaultdict(float)
        for (score_a, score_b), path_probability in states.items():
            branches = (
                (score_a + 1, score_b, path_probability * probability),
                (score_a, score_b + 1, path_probability * (1.0 - probability)),
            )
            for next_a, next_b, branch_probability in branches:
                maps_played = next_a + next_b
                is_terminal = (
                    maps_played == best_of
                    if best_of == DRAW_FORMAT
                    else max(next_a, next_b) == target_wins
                )
                if is_terminal:
                    scores[f"{next_a}-{next_b}"] += branch_probability
                    totals[maps_played] += branch_probability
                else:
                    next_states[(next_a, next_b)] += branch_probability
        states = dict(next_states)
        if not states:
            break

    if states:
        msg = "Remaining probability path did not reach a terminal series state."
        raise SeriesStateError(msg)

    team_a_win = fsum(
        probability
        for score, probability in scores.items()
        if int(score.split("-", maxsplit=1)[0]) > int(score.split("-", maxsplit=1)[1])
    )
    team_b_win = fsum(
        probability
        for score, probability in scores.items()
        if int(score.split("-", maxsplit=1)[0]) < int(score.split("-", maxsplit=1)[1])
    )
    draw = max(0.0, 1.0 - team_a_win - team_b_win) if best_of == DRAW_FORMAT else 0.0
    score_total = fsum(scores.values())
    if abs(score_total - 1.0) > PROBABILITY_TOLERANCE:
        msg = f"Series score probabilities must sum to one, got {score_total}."
        raise SeriesStateError(msg)
    return SeriesDistribution(
        best_of=best_of,
        starting_score=(team_a_maps, team_b_maps),
        team_a_win=team_a_win,
        team_b_win=team_b_win,
        draw=draw,
        score_probabilities=MappingProxyType(dict(sorted(scores.items()))),
        total_maps_probabilities=MappingProxyType(dict(sorted(totals.items()))),
    )


def total_maps_probability_range(
    best_of: int,
    total_maps: int,
    probability_lower: float,
    probability_upper: float,
) -> tuple[float, float]:
    """Return the extrema across a constant-map probability interval."""
    lower, upper = sorted((float(probability_lower), float(probability_upper)))
    if lower < 0 or upper > 1:
        raise SeriesStateError("Probability interval must lie in [0, 1].")

    def probability_at(value: float) -> float:
        distribution = derive_series_distribution(best_of, [value] * best_of)
        return float(distribution.total_maps_probabilities.get(total_maps, 0.0))

    # Total-map probability is a polynomial of degree at most best_of. Its
    # derivative roots contain every interior extremum missed by endpoints.
    nodes = np.linspace(0.0, 1.0, best_of + 1)
    values = np.asarray([probability_at(value) for value in nodes])
    polynomial = np.polynomial.Polynomial.fit(nodes, values, deg=best_of).convert()
    candidates = [lower, upper]
    for root in polynomial.deriv().roots():
        if abs(float(np.imag(root))) > ROOT_IMAGINARY_TOLERANCE:
            continue
        value = float(np.real(root))
        if lower <= value <= upper:
            candidates.append(value)
    probabilities = [probability_at(value) for value in candidates]
    return min(probabilities), max(probabilities)
