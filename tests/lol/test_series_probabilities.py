from __future__ import annotations

import math

import pytest
from lol_bets.inference.series import SeriesStateError, derive_series_distribution


def test_bo3_distribution_is_derived_from_map_probability():
    distribution = derive_series_distribution(3, [0.6, 0.6, 0.6])

    assert distribution.team_a_win == pytest.approx(0.648)
    assert distribution.team_b_win == pytest.approx(0.352)
    assert distribution.draw == 0
    assert distribution.score_probabilities == pytest.approx(
        {"2-0": 0.36, "2-1": 0.288, "0-2": 0.16, "1-2": 0.192}
    )
    assert distribution.total_maps_probabilities == pytest.approx({2: 0.52, 3: 0.48})


def test_bo2_supports_draws_without_inventing_a_series_winner():
    distribution = derive_series_distribution(2, [0.6, 0.6])

    assert distribution.team_a_win == pytest.approx(0.36)
    assert distribution.draw == pytest.approx(0.48)
    assert distribution.team_b_win == pytest.approx(0.16)
    assert distribution.score_probabilities == pytest.approx(
        {"2-0": 0.36, "1-1": 0.48, "0-2": 0.16}
    )


def test_after_map_1_state_uses_only_remaining_map_probabilities():
    distribution = derive_series_distribution(
        3,
        [0.5, 0.5],
        team_a_maps=0,
        team_b_maps=1,
    )

    assert distribution.team_a_win == pytest.approx(0.25)
    assert distribution.team_b_win == pytest.approx(0.75)
    assert distribution.score_probabilities == pytest.approx(
        {"0-2": 0.5, "1-2": 0.25, "2-1": 0.25}
    )


def test_map_number_specific_probabilities_remain_coherent():
    distribution = derive_series_distribution(3, [0.6, 0.7, 0.8])

    assert distribution.score_probabilities["2-0"] == pytest.approx(0.42)
    assert distribution.score_probabilities["2-1"] == pytest.approx(0.368)
    assert math.fsum(distribution.score_probabilities.values()) == pytest.approx(1)
    assert (
        distribution.team_a_win + distribution.team_b_win + distribution.draw
        == pytest.approx(1)
    )


@pytest.mark.parametrize(
    ("best_of", "probabilities", "score"),
    [
        (4, [0.5] * 4, (0, 0)),
        (3, [1.1, 0.5, 0.5], (0, 0)),
        (3, [0.5], (0, 1)),
        (3, [0.5, 0.5], (2, 0)),
        (3, [0.5, 0.5, 0.5], (1, 1)),
    ],
)
def test_invalid_or_already_terminal_series_state_is_rejected(
    best_of, probabilities, score
):
    with pytest.raises(SeriesStateError):
        derive_series_distribution(
            best_of,
            probabilities,
            team_a_maps=score[0],
            team_b_maps=score[1],
        )
