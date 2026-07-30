"""Best-of probability dictionaries and compact handicap formatting."""

from __future__ import annotations

from math import isclose

from lol_bets.inference.series import derive_series_distribution

_ABS_TOL = 1e-6
MIN_LONG_SERIES_MAPS = 4


def _canon(p1: float, p2: float | None) -> tuple[float, float]:
    if not 0.0 <= p1 <= 1.0:
        raise ValueError(f"p1 must be in [0,1], got {p1!r}")
    p2 = 1.0 - p1 if p2 is None else p2
    if not 0.0 <= p2 <= 1.0:
        raise ValueError(f"p2 must be in [0,1], got {p2!r}")
    if not isclose(p1 + p2, 1.0, rel_tol=0.0, abs_tol=_ABS_TOL):
        raise ValueError(f"p1 + p2 must equal 1 (got {p1 + p2:.8f})")
    return p1, 1.0 - p1


def _distribution(best_of: int, p1: float, p2: float | None):
    p1, _ = _canon(p1, p2)
    return derive_series_distribution(best_of, [p1] * best_of)


def bo1(p1: float, p2: float | None = None) -> dict[str, float]:
    distribution = _distribution(1, p1, p2)
    return {"t1": distribution.team_a_win, "t2": distribution.team_b_win}


def bo2(p1: float, p2: float | None = None) -> dict[str, float]:
    scores = _distribution(2, p1, p2).score_probabilities
    return {
        "t1_2_0": scores["2-0"],
        "tie_1_1": scores["1-1"],
        "t2_0_2": scores["0-2"],
    }


def bo3(p1: float, p2: float | None = None) -> dict[str, float]:
    distribution = _distribution(3, p1, p2)
    scores = distribution.score_probabilities
    return {
        "t1_series": distribution.team_a_win,
        "t2_series": distribution.team_b_win,
        "t1_2_0": scores["2-0"],
        "t1_2_1": scores["2-1"],
        "t2_2_0": scores["0-2"],
        "t2_2_1": scores["1-2"],
        "t1_at_least_one": 1.0 - scores["0-2"],
        "t2_at_least_one": 1.0 - scores["2-0"],
        "exactly_3": distribution.total_maps_probabilities[3],
    }


def bo5(p1: float, p2: float | None = None) -> dict[str, float]:
    distribution = _distribution(5, p1, p2)
    scores = distribution.score_probabilities
    return {
        "t1_series": distribution.team_a_win,
        "t2_series": distribution.team_b_win,
        "t1_3_0": scores["3-0"],
        "t1_3_1": scores["3-1"],
        "t1_3_2": scores["3-2"],
        "t2_0_3": scores["0-3"],
        "t2_1_3": scores["1-3"],
        "t2_2_3": scores["2-3"],
        "t1_at_least_one": 1.0 - scores["0-3"],
        "t2_at_least_one": 1.0 - scores["3-0"],
        "exactly_3": distribution.total_maps_probabilities[3],
        "at_least_4": sum(
            probability
            for maps, probability in distribution.total_maps_probabilities.items()
            if maps >= MIN_LONG_SERIES_MAPS
        ),
        "exactly_5": distribution.total_maps_probabilities[5],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fair_odds(value: float) -> str:
    return "∞" if value <= 0 else f"{1.0 / value:.2f}"


def _line(team: str, handicap: str, meaning: str, probability: float) -> str:
    return (
        f"• {team} {handicap}{meaning}: "
        f"{_pct(probability)} → fair odds {_fair_odds(probability)}"
    )


def handicap_lines_bo3(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
    d = bo3(p1, p2)
    rows = (
        _line(t1, "+1.5", " (not swept)", 1.0 - d["t2_2_0"]),
        _line(t2, "+1.5", " (not swept)", 1.0 - d["t1_2_0"]),
        _line(t1, "-1.5", " (must sweep)", d["t1_2_0"]),
        _line(t2, "-1.5", " (must sweep)", d["t2_2_0"]),
    )
    return "**BO3 – Handicap Markets (+1.5 / -1.5)**\n" + "\n".join(rows)


def handicap_lines_bo5(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
    d = bo5(p1, p2)
    plus_15 = (
        _line(t1, "+1.5", "", d["t1_series"] + d["t2_2_3"]),
        _line(t2, "+1.5", "", d["t2_series"] + d["t1_3_2"]),
        _line(t1, "-1.5", "", d["t1_3_0"] + d["t1_3_1"]),
        _line(t2, "-1.5", "", d["t2_0_3"] + d["t2_1_3"]),
    )
    plus_25 = (
        _line(t1, "+2.5", " (not swept)", 1.0 - d["t2_0_3"]),
        _line(t2, "+2.5", " (not swept)", 1.0 - d["t1_3_0"]),
        _line(t1, "-2.5", " (3-0 sweep)", d["t1_3_0"]),
        _line(t2, "-2.5", " (3-0 sweep)", d["t2_0_3"]),
    )
    return (
        "**BO5 – Handicap Markets (+1.5 / -1.5)**\n"
        + "\n".join(plus_15)
        + "\n\n**BO5 – Handicap Markets (+2.5 / -2.5)**\n"
        + "\n".join(plus_25)
    )
