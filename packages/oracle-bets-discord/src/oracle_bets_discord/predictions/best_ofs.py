# oracle_bets_discord/predictions/best_ofs.py
"""
Best-of series probabilities + Discord formatters.

- Keeps original math (bo1/bo2/bo3/bo5 dict helpers).
- Adds class BestOfs.* methods that return preformatted strings:
    * best_of_one(name1, p1, name2, p2)
    * best_of_two(name1, p1, name2, p2)
    * best_of_three(name1, p1, name2, p2)
    * best_of_five(name1, p1, name2, p2)
"""

from __future__ import annotations

from math import isclose

_ABS_TOL = 1e-6


def _canon(p1: float, p2: float | None) -> tuple[float, float]:
    if not (0.0 <= p1 <= 1.0):
        msg = f"p1 must be in [0,1], got {p1!r}"
        raise ValueError(msg)
    p2 = (1.0 - p1) if p2 is None else p2
    if not (0.0 <= p2 <= 1.0):
        msg = f"p2 must be in [0,1], got {p2!r}"
        raise ValueError(msg)
    if not isclose(p1 + p2, 1.0, rel_tol=0.0, abs_tol=_ABS_TOL):
        msg = f"p1 + p2 must equal 1 (got {p1 + p2:.8f})"
        raise ValueError(msg)
    return p1, 1.0 - p1  # canonicalize to avoid drift


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ---------- original dict-returning helpers (kept for reuse) ---------- #


def bo1(p1: float, p2: float | None = None) -> dict[str, float]:
    p1, p2 = _canon(p1, p2)
    return {"t1": p1, "t2": p2}


def bo2(p1: float, p2: float | None = None) -> dict[str, float]:
    p1, p2 = _canon(p1, p2)
    t1_2_0 = p1 * p1
    t2_0_2 = p2 * p2
    tie_1_1 = 2.0 * p1 * p2
    assert isclose(t1_2_0 + tie_1_1 + t2_0_2, 1.0, abs_tol=1e-9)
    return {"t1_2_0": t1_2_0, "tie_1_1": tie_1_1, "t2_0_2": t2_0_2}


def bo3(p1: float, p2: float | None = None) -> dict[str, float]:
    p1, p2 = _canon(p1, p2)
    t1_2_0 = p1**2
    t1_2_1 = 2.0 * (p1**2) * p2
    t2_2_0 = p2**2
    t2_2_1 = 2.0 * (p2**2) * p1
    t1_series = t1_2_0 + t1_2_1
    t2_series = t2_2_0 + t2_2_1
    exactly_3 = 2.0 * p1 * p2
    t1_at_least_one = 1.0 - (p2**2)
    t2_at_least_one = 1.0 - (p1**2)
    assert isclose(t1_series + t2_series, 1.0, abs_tol=1e-9)
    return {
        "t1_series": t1_series,
        "t2_series": t2_series,
        "t1_2_0": t1_2_0,
        "t1_2_1": t1_2_1,
        "t2_2_0": t2_2_0,
        "t2_2_1": t2_2_1,
        "t1_at_least_one": t1_at_least_one,
        "t2_at_least_one": t2_at_least_one,
        "exactly_3": exactly_3,
    }


def bo5(p1: float, p2: float | None = None) -> dict[str, float]:
    p1, p2 = _canon(p1, p2)
    t1_3_0 = p1**3
    t1_3_1 = 3.0 * (p1**3) * p2
    t1_3_2 = 6.0 * (p1**3) * (p2**2)
    t2_0_3 = p2**3
    t2_1_3 = 3.0 * (p2**3) * p1
    t2_2_3 = 6.0 * (p2**3) * (p1**2)
    t1_series = t1_3_0 + t1_3_1 + t1_3_2
    t2_series = t2_0_3 + t2_1_3 + t2_2_3
    exactly_3 = t1_3_0 + t2_0_3
    at_least_4 = 1.0 - exactly_3
    exactly_5 = 6.0 * (p1**2) * (p2**2)
    t1_at_least_one = 1.0 - (p2**3)
    t2_at_least_one = 1.0 - (p1**3)
    assert isclose(t1_series + t2_series, 1.0, abs_tol=1e-9)
    return {
        "t1_series": t1_series,
        "t2_series": t2_series,
        "t1_3_0": t1_3_0,
        "t1_3_1": t1_3_1,
        "t1_3_2": t1_3_2,
        "t2_0_3": t2_0_3,
        "t2_1_3": t2_1_3,
        "t2_2_3": t2_2_3,
        "t1_at_least_one": t1_at_least_one,
        "t2_at_least_one": t2_at_least_one,
        "exactly_3": exactly_3,
        "at_least_4": at_least_4,
        "exactly_5": exactly_5,
    }


# ---------- Discord-facing class (what your bot expects) ---------- #


class BestOfs:
    @staticmethod
    def best_of_one(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
        d = bo1(p1, p2)
        return (
            f"**BO1 – Single Game Win Probabilities**\n"
            f"• {t1}: {_pct(d['t1'])}\n"
            f"• {t2}: {_pct(d['t2'])}"
        )

    @staticmethod
    def best_of_two(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
        d = bo2(p1, p2)
        return (
            f"**BO2 – Series Outcomes**\n"
            f"• {t1} 2–0: {_pct(d['t1_2_0'])}\n"
            f"• 1–1 Tie: {_pct(d['tie_1_1'])}\n"
            f"• {t2} 2–0: {_pct(d['t2_0_2'])}"
        )

    @staticmethod
    def best_of_three(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
        d = bo3(p1, p2)
        return (
            f"**BO3 – Series Win Probabilities**\n"
            f"• {t1} wins series: {_pct(d['t1_series'])}\n"
            f"• {t2} wins series: {_pct(d['t2_series'])}\n\n"
            f"**BO3 – Scorelines**\n"
            f"• {t1} 2–0: {_pct(d['t1_2_0'])}\n"
            f"• {t1} 2–1: {_pct(d['t1_2_1'])}\n"
            f"• {t2} 2–0: {_pct(d['t2_2_0'])}\n"
            f"• {t2} 2–1: {_pct(d['t2_2_1'])}\n"
            f"• Exactly 3 games: {_pct(d['exactly_3'])}"
        )

    @staticmethod
    def best_of_five(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
        d = bo5(p1, p2)
        return (
            f"**BO5 – Series Win Probabilities**\n"
            f"• {t1} wins series: {_pct(d['t1_series'])}\n"
            f"• {t2} wins series: {_pct(d['t2_series'])}\n\n"
            f"**BO5 – Scorelines**\n"
            f"• {t1} 3–0: {_pct(d['t1_3_0'])}\n"
            f"• {t1} 3–1: {_pct(d['t1_3_1'])}\n"
            f"• {t1} 3–2: {_pct(d['t1_3_2'])}\n"
            f"• {t2} 3–0: {_pct(d['t2_0_3'])}\n"
            f"• {t2} 3–1: {_pct(d['t2_1_3'])}\n"
            f"• {t2} 3–2: {_pct(d['t2_2_3'])}\n\n"
            f"**BO5 – Length**\n"
            f"• Exactly 3 games: {_pct(d['exactly_3'])}\n"
            f"• At least 4 games: {_pct(d['at_least_4'])}\n"
            f"• Exactly 5 games: {_pct(d['exactly_5'])}"
        )
