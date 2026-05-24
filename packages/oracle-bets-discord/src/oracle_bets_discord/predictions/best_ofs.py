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


# ---------- handicap helpers ---------- #


def _fair_odds(p: float) -> str:
    """Return fair decimal odds string for probability p."""
    if p <= 0:
        return "∞"
    return f"{1.0 / p:.2f}"


def handicap_lines_bo3(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
    """Return formatted BO3 handicap market block (+1.5 / -1.5)."""
    p1, p2 = _canon(p1, p2)
    d = bo3(p1, p2)
    t1_plus = 1.0 - d["t2_2_0"]  # t1 doesn't get swept: t1_2_0 + t1_2_1 + t2_2_1
    t2_plus = 1.0 - d["t1_2_0"]  # t2 doesn't get swept
    t1_minus = d["t1_2_0"]  # t1 sweeps
    t2_minus = d["t2_2_0"]  # t2 sweeps
    return (
        f"**BO3 – Handicap Markets (+1.5 / -1.5)**\n"
        f"• {t1} +1.5 (not swept):  {_pct(t1_plus)} → fair odds {_fair_odds(t1_plus)}\n"
        f"• {t2} +1.5 (not swept):  {_pct(t2_plus)} → fair odds {_fair_odds(t2_plus)}\n"
        f"• {t1} -1.5 (must sweep): {_pct(t1_minus)} → fair odds {_fair_odds(t1_minus)}\n"
        f"• {t2} -1.5 (must sweep): {_pct(t2_minus)} → fair odds {_fair_odds(t2_minus)}\n"
        f"_Use `!edge <market_odds> <prob%>` to check edge, "
        f'then `!bet record --market handicap --selection "{t1} +1.5"`_'
    )


def handicap_lines_bo5(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
    """Return formatted BO5 handicap market block (+1.5, -1.5, +2.5, -2.5)."""
    p1, p2 = _canon(p1, p2)
    d = bo5(p1, p2)
    # +1.5: wins series OR loses 2-3
    t1_plus15 = d["t1_series"] + d["t2_2_3"]
    t2_plus15 = d["t2_series"] + d["t1_3_2"]
    # -1.5: wins 3-0 or 3-1
    t1_minus15 = d["t1_3_0"] + d["t1_3_1"]
    t2_minus15 = d["t2_0_3"] + d["t2_1_3"]
    # +2.5: doesn't get swept 0-3
    t1_plus25 = 1.0 - d["t2_0_3"]
    t2_plus25 = 1.0 - d["t1_3_0"]
    # -2.5: wins 3-0
    t1_minus25 = d["t1_3_0"]
    t2_minus25 = d["t2_0_3"]
    return (
        f"**BO5 – Handicap Markets (+1.5 / -1.5)**\n"
        f"• {t1} +1.5: {_pct(t1_plus15)} → fair odds {_fair_odds(t1_plus15)}\n"
        f"• {t2} +1.5: {_pct(t2_plus15)} → fair odds {_fair_odds(t2_plus15)}\n"
        f"• {t1} -1.5: {_pct(t1_minus15)} → fair odds {_fair_odds(t1_minus15)}\n"
        f"• {t2} -1.5: {_pct(t2_minus15)} → fair odds {_fair_odds(t2_minus15)}\n\n"
        f"**BO5 – Handicap Markets (+2.5 / -2.5)**\n"
        f"• {t1} +2.5 (not swept): {_pct(t1_plus25)} → fair odds {_fair_odds(t1_plus25)}\n"
        f"• {t2} +2.5 (not swept): {_pct(t2_plus25)} → fair odds {_fair_odds(t2_plus25)}\n"
        f"• {t1} -2.5 (3-0 sweep): {_pct(t1_minus25)} → fair odds {_fair_odds(t1_minus25)}\n"
        f"• {t2} -2.5 (3-0 sweep): {_pct(t2_minus25)} → fair odds {_fair_odds(t2_minus25)}\n"
        f"_Use `!edge <market_odds> <prob%>` to check edge, "
        f'then `!bet record --market handicap --selection "{t1} +1.5"`_'
    )


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
        series_block = (
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
        return series_block + "\n\n" + handicap_lines_bo3(t1, p1, t2, p2)

    @staticmethod
    def best_of_five(t1: str, p1: float, t2: str, p2: float | None = None) -> str:
        d = bo5(p1, p2)
        series_block = (
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
        return series_block + "\n\n" + handicap_lines_bo5(t1, p1, t2, p2)
