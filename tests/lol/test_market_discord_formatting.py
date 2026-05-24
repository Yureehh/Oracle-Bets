import pytest
from oracle_bets_core.betting import price_over_under
from oracle_bets_core.interfaces import ArtifactCheck, ArtifactHealth
from oracle_bets_discord.bot import (
    _parse_bet_options,
    _parse_lol_options,
    _raise_for_blocking_artifact_failures,
)
from oracle_bets_discord.predictions.best_ofs import (
    bo3,
    bo5,
    handicap_lines_bo3,
    handicap_lines_bo5,
)
from oracle_bets_discord.predictions.lol import format_prop_market_output

KILLS_LINE = 26.5
OVER_ODDS = 1.85


def test_lol_option_parser_reads_prop_lines_and_odds():
    parsed = _parse_lol_options(
        '--kills-line 26.5 --kills-over-odds 1.85 --side Blue --first-pick "Team WE"'
    )

    assert parsed["kills_line"] == KILLS_LINE
    assert parsed["kills_over_odds"] == OVER_ODDS
    assert parsed["side"] == "Blue"
    assert parsed["first_pick_team_name"] == "Team WE"


def test_prop_output_explains_line_pricing_and_confidence():
    signal = price_over_under(mean=27.4, line=26.5, sigma=2.5, over_odds=1.85)
    output = format_prop_market_output(
        blue_team_name="Team WE",
        red_team_name="LNG Esports",
        gamelength=31.8,
        total_kills=27.4,
        total_towers=12.9,
        line_signals={"Kills": signal},
        warnings=[],
    )

    assert "Expected total kills" in output
    assert "Fair odds" in output
    assert "Edge" in output
    assert "Half-Kelly" in output
    assert "Meaning" in output
    assert "Confidence" in output
    assert "!bet record" in output


def test_bet_option_parser_reads_ledger_record_flags():
    parsed = _parse_bet_options(
        '--event "Team WE vs LNG" --market kills --selection over '
        "--line 26.5 --odds 1.85 --stake 25 --book polymarket --prob 0.557 "
        "--edge 0.031 --map 1 --league LPL --smoke"
    )

    assert parsed["event_name"] == "Team WE vs LNG"
    assert parsed["market_type"] == "kills"
    assert parsed["selection"] == "over"
    assert parsed["odds_decimal"] == OVER_ODDS
    assert parsed["map_number"] == 1
    assert parsed["is_smoke"] is True


class _Module:
    id = "lol-bets"

    def __init__(self, checks):
        self._checks = tuple(checks)

    def artifact_health(self):
        return ArtifactHealth(module_id=self.id, checks=self._checks)


def test_discord_startup_allows_missing_prop_artifacts():
    _raise_for_blocking_artifact_failures(
        _Module(
            [
                ArtifactCheck("outcome model", "ok.pkl", True),
                ArtifactCheck("total kills model", "missing.pkl", False, "missing"),
            ]
        )
    )


def test_discord_startup_blocks_missing_outcome_artifacts():
    with pytest.raises(RuntimeError, match="outcome model"):
        _raise_for_blocking_artifact_failures(
            _Module([ArtifactCheck("outcome model", "missing.pkl", False, "missing")])
        )


# ---------- handicap math tests ---------- #


def test_bo3_handicap_50_50():
    """At 50/50, both teams' +1.5 must be 75% and -1.5 must be 25%."""
    d = bo3(0.5)
    t1_plus = 1.0 - d["t2_2_0"]
    t2_plus = 1.0 - d["t1_2_0"]
    assert t1_plus == pytest.approx(0.75)
    assert t2_plus == pytest.approx(0.75)
    assert d["t1_2_0"] == pytest.approx(0.25)  # -1.5 for t1
    assert d["t2_2_0"] == pytest.approx(0.25)  # -1.5 for t2


def test_bo3_handicap_60_40():
    """At 60/40, favourite +1.5 = 84%, underdog +1.5 = 64%."""
    d = bo3(0.6)
    t1_plus = 1.0 - d["t2_2_0"]  # 1 - 0.16 = 0.84
    t2_plus = 1.0 - d["t1_2_0"]  # 1 - 0.36 = 0.64
    assert t1_plus == pytest.approx(0.84)
    assert t2_plus == pytest.approx(0.64)


def test_bo3_handicap_output_contains_fair_odds():
    out = handicap_lines_bo3("Vitality", 0.5, "KOI")
    assert "75.00%" in out
    assert "1.33" in out
    assert "Vitality +1.5" in out
    assert "KOI +1.5" in out


def test_bo3_handicap_output_plus_minus_sum_to_one():
    """
    For any p1, t1 +1.5 + t1 -1.5 != 1 (they are independent), but
    t1_plus + t2_minus == 1 because t1_plus = 1 - t2_sweep and t2_minus = t2_sweep.
    """
    d = bo3(0.65)
    t1_plus = 1.0 - d["t2_2_0"]
    t2_minus = d["t2_2_0"]
    assert t1_plus + t2_minus == pytest.approx(1.0)


def test_bo5_handicap_50_50():
    """At 50/50, +2.5 = 87.5% (1 - 0.5^3), +1.5 = 81.25%."""
    d = bo5(0.5)
    t1_plus25 = 1.0 - d["t2_0_3"]  # 1 - 0.125 = 0.875
    t1_plus15 = d["t1_series"] + d["t2_2_3"]
    assert t1_plus25 == pytest.approx(0.875)
    assert t1_plus15 == pytest.approx(0.8125)


def test_bo5_handicap_output_contains_both_lines():
    out = handicap_lines_bo5("T1", 0.5, "G2")
    assert "+1.5" in out
    assert "+2.5" in out
    assert "T1 +1.5" in out
    assert "G2 +2.5" in out
