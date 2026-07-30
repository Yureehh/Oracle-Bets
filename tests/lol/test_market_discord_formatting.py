import pytest
from oracle_bets_core.betting import price_over_under
from oracle_bets_discord.predictions.best_ofs import (
    bo3,
    bo5,
    handicap_lines_bo3,
    handicap_lines_bo5,
)
from oracle_bets_discord.predictions.lol import (
    format_prop_market_output,
    format_winner_market_output,
)

KILLS_LINE = 26.5
OVER_ODDS = 1.85
DEFAULT_BANKROLL = 500.0
DEFAULT_KELLY_FRACTION = 0.25
CUSTOM_BANKROLL = 2000.0
CUSTOM_KELLY_FRACTION = 0.5
DISCORD_MESSAGE_LIMIT = 2000


def test_prop_output_explains_line_pricing_and_confidence():
    signal = price_over_under(mean=27.4, line=26.5, sigma=2.5, over_odds=1.85)
    output = format_prop_market_output(
        blue_team_name="Team WE",
        red_team_name="LNG Esports",
        gamelength=31.8,
        total_kills=27.4,
        total_towers=12.9,
        line_signals={"Kills": signal},
        calibration_sources={"Kills": "prop calibrator (empirical_global)"},
        warnings=[],
    )

    assert "Kills: **27.4**" in output
    assert "fair" in output
    assert "Edge" in output
    assert "Half-Kelly" in output
    assert "Probability source" in output
    assert "Confidence" in output
    assert "Paper research only" in output


def test_winner_output_is_compact_and_betting_focused():
    output = format_winner_market_output(
        blue_team_name="Dplus KIA",
        red_team_name="T1",
        match_type="bo3",
        blue_win=0.306,
        red_win=0.694,
        probability_source="calibrated model (isotonic)",
        warnings=[],
        context="- Context: side ignored | first pick unknown",
        blue_range=(0.27, 0.35),
        red_range=(0.65, 0.73),
        uncertainty_confidence=0.9,
        drivers=["Team rating strength pushed the model toward T1"],
    )

    assert "LoL Markets" in output
    assert "Market" in output
    assert "PRICE NEEDED" in output
    assert "calibrated model (isotonic)" in output
    assert "Game 1 Winner" in output
    assert "Series Winner" in output
    assert "series markets are derived" in output
    assert "Probability range (90% calibration uncertainty)" in output
    assert "Main model drivers (descriptive, not causal)" in output
    assert "Team rating strength pushed the model toward T1" in output
    assert "Bankroll:" not in output
    assert "Found Rosters" not in output
    assert "Meaning" not in output


def test_winner_output_prices_polymarket_game_winner_with_quarter_kelly():
    output = format_winner_market_output(
        blue_team_name="Solary",
        red_team_name="Karmine Corp Blue",
        match_type="bo5",
        blue_win=0.641,
        red_win=0.359,
        probability_source="raw model (selected by calibration)",
        warnings=["Side selection is ignored for this command."],
        context="- Context: side ignored | first pick unknown",
        price_quotes=["g3: SLY=74c,KCB=27c"],
    )

    assert "Bankroll: €500 | Kelly: 1/4 | No stake cap" in output
    assert "Game 3 Winner" in output
    assert "Karmine Cor" in output
    assert "27c" in output
    assert "+33.0%" in output
    assert "€ 15.24" in output
    assert "VALUE" in output
    assert "Solary" in output
    assert "74c" in output
    assert "PASS" in output
    assert "Warning: Side selection is ignored for this command." in output


def test_winner_output_maps_total_games_and_sorts_priced_edges():
    output = format_winner_market_output(
        blue_team_name="Solary",
        red_team_name="Karmine Corp Blue",
        match_type="bo5",
        blue_win=0.641,
        red_win=0.359,
        probability_source="raw model",
        warnings=[],
        context="- Context: side ignored | first pick unknown",
        price_quotes=[
            "g3: Solary=74c,Karmine Corp Blue=27c",
            "tg3.5: Over=61c,Under=40c",
        ],
    )

    table = output.split("```text", 1)[1].split("```", 1)[0]
    kcb_index = table.index("Karmine Cor")
    solary_index = table.index("Solary")
    assert kcb_index < solary_index
    assert "Over 3.5 Maps" in output
    assert "61c" in output
    assert "Under 3.5 Maps" in output
    assert "40c" in output


def test_winner_output_maps_numeric_price_pair_positions():
    output = format_winner_market_output(
        blue_team_name="Solary",
        red_team_name="Karmine Corp Blue",
        match_type="bo5",
        blue_win=0.641,
        red_win=0.359,
        probability_source="raw model",
        warnings=[],
        context="- Context: side ignored | first pick unknown",
        price_quotes=["g3: 1=74,2=27", "tg3.5: 1=61,2=40"],
    )

    assert "Game 3 Winner" in output
    assert "Solary" in output
    assert "74c" in output
    assert "Karmine Cor" in output
    assert "27c" in output
    assert "Over 3.5 Maps" in output
    assert "61c" in output
    assert "Under 3.5 Maps" in output
    assert "40c" in output


def test_winner_output_ignores_unsupported_polymarket_markets():
    output = format_winner_market_output(
        blue_team_name="Solary",
        red_team_name="Karmine Corp Blue",
        match_type="bo5",
        blue_win=0.641,
        red_win=0.359,
        probability_source="raw model",
        warnings=[],
        context="- Context: side ignored | first pick unknown",
        price_quotes=["First Blood Game 1: Solary=50c,Karmine Corp Blue=50c"],
    )

    assert "Unsupported prices ignored: First Blood Game 1" in output
    assert "First Blood" not in output.split("```text", 1)[1].split("```", 1)[0]


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
    """At 50/50, +2.5 = 87.5%; +1.5 = series win or 2-3 loss = 68.75%."""
    d = bo5(0.5)
    t1_plus25 = 1.0 - d["t2_0_3"]  # 1 - 0.125 = 0.875
    t1_plus15 = d["t1_series"] + d["t2_2_3"]
    assert t1_plus25 == pytest.approx(0.875)
    assert t1_plus15 == pytest.approx(0.6875)


def test_bo5_handicap_output_contains_both_lines():
    out = handicap_lines_bo5("T1", 0.5, "G2")
    assert "+1.5" in out
    assert "+2.5" in out
    assert "T1 +1.5" in out
    assert "G2 +2.5" in out


def test_staking_config_defaults_match_legacy_display(monkeypatch):
    from oracle_bets_discord.predictions.lol import get_staking_config

    monkeypatch.delenv("ORACLE_BETS_BANKROLL", raising=False)
    monkeypatch.delenv("ORACLE_BETS_KELLY_FRACTION", raising=False)
    monkeypatch.delenv("ORACLE_BETS_STAKE_CAP", raising=False)

    staking = get_staking_config()

    assert staking.bankroll == DEFAULT_BANKROLL
    assert staking.kelly_fraction == DEFAULT_KELLY_FRACTION
    assert staking.kelly_label == "1/4"
    assert staking.cap_label == "No stake cap"


def test_staking_config_reads_environment_overrides(monkeypatch):
    from oracle_bets_discord.predictions.lol import get_staking_config

    monkeypatch.setenv("ORACLE_BETS_BANKROLL", "2000")
    monkeypatch.setenv("ORACLE_BETS_KELLY_FRACTION", "0.5")
    monkeypatch.setenv("ORACLE_BETS_STAKE_CAP", "100")

    staking = get_staking_config()

    assert staking.bankroll == CUSTOM_BANKROLL
    assert staking.kelly_fraction == CUSTOM_KELLY_FRACTION
    assert staking.kelly_label == "1/2"
    assert staking.cap_label == "Stake cap: €100"


def test_staking_config_rejects_invalid_environment_values(monkeypatch):
    from oracle_bets_discord.predictions.lol import get_staking_config

    monkeypatch.setenv("ORACLE_BETS_BANKROLL", "-5")
    monkeypatch.setenv("ORACLE_BETS_KELLY_FRACTION", "abc")
    monkeypatch.setenv("ORACLE_BETS_STAKE_CAP", "0")

    staking = get_staking_config()

    assert staking.bankroll == DEFAULT_BANKROLL
    assert staking.kelly_fraction == DEFAULT_KELLY_FRACTION
    assert staking.stake_cap is None


def test_side_note_does_not_lower_confidence_tier():
    output = format_winner_market_output(
        blue_team_name="T1",
        red_team_name="Gen.G",
        match_type="bo3",
        blue_win=0.6,
        red_win=0.4,
        probability_source="calibrated",
        warnings=[],
        context="- Context: side ignored | first pick unknown",
        notes=["Side selection is ignored for this command."],
    )

    assert "Confidence: High" in output
    assert "Side selection is ignored" in output


def test_oversized_single_line_is_truncated_in_schedule_chunks():
    from oracle_bets_discord.predictions.lol import _split_schedule_block

    giant_line = "x" * 3000
    block = "header\n" + giant_line + "\nfooter"

    chunks = _split_schedule_block(block, DISCORD_MESSAGE_LIMIT)

    assert all(len(chunk) < DISCORD_MESSAGE_LIMIT for chunk in chunks)
