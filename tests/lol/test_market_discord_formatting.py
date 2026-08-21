from oracle_bets_core.betting import price_over_under
from oracle_bets_discord.predictions.lol import (
    format_prop_market_output,
    format_winner_market_output,
)

KILLS_LINE = 26.5
OVER_ODDS = 1.85
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
    assert "Fair" in output
    assert "calibrated model (isotonic)" in output
    assert "Series Winner" in output
    assert "independent prematch series-winner model" in output
    assert "Game 1 Winner" not in output
    assert "69.4%" in output
    assert "78.0%" not in output
    assert "Probability range (90% calibration uncertainty)" in output
    assert "Main model drivers (descriptive, not causal)" in output
    assert "Team rating strength pushed the model toward T1" in output
    assert "Bankroll:" not in output
    assert "Found Rosters" not in output
    assert "Meaning" not in output


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
