from oracle_bets_core.betting import price_over_under
from oracle_bets_discord.bot import _parse_lol_options
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
