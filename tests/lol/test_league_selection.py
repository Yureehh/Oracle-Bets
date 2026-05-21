import json
from pathlib import Path

from oracle_bets_core.league_selection import selected_leagues

ROOT = Path(__file__).resolve().parents[2]


def test_active_league_profile_drives_selected_leagues():
    config = json.loads(
        (ROOT / "config/lol/data_ingestion/considered_leagues.json").read_text()
    )

    assert set(config) == {"active_profile", "profiles"}
    assert selected_leagues() == config["profiles"][config["active_profile"]]
    assert {"LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"} <= set(selected_leagues())
    assert {"LTA", "LTA N", "LTA S"} <= set(selected_leagues())
    assert {"LFL", "LES", "PRM", "TCL", "PCS", "VCS", "LJL"} <= set(selected_leagues())
    assert "LCKC" not in selected_leagues()


def test_active_profile_has_taxonomy_entries():
    taxonomy = json.loads(
        (ROOT / "config/lol/data_ingestion/league_taxonomy.json").read_text()
    )
    known = set(taxonomy["leagues"])

    assert set(selected_leagues()) <= known
