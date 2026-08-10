import json
from pathlib import Path

import pytest
from oracle_bets_core.league_selection import selected_leagues
from oracle_bets_core.league_taxonomy import add_league_taxonomy_columns
from oracle_bets_core.pd import pd

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


def test_product_profiles_separate_research_from_daily_serving():
    product = json.loads((ROOT / "config/product/product.json").read_text())
    selection = json.loads(
        (ROOT / "config/lol/data_ingestion/considered_leagues.json").read_text()
    )

    assert product["leagues"]["training_profile"] == "research_all_supported"
    assert product["leagues"]["prediction_profile"] == "tier1_plus_erls"
    assert product["leagues"]["prediction_profile"] == selection["active_profile"]
    assert set(product["leagues"]["actionable_exclusions"]) == {"CBLOL", "LCP"}
    assert set(selected_leagues(product["leagues"]["prediction_profile"])) < set(
        selected_leagues(product["leagues"]["training_profile"])
    )


def test_actionable_leagues_exclude_shadow_only_competitions():
    from oracle_bets_core.league_selection import actionable_leagues

    assert "LCP" not in actionable_leagues()
    assert "CBLOL" not in actionable_leagues()
    assert {"LCK", "LPL", "LEC", "LCS"} <= set(actionable_leagues())


def test_active_profile_has_taxonomy_entries():
    config = json.loads(
        (ROOT / "config/lol/data_ingestion/considered_leagues.json").read_text()
    )
    taxonomy = json.loads(
        (ROOT / "config/lol/data_ingestion/league_taxonomy.json").read_text()
    )
    known = set(taxonomy["leagues"])
    configured = {
        league
        for profile_leagues in config["profiles"].values()
        for league in profile_leagues
    }

    assert configured <= known


def test_league_taxonomy_is_metadata_only():
    taxonomy = json.loads(
        (ROOT / "config/lol/data_ingestion/league_taxonomy.json").read_text()
    )
    values = json.dumps(taxonomy)

    assert set(taxonomy["defaults"]) == {"region", "tier", "strength_pool"}
    assert all(
        set(entry) == {"region", "tier", "strength_pool"}
        for entry in taxonomy["leagues"].values()
    )
    assert "prior_settings" not in taxonomy
    assert "strength" + "_prior" not in values


def test_league_taxonomy_uses_single_macro_strength_key():
    taxonomy = json.loads(
        (ROOT / "config/lol/data_ingestion/league_taxonomy.json").read_text()
    )
    values = json.dumps(taxonomy)

    assert "strength_pool" in taxonomy["defaults"]
    assert "league_group" not in values
    assert "strength_group" not in values
    assert "macro_group" not in values


def test_league_taxonomy_attachment_rejects_unknown_leagues_by_default():
    df = pd.DataFrame({"league": ["LCK", "Definitely Unknown League"]})

    with pytest.raises(ValueError, match="League taxonomy missing entries"):
        add_league_taxonomy_columns(df)


def test_league_taxonomy_attachment_can_use_defaults_explicitly():
    df = pd.DataFrame({"league": ["Definitely Unknown League"]})

    out = add_league_taxonomy_columns(df, strict=False)

    assert out.loc[0, "league_region"] == "Unknown"
    assert out.loc[0, "league_tier"] == "minor"
    assert out.loc[0, "strength_pool"] == "minor"


def test_removed_league_prior_columns_are_not_configured():
    forbidden = {
        "league" + "_strength_prior",
        "league_elo" + "_prior_win_likelihood",
    }
    config_paths = (ROOT / "config/lol/training").glob("*.json")

    for path in config_paths:
        values = json.dumps(json.loads(path.read_text()))
        assert not any(column in values for column in forbidden), path


def test_removed_league_prior_code_paths_are_gone():
    forbidden = (
        "league" + "_strength_prior",
        "league_elo" + "_prior_win_likelihood",
        "league_strength" + "_priors.json",
    )
    search_roots = [ROOT / "packages", ROOT / "config", ROOT / "docs"]

    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                text = path.read_text(errors="ignore")
                assert not any(token in text for token in forbidden), path
