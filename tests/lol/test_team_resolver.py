"""Tests for external→Oracle's Elixir team name resolution."""

import json

import pytest
from lol_bets.inference import team_resolver
from lol_bets.inference.team_resolver import (
    ResolvedTeam,
    TeamResolutionError,
    canonical_team_name,
    resolve_team_name,
    team_name_variants,
)

KNOWN = [
    "T1",
    "Gen.G",
    "Bilibili Gaming",
    "LNG Esports",
    "Karmine Corp Blue",
    "Team WE",
]


@pytest.fixture(autouse=True)
def _fresh_alias_cache():
    team_resolver.clear_alias_cache()
    yield
    team_resolver.clear_alias_cache()


def test_exact_match_is_case_insensitive():
    out = resolve_team_name("gen.g", KNOWN)

    assert out.ok
    assert out.resolved_name == "Gen.G"
    assert out.method == "exact"


def test_normalized_match_strips_org_suffixes():
    out = resolve_team_name("LNG", KNOWN)

    assert out.ok
    assert out.resolved_name == "LNG Esports"
    assert out.method == "normalized"


def test_normalized_match_ignores_punctuation():
    out = resolve_team_name("GEN G", KNOWN)

    assert out.ok
    assert out.resolved_name == "Gen.G"


def test_unknown_team_fails_with_fuzzy_suggestions_only():
    out = resolve_team_name("Karmine Corp Bleu", KNOWN)

    assert not out.ok
    assert out.resolved_name is None
    assert "Karmine Corp Blue" in out.suggestions


def test_fuzzy_matches_are_never_auto_accepted():
    out = resolve_team_name("Bilibili Gaming Junior", KNOWN)

    assert not out.ok


def test_empty_name_fails_cleanly():
    out = resolve_team_name("   ", KNOWN)

    assert not out.ok
    assert out.suggestions == ()


def test_alias_config_resolves_external_names(tmp_path, monkeypatch):
    alias_path = tmp_path / "team_aliases.json"
    alias_path.write_text(
        json.dumps(
            {
                "suffixes_to_strip": ["esports"],
                "external_aliases": {"BLG": "Bilibili Gaming"},
            }
        )
    )
    monkeypatch.setattr(team_resolver, "TEAM_ALIASES", alias_path)
    team_resolver.clear_alias_cache()

    out = resolve_team_name("blg", KNOWN)

    assert out.ok
    assert out.resolved_name == "Bilibili Gaming"
    assert out.method == "alias"


def test_production_alias_resolves_ag_al_to_anyones_legend():
    out = resolve_team_name("AG.AL", ["Anyone's Legend"])

    assert out.ok
    assert out.resolved_name == "Anyone's Legend"
    assert out.method == "alias"


def test_alias_variants_include_provider_and_canonical_names():
    assert set(team_name_variants("AG.AL")) >= {"AG.AL", "Anyone's Legend"}
    assert canonical_team_name("AG.AL") == "Anyone's Legend"


def test_production_alias_resolves_nongshim_provider_name():
    out = resolve_team_name("Nongshim Red Force", ["Nongshim RedForce"])

    assert out.ok
    assert out.resolved_name == "Nongshim RedForce"
    assert out.method == "alias"


def test_stopword_only_names_require_full_token_set():
    # "Team WE" is made of generic tokens; it must still resolve exactly.
    out = resolve_team_name("team we", KNOWN)

    assert out.ok
    assert out.resolved_name == "Team WE"


def test_resolution_error_message_includes_suggestions():
    resolved = ResolvedTeam(
        query="Karmine Corp Bleu",
        resolved_name=None,
        method=None,
        suggestions=("Karmine Corp Blue",),
    )
    error = TeamResolutionError(resolved)

    assert "Karmine Corp Bleu" in str(error)
    assert "Karmine Corp Blue" in str(error)
    assert "team_aliases.json" in str(error)
