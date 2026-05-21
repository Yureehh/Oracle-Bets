from oracle_bets_discord.registry import default_registry


def test_default_registry_contains_lol_module():
    registry = default_registry()

    assert registry.get("lol-bets").id == "lol-bets"
