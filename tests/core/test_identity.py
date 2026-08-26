from oracle_bets_core.evidence.identity import EntityType, canonical_identity_id


def test_canonical_identity_id_is_stable_and_provider_scoped():
    first = canonical_identity_id("lol", EntityType.TEAM, "oracles_elixir", "oe:t1")
    repeated = canonical_identity_id("lol", EntityType.TEAM, "oracles_elixir", "oe:t1")
    other_provider = canonical_identity_id(
        "lol", EntityType.TEAM, "pandascore", "oe:t1"
    )

    assert first == repeated
    assert first != other_provider
    assert first.startswith("lol:team:")
