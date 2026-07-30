from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from oracle_bets_core.evidence import EvidenceStore, EvidenceTable
from oracle_bets_core.evidence.identity import (
    CanonicalIdentity,
    EntityType,
    FixtureFacts,
    IdentityConflictError,
    IdentityRegistry,
    ProviderLink,
    canonical_identity_id,
    fixtures_agree,
    persist_identity,
)

NOW = datetime(2026, 7, 26, 8, 15, tzinfo=UTC)


def test_canonical_identity_id_is_stable_and_provider_scoped():
    first = canonical_identity_id("lol", EntityType.TEAM, "oracles_elixir", "oe:t1")
    repeated = canonical_identity_id("lol", EntityType.TEAM, "oracles_elixir", "oe:t1")
    other_provider = canonical_identity_id(
        "lol", EntityType.TEAM, "pandascore", "oe:t1"
    )

    assert first == repeated
    assert first != other_provider
    assert first.startswith("lol:team:")


def test_provider_links_are_versioned_and_expire():
    link = ProviderLink(
        link_id="link-1",
        identity_id="lol:team:t1",
        provider="pandascore",
        provider_entity_id="1",
        valid_from=NOW,
        valid_to=NOW + timedelta(days=1),
    )

    assert link.active_at(NOW)
    assert link.active_at(NOW + timedelta(hours=23))
    assert not link.active_at(NOW + timedelta(days=1))


def test_registry_rejects_overlapping_provider_link_to_another_identity():
    registry = IdentityRegistry()
    registry.add(
        CanonicalIdentity(
            identity_id="lol:team:t1",
            sport="lol",
            entity_type=EntityType.TEAM,
            canonical_name="T1",
            classification="major",
            created_at=NOW,
        )
    )
    registry.add(
        CanonicalIdentity(
            identity_id="lol:team:t1-academy",
            sport="lol",
            entity_type=EntityType.TEAM,
            canonical_name="T1 Esports Academy",
            classification="academy",
            created_at=NOW,
        )
    )
    registry.add_link(
        ProviderLink(
            link_id="link-1",
            identity_id="lol:team:t1",
            provider="pandascore",
            provider_entity_id="1",
            valid_from=NOW,
        )
    )

    with pytest.raises(IdentityConflictError, match="already linked"):
        registry.add_link(
            ProviderLink(
                link_id="link-2",
                identity_id="lol:team:t1-academy",
                provider="pandascore",
                provider_entity_id="1",
                valid_from=NOW,
            )
        )


def test_exact_name_lookup_keeps_academy_and_main_team_separate():
    registry = IdentityRegistry()
    main = CanonicalIdentity(
        identity_id="lol:team:kcorp",
        sport="lol",
        entity_type=EntityType.TEAM,
        canonical_name="Karmine Corp",
        classification="major",
        created_at=NOW,
    )
    academy = CanonicalIdentity(
        identity_id="lol:team:kcorp-blue",
        sport="lol",
        entity_type=EntityType.TEAM,
        canonical_name="Karmine Corp Blue",
        classification="academy",
        created_at=NOW,
    )
    registry.add(main)
    registry.add(academy)

    assert registry.find_exact_name("Karmine Corp Blue") == (academy,)
    assert registry.find_exact_name("Karmine Corp") == (main,)
    assert registry.find_exact_name("Karmine Corp", classification="academy") == ()


def test_duplicate_fixtures_require_all_identity_facts_to_agree():
    fixture = FixtureFacts(
        provider="pandascore",
        provider_fixture_id="42",
        competition_id="LCK",
        team_a_identity_id="team-a",
        team_b_identity_id="team-b",
        start_time=NOW,
        best_of=3,
    )

    assert fixtures_agree(fixture, fixture)
    assert not fixtures_agree(
        fixture,
        FixtureFacts(
            provider="pandascore",
            provider_fixture_id="42",
            competition_id="LCK",
            team_a_identity_id="team-a",
            team_b_identity_id="team-b",
            start_time=NOW + timedelta(minutes=5),
            best_of=3,
        ),
    )
    assert not fixtures_agree(
        fixture,
        FixtureFacts(
            provider="pandascore",
            provider_fixture_id="42",
            competition_id="LCK",
            team_a_identity_id="team-b",
            team_b_identity_id="team-a",
            start_time=NOW,
            best_of=3,
        ),
    )


def test_identity_records_require_utc_timestamps():
    with pytest.raises(ValueError, match="UTC"):
        CanonicalIdentity(
            identity_id="lol:team:t1",
            sport="lol",
            entity_type=EntityType.TEAM,
            canonical_name="T1",
            classification="major",
            created_at=datetime(2026, 7, 26, 8, 15),
        )


def test_identity_and_provider_link_persist_as_separate_evidence(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    identity = CanonicalIdentity(
        identity_id="lol:team:t1",
        sport="lol",
        entity_type=EntityType.TEAM,
        canonical_name="T1",
        classification="major",
        created_at=NOW,
    )
    link = ProviderLink(
        link_id="link-1",
        identity_id=identity.identity_id,
        provider="oracles_elixir",
        provider_entity_id="oe:t1",
        valid_from=NOW,
    )

    persist_identity(store, identity, (link,))

    assert (
        store.get(EvidenceTable.IDENTITIES, identity.identity_id)["canonical_name"]
        == "T1"
    )
    assert (
        store.get(EvidenceTable.PROVIDER_LINKS, link.link_id)["provider_entity_id"]
        == "oe:t1"
    )
