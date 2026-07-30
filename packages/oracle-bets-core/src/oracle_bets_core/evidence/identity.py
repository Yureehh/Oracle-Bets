"""Strict canonical identities and versioned provider links."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from oracle_bets_core.evidence import EvidenceTable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from oracle_bets_core.evidence import EvidenceStore


class IdentityConflictError(ValueError):
    """Raised when one provider identifier would resolve to multiple entities."""


class EntityType(StrEnum):
    TEAM = "team"
    PLAYER = "player"
    LEAGUE = "league"
    SERIES = "series"
    MAP = "map"


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        msg = f"{field_name} must be timezone-aware UTC."
        raise ValueError(msg)
    return value


def canonical_identity_id(
    sport: str,
    entity_type: EntityType,
    provider: str,
    provider_entity_id: str,
) -> str:
    """Build a stable internal ID from the first authoritative provider identity."""
    sport_value = _text(sport, "sport").casefold()
    provider_value = _text(provider, "provider").casefold()
    provider_id = _text(provider_entity_id, "provider_entity_id")
    if not isinstance(entity_type, EntityType):
        entity_type = EntityType(entity_type)
    identity_uuid = uuid5(
        NAMESPACE_URL,
        f"{sport_value}|{entity_type.value}|{provider_value}|{provider_id}",
    )
    return f"{sport_value}:{entity_type.value}:{identity_uuid}"


@dataclass(frozen=True)
class CanonicalIdentity:
    identity_id: str
    sport: str
    entity_type: EntityType
    canonical_name: str
    classification: str
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "identity_id",
            "sport",
            "canonical_name",
            "classification",
        ):
            _text(getattr(self, field_name), field_name)
        if not isinstance(self.entity_type, EntityType):
            msg = "entity_type must be an EntityType."
            raise TypeError(msg)
        _utc(self.created_at, "created_at")


@dataclass(frozen=True)
class ProviderLink:
    link_id: str
    identity_id: str
    provider: str
    provider_entity_id: str
    valid_from: datetime
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "link_id",
            "identity_id",
            "provider",
            "provider_entity_id",
        ):
            _text(getattr(self, field_name), field_name)
        _utc(self.valid_from, "valid_from")
        if self.valid_to is not None:
            _utc(self.valid_to, "valid_to")
            if self.valid_to <= self.valid_from:
                msg = "valid_to must be after valid_from."
                raise ValueError(msg)

    def active_at(self, moment: datetime) -> bool:
        checked = _utc(moment, "moment")
        return self.valid_from <= checked and (
            self.valid_to is None or checked < self.valid_to
        )


def _links_overlap(first: ProviderLink, second: ProviderLink) -> bool:
    first_end = first.valid_to or datetime.max.replace(tzinfo=UTC)
    second_end = second.valid_to or datetime.max.replace(tzinfo=UTC)
    return first.valid_from < second_end and second.valid_from < first_end


class IdentityRegistry:
    """In-memory strict resolver used before identities are persisted."""

    def __init__(self) -> None:
        self._identities: dict[str, CanonicalIdentity] = {}
        self._links: dict[str, ProviderLink] = {}

    def add(self, identity: CanonicalIdentity) -> None:
        existing = self._identities.get(identity.identity_id)
        if existing is not None and existing != identity:
            msg = f"Identity {identity.identity_id} already has different facts."
            raise IdentityConflictError(msg)
        self._identities[identity.identity_id] = identity

    def add_link(self, link: ProviderLink) -> None:
        if link.identity_id not in self._identities:
            msg = f"Unknown canonical identity: {link.identity_id}"
            raise IdentityConflictError(msg)
        existing_by_id = self._links.get(link.link_id)
        if existing_by_id is not None:
            if existing_by_id == link:
                return
            msg = f"Provider link {link.link_id} already has different facts."
            raise IdentityConflictError(msg)
        for existing in self._links.values():
            same_provider_key = (
                existing.provider.casefold() == link.provider.casefold()
                and existing.provider_entity_id == link.provider_entity_id
            )
            if same_provider_key and _links_overlap(existing, link):
                msg = (
                    f"{link.provider}:{link.provider_entity_id} is already linked "
                    f"to {existing.identity_id} for an overlapping period."
                )
                raise IdentityConflictError(msg)
        self._links[link.link_id] = link

    def resolve(
        self,
        provider: str,
        provider_entity_id: str,
        *,
        at: datetime,
    ) -> CanonicalIdentity | None:
        active = [
            link
            for link in self._links.values()
            if link.provider.casefold() == provider.casefold()
            and link.provider_entity_id == provider_entity_id
            and link.active_at(at)
        ]
        if len(active) > 1:
            msg = (
                f"Ambiguous active provider links for {provider}:{provider_entity_id}."
            )
            raise IdentityConflictError(msg)
        return self._identities[active[0].identity_id] if active else None

    def find_exact_name(
        self,
        name: str,
        *,
        classification: str | None = None,
        entity_type: EntityType | None = None,
    ) -> tuple[CanonicalIdentity, ...]:
        query = _text(name, "name").casefold()
        return tuple(
            identity
            for identity in self._identities.values()
            if identity.canonical_name.casefold() == query
            and (
                classification is None
                or identity.classification.casefold() == classification.casefold()
            )
            and (entity_type is None or identity.entity_type == entity_type)
        )


@dataclass(frozen=True)
class FixtureFacts:
    provider: str
    provider_fixture_id: str
    competition_id: str
    team_a_identity_id: str
    team_b_identity_id: str
    start_time: datetime
    best_of: int | None

    def __post_init__(self) -> None:
        for field_name in (
            "provider",
            "provider_fixture_id",
            "competition_id",
            "team_a_identity_id",
            "team_b_identity_id",
        ):
            _text(getattr(self, field_name), field_name)
        if self.team_a_identity_id == self.team_b_identity_id:
            msg = "Fixture teams must be distinct."
            raise ValueError(msg)
        _utc(self.start_time, "start_time")
        if self.best_of not in {None, 1, 2, 3, 5}:
            msg = "best_of must be 1, 2, 3, 5, or unknown."
            raise ValueError(msg)


def fixtures_agree(first: FixtureFacts, second: FixtureFacts) -> bool:
    """Return true only when all provider and fixture identity facts agree."""
    return first == second


def persist_identity(
    store: EvidenceStore,
    identity: CanonicalIdentity,
    links: Iterable[ProviderLink] = (),
) -> str:
    """Append a canonical identity and its versioned provider links."""
    identity_id = store.append(
        EvidenceTable.IDENTITIES,
        {
            "id": identity.identity_id,
            "entity_type": identity.entity_type.value,
            "canonical_name": identity.canonical_name,
            "created_at": identity.created_at,
            "idempotency_key": f"identity:{identity.identity_id}",
            "payload_json": {
                "sport": identity.sport,
                "classification": identity.classification,
            },
        },
    )
    for link in links:
        if link.identity_id != identity.identity_id:
            msg = (
                f"Provider link {link.link_id} belongs to {link.identity_id}, "
                f"not {identity.identity_id}."
            )
            raise IdentityConflictError(msg)
        store.append(
            EvidenceTable.PROVIDER_LINKS,
            {
                "id": link.link_id,
                "identity_id": link.identity_id,
                "provider": link.provider,
                "provider_entity_id": link.provider_entity_id,
                "valid_from": link.valid_from,
                "valid_to": link.valid_to,
                "idempotency_key": (
                    f"provider-link:{link.provider.casefold()}:"
                    f"{link.provider_entity_id}:{link.valid_from.isoformat()}"
                ),
                "payload_json": {},
            },
        )
    return identity_id
