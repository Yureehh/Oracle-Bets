"""Strict canonical identities and versioned provider links."""

from __future__ import annotations

from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5


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
