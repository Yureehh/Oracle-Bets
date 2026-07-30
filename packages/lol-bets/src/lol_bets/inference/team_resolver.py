"""
Team name resolution between external schedule sources and Oracle's Elixir.

External sources (PandaScore schedules, Polymarket questions) spell team names
differently from the Oracle's Elixir ``teamname`` column. This module resolves
a raw external name to a known teamname using, in order:

1. exact case-insensitive match,
2. curated alias map (``config/lol/data_ingestion/team_aliases.json``),
3. normalized match (punctuation stripped, org suffixes like "Esports" removed),
4. fuzzy *suggestions only* — never auto-accepted, surfaced so a human can add
   an alias entry.

The resolver is a pure function over ``known_names``; it performs no I/O beyond
reading the alias config (cached).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache

from oracle_bets_core.io_utils import FileLoadError, json_loader
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import TEAM_ALIASES

_DEFAULT_SUFFIXES: tuple[str, ...] = (
    "esports",
    "esport",
    "e-sports",
    "gaming",
    "team",
    "club",
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
MAX_SUGGESTIONS = 3
FUZZY_CUTOFF = 0.75


@dataclass(frozen=True)
class ResolvedTeam:
    """Outcome of a resolution attempt."""

    query: str
    resolved_name: str | None
    method: str | None  # "exact" | "alias" | "normalized" | None
    suggestions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.resolved_name is not None


@lru_cache(maxsize=1)
def _load_alias_config() -> tuple[dict[str, str], tuple[str, ...]]:
    """Return (casefolded alias map, suffixes to strip). Fail soft to defaults."""
    try:
        cfg = json_loader(TEAM_ALIASES)
    except (FileLoadError, FileNotFoundError):
        logger.warning("Team alias config missing at %s; using defaults.", TEAM_ALIASES)
        return {}, _DEFAULT_SUFFIXES
    aliases_raw = cfg.get("external_aliases", {})
    aliases = {
        str(key).casefold(): str(value)
        for key, value in aliases_raw.items()
        if str(key).strip() and str(value).strip()
    }
    suffixes = tuple(
        str(s).casefold() for s in cfg.get("suffixes_to_strip", _DEFAULT_SUFFIXES)
    )
    return aliases, suffixes


def clear_alias_cache() -> None:
    """Testing hook: drop the cached alias config."""
    _load_alias_config.cache_clear()


def _normalize(name: str, suffixes: tuple[str, ...]) -> str:
    """Lowercase, strip punctuation, drop org-suffix tokens."""
    tokens = [t for t in _NON_ALNUM.split(name.casefold()) if t]
    stripped = [t for t in tokens if t not in suffixes]
    # Never normalize a name away entirely (e.g. "Team Gaming").
    return " ".join(stripped or tokens)


def resolve_team_name(
    name: str,
    known_names: list[str] | tuple[str, ...],
) -> ResolvedTeam:
    """
    Resolve *name* against *known_names* (Oracle's Elixir teamname values).

    Fuzzy matches are returned only as suggestions; the caller must not treat
    them as resolutions. A wrong silent match is worse than a loud failure.
    """
    query = str(name).strip()
    if not query:
        return ResolvedTeam(query=query, resolved_name=None, method=None)

    aliases, suffixes = _load_alias_config()
    by_casefold: dict[str, str] = {}
    for known in known_names:
        by_casefold.setdefault(str(known).casefold(), str(known))

    # 1. Exact (case-insensitive)
    exact = by_casefold.get(query.casefold())
    if exact is not None:
        return ResolvedTeam(query=query, resolved_name=exact, method="exact")

    # 2. Curated alias
    alias_target = aliases.get(query.casefold())
    if alias_target is not None:
        canonical = by_casefold.get(alias_target.casefold())
        if canonical is not None:
            return ResolvedTeam(query=query, resolved_name=canonical, method="alias")
        logger.warning(
            "Alias '%s' -> '%s' does not match any known team; fix team_aliases.json.",
            query,
            alias_target,
        )

    # 3. Normalized (suffix/punctuation-insensitive), only when unambiguous
    normalized_query = _normalize(query, suffixes)
    normalized_known: dict[str, list[str]] = {}
    for casefolded, canonical in by_casefold.items():
        normalized_known.setdefault(_normalize(casefolded, suffixes), []).append(
            canonical
        )
    candidates = normalized_known.get(normalized_query, [])
    if len(candidates) == 1:
        return ResolvedTeam(
            query=query, resolved_name=candidates[0], method="normalized"
        )
    if len(candidates) > 1:
        logger.warning(
            "Team name '%s' is ambiguous after normalization: %s", query, candidates
        )
        return ResolvedTeam(
            query=query,
            resolved_name=None,
            method=None,
            suggestions=tuple(sorted(candidates)[:MAX_SUGGESTIONS]),
        )

    # 4. Fuzzy suggestions only (never auto-accepted)
    matches = difflib.get_close_matches(
        normalized_query,
        list(normalized_known),
        n=MAX_SUGGESTIONS,
        cutoff=FUZZY_CUTOFF,
    )
    suggestions: list[str] = []
    for match in matches:
        for canonical in normalized_known[match]:
            if canonical not in suggestions:
                suggestions.append(canonical)
    return ResolvedTeam(
        query=query,
        resolved_name=None,
        method=None,
        suggestions=tuple(suggestions[:MAX_SUGGESTIONS]),
    )


class TeamResolutionError(ValueError):
    """Raised when an external team name cannot be resolved to a known team."""

    def __init__(self, resolved: ResolvedTeam) -> None:
        self.resolved = resolved
        hint = (
            f" Did you mean: {', '.join(resolved.suggestions)}?"
            " Add an entry to config/lol/data_ingestion/team_aliases.json."
            if resolved.suggestions
            else " Add an entry to config/lol/data_ingestion/team_aliases.json"
            " if this team is known under another name."
        )
        super().__init__(f"Team '{resolved.query}' not found in teams table.{hint}")
