"""Required provider/target coverage, including fixtures with no usable quote."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

TARGET_LABELS = {
    "series_winner": "Series",
    "map_winner": "Map",
    "series_total_maps": "Total maps",
    "series_handicap": "Handicap",
    "gamelength_mean": "Length",
    "total_kills_mean": "Kills",
    "total_towers_mean": "Towers",
}
PROVIDERS = ("polymarket", "thunderpick")


def required_market_capture(
    fixture_key: str, *, actions: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
    """Retain every required family; capture does not attest tradability or terms."""
    coverage = []
    for provider in PROVIDERS:
        for target in TARGET_LABELS:
            matching = [
                row
                for row in actions
                if row.get("provider") == provider
                and row.get("target") == target
                and str(row.get("fixture_key") or fixture_key) == fixture_key
            ]
            usable = [
                row
                for row in matching
                if not row.get("hard_blocks") and row.get("decimal_odds") is not None
            ]
            if usable:
                state, reason = (
                    "captured",
                    "Observed quote; settlement terms and stake still require verification.",
                )
            elif matching:
                blocks = sorted(
                    {
                        str(block)
                        for row in matching
                        for block in row.get("hard_blocks") or []
                    }
                )
                unsupported = any(
                    block.startswith(("semantic_contract:", "unsupported_"))
                    for block in blocks
                )
                state = "unsupported" if unsupported else "missing"
                reason = "; ".join(blocks) or str(
                    matching[0].get("reason") or "No usable quote returned."
                )
            else:
                state, reason = (
                    "missing",
                    "No line captured for this provider and target.",
                )
            coverage.append(
                {
                    "fixture_key": fixture_key,
                    "provider": provider,
                    "target": target,
                    "state": state,
                    "reason": reason,
                }
            )
    return coverage


def capture_checklist(coverage: Sequence[dict[str, str]]) -> str:
    """Compact checklist used before and after owner review."""
    symbols = {"captured": "✓", "missing": "—", "unsupported": "!"}
    lines = ["**Required market capture** · ✓ captured · — missing · ! unsupported"]
    for provider in PROVIDERS:
        cells = [row for row in coverage if row["provider"] == provider]
        lines.append(
            f"{provider.title()}: "
            + " · ".join(
                f"{TARGET_LABELS[row['target']]} {symbols[row['state']]}"
                for row in cells
            )
        )
    return "\n".join(lines)
