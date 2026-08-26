"""Pure presentation helpers for the Discord owner console."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lol_bets.operations.models import ModelRegistry
    from oracle_bets_core.evidence import EvidenceStore

_DISCORD_SELECT_LIMIT = 25


def schedule_pages(rows: Sequence[dict[str, Any]], *, page_size: int = 8) -> list[str]:
    lines = [
        f"{_short_datetime(row.get('start_utc'))} · {row.get('league')} · "
        f"BO{row.get('best_of')} · **{row.get('team_a')} vs {row.get('team_b')}**"
        for row in rows
    ]
    return _pages("**Upcoming actionable fixtures (UTC)**", lines, page_size)


def bet_pages(
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    page_size: int = 6,
) -> list[str]:
    lines = []
    for row in rows:
        payload = row.get("payload") or {}
        line = f" {payload['line']}" if payload.get("line") is not None else ""
        settlement = row.get("settlement") or {}
        result = f" · **{settlement['result']}**" if settlement else ""
        lines.append(
            f"`{row['id']}` · **{row['selection']}** · "
            f"{row['provider']} {row['target']}{line} · {row['accepted_odds']} · "
            f"{row['stake_percent']}% ({row['stake_amount']} {row['currency']})"
            f"{result}"
        )
    return _pages(f"**{title}**", lines, page_size)


def market_options(
    store: EvidenceStore, market_ids: Sequence[str] = ()
) -> list[dict[str, str]]:
    from oracle_bets_core.evidence import EvidenceTable

    candidates = list(store.get_many(EvidenceTable.MARKET_CANDIDATES, market_ids))
    candidates.sort(
        key=lambda row: _payload(row.get("payload_json")).get("probability") is None
    )
    output: list[dict[str, str]] = []
    for row in candidates:
        payload = _payload(row.get("payload_json"))
        odds = payload.get("decimal_odds")
        line = f" {payload['line']}" if payload.get("line") is not None else ""
        output.append(
            {
                "market_id": str(row["id"]),
                "review_id": str(row["run_id"]),
                "label": f"{payload.get('target', 'unknown')}: {payload.get('selection', '')}"[
                    :100
                ],
                "description": (
                    f"{row['provider']}{line} · odds {float(odds):.2f}"
                    if odds is not None
                    else f"{row['provider']}{line} · model unavailable"
                )[:100],
            }
        )
        if len(output) == _DISCORD_SELECT_LIMIT:
            break
    return output


def health_message(store: EvidenceStore, registry: ModelRegistry) -> str:
    from lol_bets.module import LoLBetsModule
    from oracle_bets_core.markets import PolymarketGammaAdapter
    from oracle_bets_core.operations.bets import count_open_bets

    champion = registry.champion_id()
    actionable = bool(champion and registry.is_actionable(champion))
    try:
        data_ok = LoLBetsModule().artifact_health().ok
    except Exception:
        data_ok = False
    try:
        market_ok = bool(
            PolymarketGammaAdapter().search_markets("League of Legends", limit=1)
        )
    except Exception:
        market_ok = False
    integrity = store.integrity_check()
    open_bets = count_open_bets(store)
    overall_ok = actionable and data_ok and integrity == "ok" and market_ok
    return "\n".join(
        (
            "**Oracle Bets · System Health**",
            "🟢 **All systems operational**"
            if overall_ok
            else "🟠 **Attention required**",
            "",
            "**Serving model**",
            f"> Champion: `{champion or 'none'}`",
            f"> Status: **{'Actionable' if actionable else 'Research only'}**",
            f"> Artifacts: {'✅ Healthy' if data_ok else '❌ Failed'}",
            "",
            "**Evidence**",
            f"> Database: {'✅ Healthy' if integrity == 'ok' else '❌ Failed'}",
            f"> Open tracked bets: **{open_bets}**",
            "",
            "**Integrations**",
            f"> Polymarket: {'✅ Connected' if market_ok else '❌ Unavailable'} · read-only",
            "> Thunderpick: 📝 Manual lines only",
            "> Discord Gateway: ✅ Online",
        )
    )


def _short_datetime(value: Any) -> str:
    try:
        parsed = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
    except (TypeError, ValueError):
        return str(value).replace("T", " ")[:16].replace(" ", ":")
    return parsed.strftime("%Y-%m-%d:%H:%M")


def _pages(title: str, lines: Sequence[str], page_size: int) -> list[str]:
    if not lines:
        return [f"{title}\nNo rows available."]
    return [
        "\n".join((title, *lines[index : index + page_size]))
        for index in range(0, len(lines), page_size)
    ]


def _payload(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
