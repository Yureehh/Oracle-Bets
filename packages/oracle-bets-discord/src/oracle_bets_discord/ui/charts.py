"""In-memory unified bet-performance charts for Discord."""

from __future__ import annotations

from io import BytesIO
from typing import Any


def performance_png(
    rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[bytes, dict[str, Any]]:
    """Render one mode without combining currencies."""
    from oracle_bets_core.operations.bets import summarize_bets

    rows.reverse()
    summary = summarize_bets(rows, mode=mode) | {"settled_bets": len(rows)}
    if not rows:
        return b"", summary

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    by_currency: dict[str, list[float]] = {}
    running: dict[str, float] = {}
    for row in rows:
        currency = str(row["currency"])
        running[currency] = running.get(currency, 0.0) + float(
            row["settlement"]["pnl_amount"]
        )
        by_currency.setdefault(currency, []).append(running[currency])
    for currency, values in by_currency.items():
        axes[0, 0].plot(range(1, len(values) + 1), values, marker="o", label=currency)
    axes[0, 0].axhline(0, color="grey", linewidth=0.8)
    axes[0, 0].legend()
    axes[0, 0].set(title="Cumulative PnL", xlabel="Settled bets")

    target_roi = _group_roi(rows, "target")
    axes[0, 1].bar(list(target_roi), list(target_roi.values()))
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set(title="ROI by target", ylabel="ROI")

    provider_roi = _group_roi(rows, "provider")
    axes[1, 0].bar(list(provider_roi), list(provider_roi.values()))
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set(title="ROI by provider", ylabel="ROI")

    supported = [
        row
        for row in rows
        if row["evidence_classification"] != "model_unavailable"
        and row["settlement"]["result"] in {"win", "loss"}
        and row["payload"].get("model_probability") is not None
    ]
    probabilities = [float(row["payload"]["model_probability"]) for row in supported]
    outcomes = [
        1.0 if row["settlement"]["result"] == "win" else 0.0 for row in supported
    ]
    if probabilities:
        axes[1, 1].scatter(probabilities, outcomes, alpha=0.6)
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1, 1].set(
        title="Model-backed outcomes",
        xlabel="Entry probability",
        ylabel="Result",
        xlim=(0, 1),
        ylim=(-0.05, 1.05),
    )
    figure.suptitle(f"Oracle Bets · {mode} performance")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130)
    plt.close(figure)
    return buffer.getvalue(), summary


def _group_roi(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row.get(field) or 'unknown'} · {row['currency']}"
        values = totals.setdefault(key, [0.0, 0.0])
        values[0] += float(row["settlement"]["pnl_amount"])
        values[1] += float(row["stake_amount"])
    return {key: pnl / stake for key, (pnl, stake) in totals.items() if stake}
