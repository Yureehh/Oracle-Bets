"""Overview page — KPIs, recent activity, performance snapshot."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oracle_bets_dashboard.database import get_all_bets, get_summary_stats

COLORS = {
    "accent": "#7C3AED",
    "accent_light": "#A78BFA",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "bg_card": "#16162A",
    "border": "#27273F",
    "text_secondary": "#A1A1AA",
}


def render() -> None:
    st.markdown("# Overview")
    st.markdown(
        '<p style="color:#A1A1AA; margin-top:-10px;">Your betting command center</p>',
        unsafe_allow_html=True,
    )

    real_stats = get_summary_stats(smoke=False)
    smoke_stats = get_summary_stats(smoke=True)

    st.markdown("### 💰 Real Bets")
    _render_kpis(real_stats)

    st.markdown("")
    st.markdown("### 🔬 Smoke Bets (Paper Trading)")
    _render_kpis(smoke_stats)

    bets = get_all_bets()
    if not bets:
        st.markdown("---")
        st.info("No bets recorded yet. Go to **Record Bet** to add your first one!")
        return

    st.markdown("---")

    df = pd.DataFrame(bets)
    df["event_date"] = pd.to_datetime(df["event_date"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cumulative P&L")
        _render_pnl_chart(df)

    with col2:
        st.markdown("### Recent Bets")
        _render_recent_bets(df)


def _render_kpis(stats: dict) -> None:
    total = stats.get("total_bets") or 0
    wins = stats.get("wins") or 0
    losses = stats.get("losses") or 0
    pnl = stats.get("total_pnl") or 0
    staked = stats.get("total_staked") or 0
    avg_edge = stats.get("avg_edge")

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    roi = (pnl / staked * 100) if staked > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bets", total)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("P&L", f"€{pnl:+.2f}")
    c4.metric("ROI", f"{roi:+.1f}%")
    c5.metric("Avg Edge", f"{avg_edge:.1%}" if avg_edge else "—")


def _render_pnl_chart(df: pd.DataFrame) -> None:
    settled = df[df["pnl"].notna()].copy()
    if settled.empty:
        st.caption("No settled bets yet.")
        return

    settled = settled.sort_values("event_date")

    for smoke_val, _label in [(0, "Real"), (1, "Smoke")]:
        subset = settled[settled["is_smoke"] == smoke_val].copy()
        if not subset.empty:
            subset["cumulative_pnl"] = subset["pnl"].cumsum()

    settled["cumulative_pnl"] = settled.sort_values("event_date")["pnl"].cumsum()

    fig = go.Figure()

    real = settled[settled["is_smoke"] == 0].copy()
    if not real.empty:
        real["cumulative_pnl"] = real["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=real["event_date"],
                y=real["cumulative_pnl"],
                mode="lines+markers",
                name="Real",
                line={"color": COLORS["success"], "width": 2},
                marker={"size": 4},
            )
        )

    smoke = settled[settled["is_smoke"] == 1].copy()
    if not smoke.empty:
        smoke["cumulative_pnl"] = smoke["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=smoke["event_date"],
                y=smoke["cumulative_pnl"],
                mode="lines+markers",
                name="Smoke",
                line={"color": COLORS["warning"], "width": 2, "dash": "dash"},
                marker={"size": 4},
            )
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_secondary"]},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=300,
        legend={"orientation": "h", "y": -0.2},
        xaxis={"gridcolor": COLORS["border"], "showgrid": False},
        yaxis={"gridcolor": COLORS["border"], "zerolinecolor": COLORS["border"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_recent_bets(df: pd.DataFrame) -> None:
    recent = df.head(8)
    for _, row in recent.iterrows():
        result_icon = {"win": "🟢", "loss": "🔴", "push": "🟡"}.get(
            row.get("result") or "", "⏳"
        )
        smoke_tag = (
            '<span class="tag-pill tag-smoke">SMOKE</span>' if row["is_smoke"] else ""
        )
        pnl_text = f"€{row['pnl']:+.2f}" if row.get("pnl") is not None else "pending"

        st.markdown(
            f"""<div style="
                background: #16162A;
                border: 1px solid #27273F;
                border-radius: 8px;
                padding: 0.6rem 1rem;
                margin-bottom: 0.4rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span>{result_icon} <b>{row["selection"]}</b> — {row["event_name"]} {smoke_tag}</span>
                <span style="font-weight:600; color:{"#10B981" if (row.get("pnl") or 0) >= 0 else "#EF4444"}">
                    {pnl_text}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
