"""Bankroll page — track deposits, withdrawals, and balance over time."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oracle_bets_dashboard.database import (
    get_all_bets,
    get_bankroll_history,
    get_summary_stats,
    insert_bankroll_entry,
)

COLORS = {
    "accent": "#7C3AED",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "bg_card": "#16162A",
    "border": "#27273F",
    "text_secondary": "#A1A1AA",
}


def render() -> None:  # noqa: PLR0915
    st.markdown("# Bankroll")
    st.markdown(
        '<p style="color:#A1A1AA; margin-top:-10px;">Track your betting capital</p>',
        unsafe_allow_html=True,
    )

    history = get_bankroll_history()

    # Current balance & quick stats
    if history:
        current_balance = history[0]["balance"]
        total_deposited = sum(h["deposit"] or 0 for h in history)
        total_withdrawn = sum(h["withdrawal"] or 0 for h in history)
    else:
        current_balance = 0
        total_deposited = 0
        total_withdrawn = 0

    real_stats = get_summary_stats(smoke=False)
    total_pnl = real_stats.get("total_pnl") or 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Balance", f"€{current_balance:.2f}")
    col2.metric("Total Deposited", f"€{total_deposited:.2f}")
    col3.metric("Total Withdrawn", f"€{total_withdrawn:.2f}")
    col4.metric("Lifetime P&L", f"€{total_pnl:+.2f}")
    col5.metric(
        "Profit on Capital",
        f"{total_pnl / total_deposited * 100:+.1f}%" if total_deposited > 0 else "—",
    )

    st.markdown("---")

    col_form, col_chart = st.columns([1, 2])

    with col_form:
        st.markdown("### Log Entry")
        with st.form("bankroll_form", clear_on_submit=True):
            balance = st.number_input("Current Balance (€)", min_value=0.0, step=1.0)
            deposit = st.number_input("Deposit (€)", min_value=0.0, step=1.0, value=0.0)
            withdrawal = st.number_input(
                "Withdrawal (€)", min_value=0.0, step=1.0, value=0.0
            )
            notes = st.text_input("Notes", placeholder="e.g. Initial deposit, cashout")

            if st.form_submit_button("💾 Log", use_container_width=True):
                insert_bankroll_entry(balance, deposit, withdrawal, notes)
                st.success("Bankroll entry logged!")
                st.rerun()

    with col_chart:
        st.markdown("### Balance History")
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["logged_at"] = pd.to_datetime(hist_df["logged_at"])
            hist_df = hist_df.sort_values("logged_at")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["logged_at"],
                    y=hist_df["balance"],
                    mode="lines+markers",
                    fill="tozeroy",
                    line={"color": COLORS["accent"], "width": 2},
                    marker={"size": 6},
                    fillcolor="rgba(124, 58, 237, 0.1)",
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text_secondary"]},
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                height=300,
                xaxis={"showgrid": False},
                yaxis={"gridcolor": COLORS["border"], "title": "Balance €"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No bankroll entries yet. Log your first deposit above.")

    # Risk metrics
    st.markdown("---")
    st.markdown("### Risk Metrics")

    bets = get_all_bets()
    real_bets = [b for b in bets if not b["is_smoke"] and b.get("pnl") is not None]

    if real_bets and current_balance > 0:
        pnls = [b["pnl"] for b in real_bets]
        stakes = [b["stake"] for b in real_bets if b.get("stake")]

        avg_stake = sum(stakes) / len(stakes) if stakes else 0
        max_stake = max(stakes) if stakes else 0
        max_loss = min(pnls)
        max_win = max(pnls)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Stake", f"€{avg_stake:.2f}")
        col2.metric("Max Stake", f"€{max_stake:.2f}")
        col3.metric("Worst Loss", f"€{max_loss:.2f}")
        col4.metric("Best Win", f"€{max_win:+.2f}")

        st.markdown("")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Avg Stake % of Bankroll",
            f"{avg_stake / current_balance * 100:.1f}%" if current_balance else "—",
        )
        col2.metric(
            "Max Drawdown",
            f"€{_max_drawdown(pnls):.2f}",
        )
        col3.metric(
            "Bets Until Ruin (at avg stake)",
            f"{current_balance / avg_stake:.0f}" if avg_stake > 0 else "∞",
        )
    else:
        st.caption("Settle some real bets and log your balance to see risk metrics.")


def _max_drawdown(pnls: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    return max_dd
