"""Ledger page — browse, filter, settle, and edit bets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from oracle_bets_dashboard.database import delete_bet, get_all_bets, settle_bet


def render() -> None:  # noqa: PLR0915
    st.markdown("# Ledger")
    st.markdown(
        '<p style="color:#A1A1AA; margin-top:-10px;">All your bets in one place</p>',
        unsafe_allow_html=True,
    )

    bets = get_all_bets()
    if not bets:
        st.info("No bets yet. Head to **Record Bet** to get started.")
        return

    df = pd.DataFrame(bets)

    # Filters
    with st.expander("🔍 Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_type = st.multiselect(
                "Type", ["Real", "Smoke"], default=["Real", "Smoke"]
            )
        with col2:
            sports = df["sport"].unique().tolist()
            filter_sport = st.multiselect("Sport", sports, default=sports)
        with col3:
            filter_status = st.multiselect(
                "Status",
                ["pending", "win", "loss", "push"],
                default=["pending", "win", "loss", "push"],
            )
        with col4:
            bookmakers = [b for b in df["bookmaker"].unique().tolist() if b]
            filter_book = st.multiselect("Bookmaker", bookmakers, default=bookmakers)

    mask = pd.Series(True, index=df.index)
    type_vals = []
    if "Real" in filter_type:
        type_vals.append(0)
    if "Smoke" in filter_type:
        type_vals.append(1)
    mask &= df["is_smoke"].isin(type_vals)
    mask &= df["sport"].isin(filter_sport)

    status_mask = pd.Series(False, index=df.index)
    for s in filter_status:
        if s == "pending":
            status_mask |= df["status"] == "pending"
        else:
            status_mask |= df["result"] == s
    mask &= status_mask

    if filter_book:
        mask &= df["bookmaker"].isin(filter_book)

    filtered = df[mask].copy()

    st.markdown(f"**{len(filtered)}** bets shown")
    st.markdown("")

    # Display styled table
    display_cols = [
        "id",
        "event_date",
        "sport",
        "event_name",
        "market_type",
        "selection",
        "odds_decimal",
        "stake",
        "is_smoke",
        "status",
        "result",
        "pnl",
        "bookmaker",
        "edge",
        "confidence_tier",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available_cols].copy()

    display_df["is_smoke"] = display_df["is_smoke"].map({0: "💰 Real", 1: "🔬 Smoke"})
    display_df["result"] = display_df["result"].fillna("—")
    display_df["pnl"] = display_df["pnl"].apply(
        lambda x: f"€{x:+.2f}" if pd.notna(x) else "—"
    )
    display_df["edge"] = display_df["edge"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display_df["stake"] = display_df["stake"].apply(
        lambda x: f"€{x:.2f}" if pd.notna(x) else "—"
    )

    display_df.columns = [
        "ID",
        "Date",
        "Sport",
        "Event",
        "Market",
        "Selection",
        "Odds",
        "Stake",
        "Type",
        "Status",
        "Result",
        "P&L",
        "Book",
        "Edge",
        "Confidence",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Settle bets section
    st.markdown("---")
    st.markdown("### Settle a Bet")

    pending_bets = filtered[filtered["status"] == "pending"]
    if pending_bets.empty:
        st.caption("No pending bets to settle.")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            options = {
                row[
                    "id"
                ]: f"#{row['id']} — {row['selection']} @ {row['odds_decimal']} ({row['event_name']})"
                for _, row in pending_bets.iterrows()
            }
            selected_id = st.selectbox(
                "Select bet to settle", options.keys(), format_func=lambda x: options[x]
            )
        with col2:
            result = st.selectbox("Result", ["win", "loss", "push", "void"])
        with col3:
            st.markdown("")
            st.markdown("")
            if st.button("✓ Settle", use_container_width=True):
                settled = settle_bet(int(selected_id), result)
                st.success(
                    f"Bet #{selected_id} settled: **{result}** (P&L: €{settled['pnl']:+.2f})"
                )
                st.rerun()

    # Delete section
    st.markdown("---")
    with st.expander("⚠️ Delete a Bet"):
        if not filtered.empty:
            del_options = {
                row["id"]: f"#{row['id']} — {row['selection']} ({row['event_name']})"
                for _, row in filtered.iterrows()
            }
            del_id = st.selectbox(
                "Select bet to delete",
                del_options.keys(),
                format_func=lambda x: del_options[x],
            )
            if st.button("🗑️ Delete", type="secondary"):
                delete_bet(del_id)
                st.warning(f"Bet #{del_id} deleted.")
                st.rerun()
