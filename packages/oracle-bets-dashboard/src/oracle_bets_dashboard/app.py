"""Oracle Bets Dashboard — main Streamlit app."""

from __future__ import annotations

import streamlit as st

from oracle_bets_dashboard.database import init_db
from oracle_bets_dashboard.pages import (
    analytics,
    bankroll,
    ledger,
    overview,
    record_bet,
)
from oracle_bets_dashboard.styles import CUSTOM_CSS

init_db()

st.set_page_config(
    page_title="Oracle Bets",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🔮 Oracle Bets")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Record Bet", "Ledger", "Analytics", "Bankroll"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Overview": "◈  Overview",
            "Record Bet": "✦  Record Bet",
            "Ledger": "☰  Ledger",
            "Analytics": "◉  Analytics",
            "Bankroll": "◎  Bankroll",
        }[x],
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem; color:#71717A; text-align:center;">'
        "Oracle Bets Dashboard v1.0</p>",
        unsafe_allow_html=True,
    )

if page == "Overview":
    overview.render()
elif page == "Record Bet":
    record_bet.render()
elif page == "Ledger":
    ledger.render()
elif page == "Analytics":
    analytics.render()
elif page == "Bankroll":
    bankroll.render()
