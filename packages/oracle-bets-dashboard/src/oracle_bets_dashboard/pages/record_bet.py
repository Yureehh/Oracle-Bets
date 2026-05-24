"""Record Bet page — form to log new bets with full metadata."""

from __future__ import annotations

from datetime import date

import streamlit as st

from oracle_bets_dashboard.database import insert_bet, insert_parlay

SPORTS = [
    "League of Legends",
    "CS2",
    "Valorant",
    "Football",
    "Basketball",
    "Tennis",
    "MMA",
    "Other",
]
MARKET_TYPES = [
    "Match Winner",
    "Map Winner",
    "Over/Under Kills",
    "Over/Under Towers",
    "Over/Under Rounds",
    "Over/Under Goals",
    "Over/Under Game Length",
    "Handicap",
    "First Blood",
    "First Dragon",
    "First Baron",
    "Correct Score",
    "Props",
    "Outright",
    "Other",
]
BOOKMAKERS = [
    "Betway",
    "bet365",
    "Pinnacle",
    "Unibet",
    "1xBet",
    "GG.bet",
    "Polymarket",
    "Other",
]
CONFIDENCE_TIERS = ["Very High", "High", "Medium", "Low", "Gut Feel"]


def render() -> None:
    st.markdown("# Record Bet")
    st.markdown(
        '<p style="color:#A1A1AA; margin-top:-10px;">Log a new bet with all the details</p>',
        unsafe_allow_html=True,
    )

    tab_single, tab_parlay = st.tabs(["Single Bet", "Parlay / Multi"])

    with tab_single:
        _render_single_form()

    with tab_parlay:
        _render_parlay_form()


MIN_PARLAY_LEGS = 2


def _render_single_form() -> None:  # noqa: PLR0915
    with st.form("record_bet_form", clear_on_submit=True):
        st.markdown("#### Event Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            event_date = st.date_input("Event Date", value=date.today())
        with col2:
            sport = st.selectbox("Sport", SPORTS)
        with col3:
            league = st.text_input(
                "League / Tournament", placeholder="e.g. LCK Spring 2026"
            )

        col1, col2 = st.columns(2)
        with col1:
            event_name = st.text_input(
                "Event / Match *", placeholder="e.g. T1 vs Gen.G"
            )
        with col2:
            market_type = st.selectbox("Market Type", MARKET_TYPES)

        st.markdown("#### Selection & Odds")
        col1, col2, col3 = st.columns(3)
        with col1:
            selection = st.text_input("Your Selection *", placeholder="e.g. T1 to win")
        with col2:
            odds_decimal = st.number_input(
                "Odds (decimal) *", min_value=1.01, value=1.90, step=0.01
            )
        with col3:
            side = st.text_input(
                "Side (optional)", placeholder="e.g. Over, Home, Blue side"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            line = st.number_input(
                "Line (for O/U)",
                value=0.0,
                step=0.5,
                help="e.g. 26.5 for kills over/under",
            )
        with col2:
            over_under = st.selectbox("Over/Under", ["N/A", "Over", "Under"])
        with col3:
            best_of = st.selectbox(
                "Best Of",
                [None, 1, 2, 3, 5],
                format_func=lambda x: "N/A" if x is None else f"BO{x}",
            )

        st.markdown("#### Stake & Classification")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            is_smoke = st.toggle(
                "🔬 Smoke Bet", value=False, help="Paper trade — not real money"
            )
        with col2:
            is_live = st.toggle("⚡ Live Bet", value=False)
        with col3:
            stake = st.number_input("Stake (€)", min_value=0.0, value=0.0, step=1.0)
        with col4:
            bookmaker = st.selectbox("Bookmaker", BOOKMAKERS)

        st.markdown("#### Model Signals (optional)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_probability = st.number_input(
                "Model Prob", min_value=0.0, max_value=1.0, value=0.0, step=0.01
            )
        with col2:
            implied_probability = st.number_input(
                "Implied Prob", min_value=0.0, max_value=1.0, value=0.0, step=0.01
            )
        with col3:
            edge = st.number_input("Edge", value=0.0, step=0.01)
        with col4:
            confidence = st.selectbox("Confidence", ["—", *CONFIDENCE_TIERS])

        st.markdown("#### Notes & Tags")
        col1, col2 = st.columns(2)
        with col1:
            notes = st.text_area(
                "Notes", placeholder="Reasoning, context, anything useful...", height=80
            )
        with col2:
            tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="e.g. value-bet, elo-mismatch, playoff",
            )
            source_prediction = st.text_input(
                "Source Prediction", placeholder="e.g. discord !lol predict"
            )

        submitted = st.form_submit_button("💾  Save Bet", use_container_width=True)

        if submitted:
            if not event_name or not selection:
                st.error("Event name and selection are required.")
                return

            potential_payout = stake * odds_decimal if stake > 0 else None

            data = {
                "event_date": event_date.isoformat(),
                "sport": sport,
                "league": league or None,
                "event_name": event_name,
                "market_type": market_type,
                "selection": selection,
                "side": side or None,
                "odds_decimal": odds_decimal,
                "stake": stake if stake > 0 else None,
                "potential_payout": potential_payout,
                "is_smoke": int(is_smoke),
                "is_live": int(is_live),
                "bookmaker": bookmaker,
                "model_probability": model_probability
                if model_probability > 0
                else None,
                "implied_probability": implied_probability
                if implied_probability > 0
                else None,
                "edge": edge if edge != 0 else None,
                "confidence_tier": confidence if confidence != "—" else None,
                "notes": notes or None,
                "tags": tags or None,
                "source_prediction": source_prediction or None,
                "line": line if line != 0 else None,
                "over_under": over_under if over_under != "N/A" else None,
                "best_of": best_of,
            }

            try:
                bet_id = insert_bet(data)
            except ValueError as e:
                st.error(str(e))
                return
            st.success(
                f"Bet #{bet_id} recorded! {'🔬 Smoke bet' if is_smoke else '💰 Real money'}"
            )


def _render_parlay_form() -> None:
    st.markdown("#### Parlay Builder")
    st.caption("Record a multi-leg bet. Each leg is saved as a linked single bet.")

    with st.form("parlay_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            event_date = st.date_input(
                "Event Date", value=date.today(), key="parlay_date"
            )
        with col2:
            combined_odds = st.number_input(
                "Combined Odds", min_value=1.01, value=3.0, step=0.01
            )
        with col3:
            stake = st.number_input(
                "Total Stake (€)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="parlay_stake",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            is_smoke = st.toggle("🔬 Smoke Bet", value=False, key="parlay_smoke")
        with col2:
            bookmaker = st.selectbox("Bookmaker", BOOKMAKERS, key="parlay_book")
        with col3:
            num_legs = st.number_input(
                "Number of legs", min_value=2, max_value=10, value=2
            )

        st.markdown("---")
        st.markdown("#### Legs")
        legs = []
        for i in range(int(num_legs)):
            st.markdown(f"**Leg {i + 1}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                leg_event = st.text_input(
                    "Event", key=f"leg_event_{i}", placeholder="Match name"
                )
            with c2:
                leg_selection = st.text_input(
                    "Selection", key=f"leg_sel_{i}", placeholder="Your pick"
                )
            with c3:
                leg_odds = st.number_input(
                    "Odds", min_value=1.01, value=1.50, step=0.01, key=f"leg_odds_{i}"
                )
            with c4:
                leg_sport = st.selectbox("Sport", SPORTS, key=f"leg_sport_{i}")
            legs.append(
                {
                    "event_name": leg_event,
                    "selection": leg_selection,
                    "odds_decimal": leg_odds,
                    "sport": leg_sport,
                }
            )

        notes = st.text_area("Parlay Notes", key="parlay_notes", height=60)

        submitted = st.form_submit_button("💾  Save Parlay", use_container_width=True)

        if submitted:
            valid_legs = [leg for leg in legs if leg["event_name"] and leg["selection"]]
            if len(valid_legs) < MIN_PARLAY_LEGS:
                st.error("At least 2 legs with event name and selection are required.")
                return

            parlay_data = {
                "event_date": event_date.isoformat(),
                "combined_odds": combined_odds,
                "stake": stake if stake > 0 else None,
                "potential_payout": stake * combined_odds if stake > 0 else None,
                "is_smoke": int(is_smoke),
                "bookmaker": bookmaker,
                "notes": notes or None,
            }
            parlay_id = insert_parlay(parlay_data)

            for leg in valid_legs:
                insert_bet(
                    {
                        "event_date": event_date.isoformat(),
                        "sport": leg["sport"],
                        "event_name": leg["event_name"],
                        "market_type": "Match Winner",
                        "selection": leg["selection"],
                        "odds_decimal": leg["odds_decimal"],
                        "stake": None,
                        "is_smoke": int(is_smoke),
                        "is_parlay": 1,
                        "parlay_id": parlay_id,
                        "bookmaker": bookmaker,
                    }
                )

            st.success(f"Parlay #{parlay_id} with {len(valid_legs)} legs saved!")
