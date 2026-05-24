"""Analytics page — deep performance breakdowns and insights."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from oracle_bets_dashboard.database import get_all_bets

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

PALETTE = ["#7C3AED", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899", "#8B5CF6"]


def render() -> None:
    st.markdown("# Analytics")
    st.markdown(
        '<p style="color:#A1A1AA; margin-top:-10px;">Deep dive into your betting performance</p>',
        unsafe_allow_html=True,
    )

    bets = get_all_bets()
    if not bets:
        st.info("No data yet. Record some bets to see analytics here.")
        return

    df = pd.DataFrame(bets)
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Filter: Real vs Smoke
    mode = st.radio(
        "Analysis Mode",
        ["All", "Real Only", "Smoke Only"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Real Only":
        df = df[df["is_smoke"] == 0]
    elif mode == "Smoke Only":
        df = df[df["is_smoke"] == 1]

    settled = df[df["result"].isin(["win", "loss", "push"])].copy()

    if settled.empty:
        st.warning("No settled bets to analyze yet.")
        return

    tab_perf, tab_markets, tab_timing, tab_edge = st.tabs(
        ["Performance", "Markets", "Timing", "Edge Analysis"]
    )

    with tab_perf:
        _render_performance(settled)

    with tab_markets:
        _render_markets(settled)

    with tab_timing:
        _render_timing(settled)

    with tab_edge:
        _render_edge_analysis(settled)


def _render_performance(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Win Rate by Sport")
        sport_stats = (
            df.groupby("sport")
            .agg(
                total=("id", "count"),
                wins=("result", lambda x: (x == "win").sum()),
            )
            .reset_index()
        )
        sport_stats["win_rate"] = sport_stats["wins"] / sport_stats["total"] * 100

        fig = px.bar(
            sport_stats.sort_values("win_rate", ascending=True),
            x="win_rate",
            y="sport",
            orientation="h",
            color="win_rate",
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
            text=sport_stats.sort_values("win_rate", ascending=True).apply(
                lambda r: f"{r['win_rate']:.0f}% ({r['wins']}/{r['total']})", axis=1
            ),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text_secondary"]},
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=250,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis={"visible": False},
        )
        fig.update_traces(textposition="auto")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### P&L by Sport")
        pnl_by_sport = df.groupby("sport")["pnl"].sum().reset_index()
        pnl_by_sport["color"] = pnl_by_sport["pnl"].apply(
            lambda x: COLORS["success"] if x >= 0 else COLORS["danger"]
        )

        fig = go.Figure(
            go.Bar(
                x=pnl_by_sport["sport"],
                y=pnl_by_sport["pnl"],
                marker_color=pnl_by_sport["color"],
                text=pnl_by_sport["pnl"].apply(lambda x: f"€{x:+.0f}"),
                textposition="auto",
            )
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text_secondary"]},
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=250,
            xaxis={"showgrid": False},
            yaxis={"showgrid": True, "gridcolor": COLORS["border"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Win Rate by Odds Range")
    df["odds_bucket"] = pd.cut(
        df["odds_decimal"],
        bins=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 100.0],
        labels=[
            "1.0-1.5",
            "1.5-2.0",
            "2.0-2.5",
            "2.5-3.0",
            "3.0-4.0",
            "4.0-6.0",
            "6.0+",
        ],
    )
    odds_stats = (
        df.groupby("odds_bucket", observed=True)
        .agg(
            total=("id", "count"),
            wins=("result", lambda x: (x == "win").sum()),
            pnl=("pnl", "sum"),
        )
        .reset_index()
    )
    odds_stats["win_rate"] = odds_stats["wins"] / odds_stats["total"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=odds_stats["odds_bucket"].astype(str),
            y=odds_stats["win_rate"],
            name="Win Rate %",
            marker_color=COLORS["accent"],
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=odds_stats["odds_bucket"].astype(str),
            y=odds_stats["pnl"],
            name="P&L",
            mode="lines+markers",
            marker={"color": COLORS["success"], "size": 8},
            line={"color": COLORS["success"], "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_secondary"]},
        margin={"l": 0, "r": 40, "t": 10, "b": 0},
        height=250,
        legend={"orientation": "h", "y": -0.2},
        yaxis={"title": "Win Rate %", "gridcolor": COLORS["border"]},
        yaxis2={"title": "P&L €", "overlaying": "y", "side": "right"},
        xaxis={"showgrid": False},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_markets(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bets by Market Type")
        market_counts = df["market_type"].value_counts().reset_index()
        market_counts.columns = ["market_type", "count"]

        fig = px.pie(
            market_counts,
            values="count",
            names="market_type",
            color_discrete_sequence=PALETTE,
            hole=0.5,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text_secondary"]},
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=300,
            legend={"font": {"size": 10}},
        )
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Performance by Market")
        market_stats = (
            df.groupby("market_type")
            .agg(
                total=("id", "count"),
                wins=("result", lambda x: (x == "win").sum()),
                pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        market_stats["win_rate"] = market_stats["wins"] / market_stats["total"] * 100
        market_stats = market_stats.sort_values("pnl", ascending=False)

        st.dataframe(
            market_stats.rename(
                columns={
                    "market_type": "Market",
                    "total": "Bets",
                    "wins": "Wins",
                    "win_rate": "Win %",
                    "pnl": "P&L €",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Bookmaker Comparison")
    book_stats = (
        df[df["bookmaker"].notna()]
        .groupby("bookmaker")
        .agg(
            total=("id", "count"),
            wins=("result", lambda x: (x == "win").sum()),
            pnl=("pnl", "sum"),
            avg_odds=("odds_decimal", "mean"),
        )
        .reset_index()
    )
    book_stats["win_rate"] = book_stats["wins"] / book_stats["total"] * 100
    book_stats["roi"] = (book_stats["pnl"] / book_stats["total"]).fillna(0)

    st.dataframe(
        book_stats.rename(
            columns={
                "bookmaker": "Bookmaker",
                "total": "Bets",
                "wins": "Wins",
                "win_rate": "Win %",
                "pnl": "P&L €",
                "avg_odds": "Avg Odds",
                "roi": "ROI/bet €",
            }
        ).sort_values("P&L €", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _render_timing(df: pd.DataFrame) -> None:
    st.markdown("#### Bets Over Time")
    df_time = df.copy()
    df_time["month"] = df_time["event_date"].dt.to_period("M").astype(str)

    monthly = (
        df_time.groupby("month")
        .agg(
            bets=("id", "count"),
            pnl=("pnl", "sum"),
            wins=("result", lambda x: (x == "win").sum()),
        )
        .reset_index()
    )
    monthly["win_rate"] = monthly["wins"] / monthly["bets"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["bets"],
            name="Bets",
            marker_color=COLORS["accent"],
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["pnl"],
            name="P&L",
            mode="lines+markers",
            marker={"color": COLORS["success"], "size": 8},
            line={"color": COLORS["success"], "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_secondary"]},
        margin={"l": 0, "r": 40, "t": 10, "b": 0},
        height=280,
        legend={"orientation": "h", "y": -0.2},
        yaxis={"title": "# Bets", "gridcolor": COLORS["border"]},
        yaxis2={"title": "P&L €", "overlaying": "y", "side": "right"},
        xaxis={"showgrid": False},
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Day of Week Performance")
        df_time["dow"] = df_time["event_date"].dt.day_name()
        dow_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow_stats = (
            df_time.groupby("dow")
            .agg(
                bets=("id", "count"),
                pnl=("pnl", "sum"),
            )
            .reindex(dow_order)
            .reset_index()
        )

        fig = go.Figure(
            go.Bar(
                x=dow_stats["dow"],
                y=dow_stats["pnl"],
                marker_color=[
                    COLORS["success"] if v >= 0 else COLORS["danger"]
                    for v in dow_stats["pnl"].fillna(0)
                ],
                text=dow_stats["pnl"].apply(
                    lambda x: f"€{x:+.0f}" if pd.notna(x) else "—"
                ),
                textposition="auto",
            )
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text_secondary"]},
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=220,
            xaxis={"showgrid": False},
            yaxis={"showgrid": True, "gridcolor": COLORS["border"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Streak Analysis")
        sorted_df = df_time.sort_values("event_date")
        streaks = []
        current_streak = 0
        max_win = 0
        max_loss = 0
        for result in sorted_df["result"]:
            if result == "win":
                current_streak = max(1, current_streak + 1)
            elif result == "loss":
                current_streak = min(-1, current_streak - 1)
            else:
                current_streak = 0
            streaks.append(current_streak)
            max_win = max(max_win, current_streak)
            max_loss = min(max_loss, current_streak)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Streak", f"{streaks[-1] if streaks else 0:+d}")
        c2.metric("Best Streak", f"+{max_win}")
        c3.metric("Worst Streak", str(max_loss))


def _render_edge_analysis(df: pd.DataFrame) -> None:
    edge_df = df[df["edge"].notna()].copy()

    if edge_df.empty:
        st.info(
            "No bets with edge data recorded. Fill in model signals when recording bets to see this analysis."
        )
        return

    st.markdown("#### Edge vs Outcome")
    st.caption("Did higher-edge bets actually win more often?")

    edge_df["edge_bucket"] = pd.cut(
        edge_df["edge"],
        bins=[-1, 0, 0.05, 0.10, 0.20, 0.50, 2.0],
        labels=["Negative", "0-5%", "5-10%", "10-20%", "20-50%", "50%+"],
    )

    edge_stats = (
        edge_df.groupby("edge_bucket", observed=True)
        .agg(
            total=("id", "count"),
            wins=("result", lambda x: (x == "win").sum()),
            pnl=("pnl", "sum"),
            avg_odds=("odds_decimal", "mean"),
        )
        .reset_index()
    )
    edge_stats["win_rate"] = edge_stats["wins"] / edge_stats["total"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=edge_stats["edge_bucket"].astype(str),
            y=edge_stats["win_rate"],
            name="Win Rate %",
            marker_color=COLORS["accent"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=edge_stats["edge_bucket"].astype(str),
            y=edge_stats["pnl"],
            name="P&L €",
            mode="lines+markers",
            yaxis="y2",
            marker={"color": COLORS["warning"], "size": 8},
            line={"color": COLORS["warning"]},
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_secondary"]},
        margin={"l": 0, "r": 40, "t": 10, "b": 0},
        height=280,
        legend={"orientation": "h", "y": -0.25},
        yaxis={"title": "Win Rate %", "gridcolor": COLORS["border"]},
        yaxis2={"title": "P&L €", "overlaying": "y", "side": "right"},
        xaxis={"title": "Edge Bucket", "showgrid": False},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model Calibration")
    st.caption("How well does your model probability predict actual outcomes?")
    edge_df["prob_bucket"] = pd.cut(
        edge_df["model_probability"],
        bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"],
    )
    cal_stats = (
        edge_df.groupby("prob_bucket", observed=True)
        .agg(
            total=("id", "count"),
            actual_wins=("result", lambda x: (x == "win").sum()),
        )
        .reset_index()
    )
    cal_stats["actual_rate"] = cal_stats["actual_wins"] / cal_stats["total"] * 100
    cal_stats["predicted_mid"] = [15, 35, 45, 55, 65, 75, 90][: len(cal_stats)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect",
            line={"dash": "dash", "color": COLORS["text_secondary"], "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cal_stats["predicted_mid"],
            y=cal_stats["actual_rate"],
            mode="markers+text",
            name="Actual",
            marker={"color": COLORS["accent"], "size": 12},
            text=cal_stats["total"].apply(lambda x: f"n={x}"),
            textposition="top center",
            textfont={"size": 9, "color": COLORS["text_secondary"]},
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text_secondary"]},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=280,
        xaxis={
            "title": "Predicted Win %",
            "gridcolor": COLORS["border"],
            "range": [0, 100],
        },
        yaxis={
            "title": "Actual Win %",
            "gridcolor": COLORS["border"],
            "range": [0, 100],
        },
        legend={"orientation": "h", "y": -0.25},
    )
    st.plotly_chart(fig, use_container_width=True)
