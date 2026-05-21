"""LoL-specific Discord prediction and profile helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import lol_bets.inference.match_predictor as match_predictor_module
import numpy as np
from lol_bets.inference.team import Team
from oracle_bets_core.io_utils import parquet_loader
from oracle_bets_core.paths import FLATTENED_PLAYERS, FLATTENED_TEAMS
from oracle_bets_core.pd import pd

from oracle_bets_discord.formatting import (
    CONFIG,
    MESSAGE_LIMIT,
    dataframe_to_markdown,
    handle_command_error,
)
from oracle_bets_discord.predictions.best_ofs import BestOfs

# ── config & constants ──────────────────────────────────────────────────── #

_EMPTY_ROSTER: dict[str, str | None] = CONFIG.get("EMPTY_ROSTER", {}).copy() or {
    "top": None,
    "jng": None,
    "mid": None,
    "bot": None,
    "sup": None,
}
VALID_MATCH_TYPES: list[str] = ["bo1", "bo2", "bo3", "bo5"]
POSITIONS: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
WEEKS_FOR_DELAY: int = int(CONFIG.get("WEEKS_FOR_DELAY", 3))

PLEASE_PROVIDE_TEAMS = "Please provide both a blue and red team name."
TEAMS_MUST_BE_DIFFERENT = "The two teams must be different."

# ── lazy, single-shot predictor ─────────────────────────────────────────── #


@lru_cache(maxsize=1)
def get_match_predictor() -> match_predictor_module.MatchPredictor:
    return match_predictor_module.MatchPredictor()


# ── small helpers ───────────────────────────────────────────────────────── #


def get_empty_roster() -> dict[str, str | None]:
    """Copy of the empty-roster template."""
    return dict(_EMPTY_ROSTER.items())


# ── schedule / leagues formatting ───────────────────────────────────────── #


def format_leagues_message(leagues: list[str]) -> str:
    if not leagues:
        return "No leagues found. Please check the league names."
    formatted = "\n".join(f"- {lg}" for lg in leagues)
    return f"Leagues playing in the next 7 days are:\n{formatted}"


def format_schedule_message(schedule_df: pd.DataFrame) -> str:
    if schedule_df.empty or "league" not in schedule_df.columns:
        return "No upcoming matches found."

    parts: list[str] = []
    too_long_alert = "Message too long. Please specify a narrower filter."

    for league in np.sort(schedule_df["league"].unique()):
        block = format_league(schedule_df, league)
        est_len = len("\n".join(parts)) + len(block)
        if est_len > MESSAGE_LIMIT - len(too_long_alert):
            parts.append(too_long_alert)
            break
        parts.append(block)

    final = "\n".join(parts)
    return final or "No upcoming matches found. Double-check the league names."


def format_league(df: pd.DataFrame, league: str) -> str:
    league_df = df[df["league"] == league].head(5).copy()
    league_df["league"] = (
        league_df["league"].astype(str).str.split().str[:3].str.join(" ")
    )
    md = league_df.to_markdown(index=False)
    md = "\n".join(line.lstrip() for line in md.split("\n"))
    return f"Upcoming {league} Games (Next 5 Matches Within 7 Days):\n```{md}```\n\n"


# ── cached parquet reads for profiles ───────────────────────────────────── #


@lru_cache(maxsize=2)
def _parquet_cached(path: str) -> pd.DataFrame:
    return parquet_loader(Path(path))


def get_player_data(entity_name: str, players_path: Path) -> pd.DataFrame | None:
    try:
        df = _parquet_cached(str(players_path))
        if "playername" not in df.columns:
            msg = "Column 'playername' not found in players data."
            raise KeyError(msg)  # noqa: TRY301
        filtered = df[df["playername"].str.casefold() == entity_name.casefold()]
        return None if filtered.empty else filtered
    except Exception as e:
        msg = f"Error processing player data: {e}"
        raise ValueError(msg) from e


def get_team_data(entity_name: str, teams_path: Path) -> pd.DataFrame | None:
    try:
        df = _parquet_cached(str(teams_path))
        if "teamname" not in df.columns:
            msg = "Column 'teamname' not found in teams data."
            raise KeyError(msg)  # noqa: TRY301
        filtered = df[df["teamname"].str.casefold() == entity_name.casefold()]
        return None if filtered.empty else filtered
    except Exception as e:
        msg = f"Error processing team data: {e}"
        raise ValueError(msg) from e


# ── profile formatting ──────────────────────────────────────────────────── #


def format_player_profile(data: pd.DataFrame, truncate: bool = False) -> str:
    row = data.iloc[0]
    stats_names = [
        "Position",
        "Team",
        "Elo",
        "Glicko2 Score",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "K/D/A Ratio",
        "K/D/A at 15",
        "Gold Diff At 15",
        "CS Diff At 15",
        "XP Diff At 15",
        "CSPM",
        "DPM",
        "EGPM",
        "VSPM",
        "Damage Share",
        "XP Efficiency",
    ]
    if truncate:
        stats_names = stats_names[:8]

    def val(k: str, *alts: str) -> float:
        """Prefer EMA versions if present, else raw; returns np.nan if missing."""
        for a in (k, *alts):
            if a in row.index:
                return row[a]
        return np.nan

    kd15 = (
        f"{val('ema_killsat15'):.2f} / {val('ema_deathsat15'):.2f} / {val('ema_assistsat15'):.2f}"
        if all(
            x in row.index
            for x in ("ema_killsat15", "ema_deathsat15", "ema_assistsat15")
        )
        else "N/A"
    )

    stats_values = [
        str(row.get("position", "")).capitalize(),
        str(row.get("teamname", "N/A")),
        f"{val('elo'):.2f}" if "elo" in row.index else "N/A",
        f"{val('glicko2_mu', 'gl2_mu'):.2f}"
        if any(c in row.index for c in ("glicko2_mu", "gl2_mu"))
        else "N/A",
        f"{val('pl_mu'):.2f}" if "pl_mu" in row.index else "N/A",
        f"{val('trueskill_mu'):.2f}" if "trueskill_mu" in row.index else "N/A",
        f"{float(val('ema_blue_side', 'blue_side', 'side_blue') or 0) * 100:.2f}%",
        f"{float(val('ema_red_side', 'red_side', 'side_red') or 0) * 100:.2f}%",
        f"{float(val('ema_kda', 'kda') or 0):.2f}",
        kd15,
        f"{float(val('ema_golddiffat15', 'golddiffat15') or 0):.2f}",
        f"{float(val('ema_csdiffat15', 'csdiffat15') or 0):.2f}",
        f"{float(val('ema_xpdiffat15', 'xpdiffat15') or 0):.2f}",
        f"{float(val('ema_cspm', 'cspm') or 0):.2f}",
        f"{float(val('ema_dpm', 'dpm') or 0):.2f}",
        f"{float(val('ema_egpm', 'egpm') or 0):.2f}",
        f"{float(val('ema_vspm', 'vspm') or 0):.2f}",
        f"{float(val('ema_damageshare', 'damageshare') or 0) * 100:.2f}%",
        f"{float(val('ema_xp_efficiency', 'xp_efficiency') or 0):.2f}",
    ]
    if truncate:
        stats_values = stats_values[:8]

    df_out = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return dataframe_to_markdown(df_out)


def format_team_profile(data: pd.DataFrame) -> str:
    row = data.iloc[0]
    stats_names = [
        "Elo",
        "Glicko-2 Score",
        "Plackett-Luce Score",
        "TrueSkill Score",
        "League Elo",
        "Patch Win Rate",
        "Season Win Rate",
        "Blue Side Win Rate",
        "Red Side Win Rate",
        "AVG Gamelength in Minutes",
    ]
    stats_values = [
        f"{row.get('elo', np.nan):.2f}" if "elo" in row.index else "N/A",
        f"{row.get('glicko2_mu', np.nan):.2f}" if "glicko2_mu" in row.index else "N/A",
        f"{row.get('pl_mu', np.nan):.2f}" if "pl_mu" in row.index else "N/A",
        f"{row.get('trueskill_mu', np.nan):.2f}"
        if "trueskill_mu" in row.index
        else "N/A",
        f"{row.get('league_elo', np.nan):.2f}" if "league_elo" in row.index else "N/A",
        f"{float(row.get('ema_patch_win_rate', 0)) * 100:.2f}%",
        f"{float(row.get('ema_season_win_rate', 0)) * 100:.2f}%",
        f"{float(row.get('ema_blue_side', 0)) * 100:.2f}%",
        f"{float(row.get('ema_red_side', 0)) * 100:.2f}%",
        f"{float(row.get('ema_gamelength', row.get('gamelength', 0))):.2f}",
    ]
    df_out = pd.DataFrame({"Stat": stats_names, "Value": stats_values})
    return dataframe_to_markdown(df_out)


# ── async profile getters ───────────────────────────────────────────────── #


async def get_formatted_team_profile(team_name: str) -> tuple[str | None, str | None]:
    try:
        team_profile = get_team_data(team_name, FLATTENED_TEAMS)
        if team_profile is not None and not team_profile.empty:
            return format_team_profile(team_profile), None
        return None, f"Data for team '{team_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(e, "Team profile retrieval failed.")


async def get_formatted_player_profile(
    player_name: str, truncate: bool = False
) -> tuple[str | None, str | None]:
    try:
        player_profile = get_player_data(player_name, FLATTENED_PLAYERS)
        if player_profile is not None and not player_profile.empty:
            return format_player_profile(player_profile, truncate), None
        return None, f"Data for player '{player_name}' not found in the database."
    except Exception as e:
        return None, handle_command_error(e, "Player profile retrieval failed.")


# ── prediction helpers ──────────────────────────────────────────────────── #


def add_roster_to_output(output: str, blue_team: Team, red_team: Team) -> str:
    def fmt(team: Team) -> str:
        parts: list[str] = []
        for role in POSITIONS:
            name = team.roster.get(role)
            parts.append(str(name) if name else "N/A")
        return " \t-  \t".join(parts)

    output += "\n## Found Rosters\n"
    output += f"**Blue Team:**\t {fmt(blue_team)}\n"
    output += f"**Red Team:**\t {fmt(red_team)}"
    return output


def add_break_flags_to_output(
    output: str,
    break_blue_flag: bool,
    blue_team_name: str,
    break_red_flag: bool,
    red_team_name: str,
) -> str:
    if break_blue_flag:
        output += f"\n\nCAREFUL! {blue_team_name} has not played in the last {WEEKS_FOR_DELAY} weeks."
    if break_red_flag:
        output += f"\n\nCAREFUL! {red_team_name} has not played in the last {WEEKS_FOR_DELAY} weeks."
    return output


def process_roster(
    roster_str: str | None, positions: list[str] | None = None
) -> dict[str, str]:
    if not roster_str:
        msg = "Roster string cannot be empty."
        raise ValueError(msg)
    positions = positions or list(POSITIONS)
    players = [p.strip() for p in roster_str.split(",")]
    if len(players) != len(positions):
        msg = f"Roster does not contain the correct number of players: expected {len(positions)}, got {len(players)}."
        raise ValueError(msg)
    return dict(zip(positions, players, strict=False))


def strip_team_names(team1: str | None, team2: str | None) -> tuple[str, str]:
    return (team1 or "").strip(), (team2 or "").strip()


# ── main async prediction entrypoints ───────────────────────────────────── #


async def predict_and_format_result(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    account_for_side: bool,
) -> None:
    """Create teams, predict outcomes, and format result for bo1/bo3/bo5."""
    if match_type not in VALID_MATCH_TYPES:
        await ctx.send(
            content=f"Invalid match type: {match_type}. Please specify 'bo1', 'bo2', 'bo3', or 'bo5'."
        )
        return

    msg = await ctx.send(content="```Calculating win probabilities...```")
    try:
        blue_roster = (
            process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str) if red_roster_str else get_empty_roster()
        )
        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)
        # inactivity flags (robust to missing dates)
        today = pd.Timestamp.today().normalize()
        days_delay = int(WEEKS_FOR_DELAY) * 7

        def _parse_date(x) -> pd.Timestamp | pd.NaT:  # pyright: ignore[reportInvalidTypeForm]
            d = pd.to_datetime(x, errors="coerce")
            return d.normalize() if pd.notna(d) else pd.NaT

        blue_last = _parse_date(blue_team.team_stats.get("date"))
        red_last = _parse_date(red_team.team_stats.get("date"))
        break_blue_flag = bool(
            pd.notna(blue_last) and (today - blue_last).days >= days_delay
        )
        break_red_flag = bool(
            pd.notna(red_last) and (today - red_last).days >= days_delay
        )
        predictor = get_match_predictor()
        # Predict both orientations and average
        p1 = predictor.predict_match(
            blue_team, red_team, account_for_side=account_for_side
        )
        p2 = predictor.predict_match(
            red_team, blue_team, account_for_side=account_for_side
        )
        # p1: team1=blue; p2: team2=blue (flipped)  # noqa: ERA001
        blue_win = (p1["team1_win_probability"] + p2["team2_win_probability"]) / 2.0
        red_win = 1.0 - blue_win
        # series breakdown
        if match_type == "bo1":
            output = BestOfs.best_of_one(
                blue_team_name, blue_win, red_team_name, red_win
            )
        elif match_type == "bo2":
            output = BestOfs.best_of_two(
                blue_team_name, blue_win, red_team_name, red_win
            )
        elif match_type == "bo3":
            output = BestOfs.best_of_three(
                blue_team_name, blue_win, red_team_name, red_win
            )
        else:  # "bo5"
            output = BestOfs.best_of_five(
                blue_team_name, blue_win, red_team_name, red_win
            )
        output = add_roster_to_output(output, blue_team, red_team)
        output = add_break_flags_to_output(
            output, break_blue_flag, blue_team_name, break_red_flag, red_team_name
        )
        # stay under Discord limit
        await msg.edit(content=output[: MESSAGE_LIMIT - 1])

    except Exception as e:
        await msg.edit(
            content=handle_command_error(e, "Could not complete the prediction.")
        )


async def predict_and_format_props(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    account_for_side: bool,
) -> None:
    """Predict game props (gamelength, total kills, total towers) for a single game."""
    msg = await ctx.send(content="```Calculating prop predictions...```")
    try:
        blue_roster = (
            process_roster(blue_roster_str) if blue_roster_str else get_empty_roster()
        )
        red_roster = (
            process_roster(red_roster_str) if red_roster_str else get_empty_roster()
        )
        blue_team = Team(name=blue_team_name, side="Blue", roster=blue_roster)
        red_team = Team(name=red_team_name, side="Red", roster=red_roster)

        predictor = get_match_predictor()
        gamelength = predictor.predict_gamelength(
            blue_team, red_team, account_for_side=account_for_side
        )
        total_kills = predictor.predict_total_kills(
            blue_team, red_team, account_for_side=account_for_side
        )
        total_towers = predictor.predict_total_towers(
            blue_team, red_team, account_for_side=account_for_side
        )

        output = (
            f"**Prop Predictions (single game)**\n"
            f"- Expected game length: **{gamelength:.2f}** minutes\n"
            f"- Expected total kills: **{total_kills:.2f}**\n"
            f"- Expected total towers: **{total_towers:.2f}**\n"
        )
        await msg.edit(content=output[: MESSAGE_LIMIT - 1])
    except Exception as e:
        await msg.edit(
            content=handle_command_error(e, "Could not complete prop predictions.")
        )


async def validate_and_predict(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    match_type: str,
    side_consideration: bool,
):
    """Validates inputs and triggers prediction."""
    blue_team_name, red_team_name = strip_team_names(blue_team_name, red_team_name)
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE_TEAMS)
        return
    if blue_team_name.casefold() == red_team_name.casefold():
        await ctx.send(TEAMS_MUST_BE_DIFFERENT)
        return
    await predict_and_format_result(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        match_type,
        side_consideration,
    )


async def validate_and_predict_props(
    ctx,
    blue_team_name: str,
    red_team_name: str,
    blue_roster_str: str | None,
    red_roster_str: str | None,
    side_consideration: bool,
):
    blue_team_name, red_team_name = strip_team_names(blue_team_name, red_team_name)
    if not blue_team_name or not red_team_name:
        await ctx.send(PLEASE_PROVIDE_TEAMS)
        return
    if blue_team_name.casefold() == red_team_name.casefold():
        await ctx.send(TEAMS_MUST_BE_DIFFERENT)
        return
    await predict_and_format_props(
        ctx,
        blue_team_name,
        red_team_name,
        blue_roster_str,
        red_roster_str,
        side_consideration,
    )
