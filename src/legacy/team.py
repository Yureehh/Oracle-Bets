# Housekeeping
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Class Definition
class Team:
    team_data = pd.read_csv(
        Path.cwd().parent.joinpath("data", "processed", "flattened_teams.csv"),
        usecols=["teamname", "team_elo", "egpm_dominance"]
    )
    player_data = pd.read_csv(
        Path.cwd().parent.joinpath("data", "processed", "flattened_players.csv")
    )

    def __init__(self, name: Optional[str] = None, side: Optional[str] = None, top: Optional[str] = None,
                 jng: Optional[str] = None, mid: Optional[str] = None, bot: Optional[str] = None,
                 sup: Optional[str] = None):
        self.name = name
        self.side = side
        self.top = top
        self.jng = jng
        self.mid = mid
        self.bot = bot
        self.sup = sup
        self.warning = ""

        self.team_exists = False
        if not self.side:
            self.side = "Blue"

        lower_name = str(self.name).lower()
        lower_teamname = Team.team_data.teamname.str.lower()
        team_data = Team.team_data[
            lower_teamname.isin([lower_name])
        ].reset_index(drop=True)
        player_data = Team.player_data

        if len(team_data.index) > 0:
            roster = self._get_last_roster(player_data)
            self.team_exists = True
            self.team_elo = team_data.team_elo.mean()
            self.team_egpm_dom = team_data.egpm_dominance.mean()
        elif lower_name in ["first 5", "second 5"]:
            pass
        else:
            logging.warning(
                f"Team `{str(self.name)}` not found in database. No team data was used."
            )

        positions = ['bot', 'jng', 'mid', 'sup', 'top']
        for position in positions:
            if not getattr(self, position):
                setattr(self, position, roster[positions.index(position)])

        self._update_players(player_data)
        self._validate_data(player_data)
        self._calculate_player_stats(player_data)

    def _get_last_roster(self, player_data: pd.DataFrame) -> list:
        lower_name = str(self.name).lower()
        player_data = player_data[
            player_data.teamname.str.lower().isin([lower_name])
        ].reset_index(drop=True)
        last_played = (
            player_data.sort_values(["date", "teamname", "position"])
            .drop_duplicates(
                subset=["teamname", "position"], keep="last", ignore_index=True
            )
            .reset_index()
        )
        last_starting = list(last_played.playername.unique())

        return last_starting

    def _update_players(self, player_data):
        players = [self.top, self.jng, self.mid, self.bot, self.sup]
        players = [s.lower() for s in players]

        data = player_data[player_data.playername.str.lower().isin(players)]
        data = (
            data.sort_values(["playername", "date"])
            .groupby(["playername"])
            .tail(1)
            .reset_index(drop=True)
        )
        if len(data) > 5:
            most_common_league = data.league.mode().iloc[0]
            data = data[data["league"] == most_common_league].reset_index(drop=True)
        if len(data) < 5:
            df_players = list(data.playername.str.lower().unique())
            diff = np.setdiff1d(players, df_players)
            substitutes = []
            for d in diff:
                substitute = {
                    "date": "1/1/2022 23:59",
                    "teamname": "Null",
                    "position": "Null",
                    "playername": d,
                    "player_elo": 1100,
                    "trueskill_mu": 21,
                    "trueskill_sigma": 8,
                    "egpm_dominance": 198,
                    "blue_side_ema_after": 0.4,
                    "red_side_ema_after": 0.4,
                }
                substitutes.append(substitute)
            data = data.append(substitutes, ignore_index=True)
            self.warning += (
                f"\n WARNING: {str(diff)} not found in database. "
                f"Substitute values were used."
            )
        elif len(data) > 5:
            raise ValueError(
                f"Team cannot have more than 5 player values. \n \n {data}"
            )

        self.player_elo = data.player_elo.mean()
        self.player_trueskill_mu = data.trueskill_mu.sum()
        self.player_trueskill_sigma = data.trueskill_sigma.to_list()
        self.player_egpm_dom = data.egpm_dominance.sum()
        if lower_name in ["first 5", "second 5"]:
            self.team_egpm_dom = self.player_egpm_dom
        self.side_win_rate = (
            data.blue_side_ema_after.mean()
            if self.side.lower() == "blue"
            else data.red_side_ema_after.mean()
        )

    def _validate_data(self, player_data):
        positions = ['bot', 'jng', 'mid', 'sup', 'top']
        for position in positions:
            if not getattr(self, position):
                setattr(self, position, roster[positions.index(position)])

    def _calculate_player_stats(self, player_data):
        self.player_elo = data.player_elo.mean()
        self.player_trueskill_mu = data.trueskill_mu.sum()
        self.player_trueskill_sigma = data.trueskill_sigma.to_list()
        self.player_egpm_dom = data.egpm_dominance.sum()
        if lower_name in ["first 5", "second 5"]:
            self.team_egpm_dom = self.player_egpm_dom
        self.side_win_rate = (
            data.blue_side_ema_after.mean()
            if self.side.lower() == "blue"
            else data.red_side_ema_after.mean()
        )


if __name__ in ("__main__", "__builtin__", "builtins"):
    print(Team("Oh My God"))
