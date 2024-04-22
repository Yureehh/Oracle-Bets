from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.performance_features.entity_stats import (
    apply_entity_ema_std,
    calculate_entity_kda,
    calculate_kill_participation,
    elaborate_stats,
    select_columns_for_entity,
)

player_data = """date,gameid,side,position,league,patch,playername,playerid,teamname,teamid,result,kills,deaths,assists,total_cs,egpm,earnedgoldshare,damagetochampions,dpm,damageshare,damagetakenperminute,wardsplaced,wpm,wardskilled,wcpm,controlwardsbought,visionscore,vspm,totalgold,monsterkills,minionkills,gamelength,ckpm,cspm,team_kpm,goldat15,xpat15,csat15,killsat15,assistsat15,deathsat15,opp_killsat15,opp_assistsat15,opp_deathsat15,golddiffat15,xpdiffat15,csdiffat15,opponentteam,opponentteamid,opponent_egpm,opponentplayername,opponentplayerid
2024-02-14 19:53:48,LOLTMNT01_55928,Blue,bot,LIT,14.02,Zamulek,oe:player:694ee794a9d3a8c3fbd658becb89769,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,6,1,3,372.0,298.9064,0.256102,27282.0,678.0944,0.306151,325.0787,17.0,0.4225,11.0,0.2734,6.0,61.0,1.5162,17226,24.0,348.0,40.233333333333334,0.6214,9.2461,0.3728,5527.0,6076.0,143.0,1.0,1.0,0.0,0.0,0.0,1.0,275.0,528.0,-8.0,Atleta Esport,oe:team:2857b1b013cbea649c5c610a18efa44,326.1972,PiOk,oe:player:956b4d2f499b1d362f693f82b8e652b
2024-02-14 19:53:48,LOLTMNT01_55928,Blue,jng,LIT,14.02,Rhilech,oe:player:cbe20c64274a87a21e5bdac1ccc3e77,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,1,3,7,236.0,176.222,0.150985,8488.0,210.9693,0.0952499,723.6288,10.0,0.2486,17.0,0.4225,9.0,54.0,1.3422,12290,215.0,21.0,40.233333333333334,0.6214,5.8658,0.3728,5259.0,6323.0,113.0,1.0,2.0,0.0,0.0,0.0,1.0,571.0,777.0,5.0,Atleta Esport,oe:team:2857b1b013cbea649c5c610a18efa44,262.2204,Kikis,oe:player:48e06b61418cf7d3391acd04e2462a6
2024-02-14 19:53:48,LOLTMNT01_55928,Blue,mid,LIT,14.02,SlowQ,oe:player:613a571a4b0b674547093130f2e4093,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,4,1,4,369.0,287.9702,0.246732,18824.0,467.8708,0.211237,342.7009,33.0,0.8202,19.0,0.4722,18.0,73.0,1.8144,16786,0.0,369.0,40.233333333333334,0.6214,9.1715,0.3728,6500.0,8202.0,151.0,2.0,1.0,0.0,0.0,0.0,2.0,1948.0,1405.0,31.0,Atleta Esport,oe:team:2857b1b013cbea649c5c610a18efa44,261.4002,Sebekx,oe:player:b8c5bfbff49a0f1459f1c15d824fde7
2024-02-14 19:53:48,LOLTMNT01_55928,Blue,sup,LIT,14.02,Efias,oe:player:962e7d6bbb1630bed16179c25d185e6,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,2,5,9,40.0,104.7887,0.0897805,4939.0,122.7589,0.055424,497.4731,74.0,1.8393,14.0,0.348,17.0,138.0,3.43,9416,0.0,40.0,40.233333333333334,0.6214,0.9942,0.3728,3876.0,4044.0,32.0,1.0,3.0,0.0,0.0,0.0,1.0,615.0,336.0,4.0,Atleta Esport,oe:team:2857b1b013cbea649c5c610a18efa44,102.179,TasteLess,oe:player:09a8a47a27251a4e9a8cedcd76ea518
2024-02-14 19:53:48,LOLTMNT01_55928,Blue,top,LIT,14.02,Empyros,oe:player:9db3f170213f562a9613fe4f99d7dfa,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,2,0,6,384.0,299.2543,0.2564,29580.0,735.2113,0.331938,380.5302,23.0,0.5717,6.0,0.1491,10.0,65.0,1.6156,17240,40.0,344.0,40.233333333333334,0.6214,9.5443,0.3728,6037.0,8396.0,140.0,1.0,0.0,0.0,0.0,0.0,1.0,1247.0,494.0,10.0,Atleta Esport,oe:team:2857b1b013cbea649c5c610a18efa44,193.024,EYLIPH,oe:player:1f2f07038e8a9aa12fd2144db8f3ef6
024-02-15 18:50:24,LOLTMNT01_58121,Blue,bot,LIT,14.02,Zamulek,oe:player:694ee794a9d3a8c3fbd658becb89769,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,5,0,12,267.0,295.1004,0.244924,22839.0,625.1551,0.260996,459.6898,19.0,0.5201,9.0,0.2464,9.0,66.0,1.8066,15528,24.0,243.0,36.53333333333333,0.6569,7.3084,0.5201,5408.0,6436.0,133.0,1.0,2.0,0.0,1.0,0.0,0.0,-64.0,545.0,-6.0,EKO Academy,oe:team:cd7cc0de81ba081ae734a9c0aa5bba6,259.5712,Tyrone,oe:player:34dd70244ef257a4b5f7df18076132b
2024-02-15 18:50:24,LOLTMNT01_58121,Blue,jng,LIT,14.02,Rhilech,oe:player:cbe20c64274a87a21e5bdac1ccc3e77,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,2,1,13,194.0,199.927,0.165931,5534.0,151.4781,0.0632407,850.5657,10.0,0.2737,13.0,0.3558,9.0,49.0,1.3412,12051,174.0,20.0,36.53333333333333,0.6569,5.3102,0.5201,4380.0,5288.0,97.0,0.0,2.0,0.0,1.0,1.0,2.0,75.0,1070.0,24.0,EKO Academy,oe:team:cd7cc0de81ba081ae734a9c0aa5bba6,144.1697,Ryujin,oe:player:0e6c8380e804e9a22a6e0d044dfa1e1
2024-02-15 18:50:24,LOLTMNT01_58121,Blue,mid,LIT,14.02,SlowQ,oe:player:613a571a4b0b674547093130f2e4093,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,6,1,10,320.0,298.3029,0.247582,31779.0,869.8631,0.36316,559.0785,22.0,0.6022,6.0,0.1642,11.0,45.0,1.2318,15645,0.0,320.0,36.53333333333333,0.6569,8.7591,0.5201,6092.0,8279.0,157.0,2.0,1.0,0.0,0.0,0.0,2.0,1702.0,1597.0,48.0,EKO Academy,oe:team:cd7cc0de81ba081ae734a9c0aa5bba6,235.292,Berkan,oe:player:ff5c229246dedaceb827b432d573e48
2024-02-15 18:50:24,LOLTMNT01_58121,Blue,sup,LIT,14.02,Efias,oe:player:962e7d6bbb1630bed16179c25d185e6,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,1,2,14,41.0,122.3266,0.101524,4995.0,136.7245,0.0570811,453.9964,56.0,1.5328,17.0,0.4653,14.0,124.0,3.3942,9216,0.0,41.0,36.53333333333333,0.6569,1.1223,0.5201,3178.0,3850.0,23.0,0.0,2.0,2.0,0.0,1.0,0.0,-61.0,-247.0,-2.0,EKO Academy,oe:team:cd7cc0de81ba081ae734a9c0aa5bba6,99.7172,Colden,oe:player:bda5bb29fc4ac7bc4469089be5bd8e0
2024-02-15 18:50:24,LOLTMNT01_58121,Blue,top,LIT,14.02,Empyros,oe:player:9db3f170213f562a9613fe4f99d7dfa,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,5,1,8,324.0,289.2153,0.240039,22360.0,612.0438,0.255522,985.0182,14.0,0.3832,11.0,0.3011,4.0,30.0,0.8212,15313,4.0,320.0,36.53333333333333,0.6569,8.8686,0.5201,6033.0,7751.0,138.0,2.0,2.0,0.0,0.0,1.0,1.0,1034.0,449.0,1.0,EKO Academy,oe:team:cd7cc0de81ba081ae734a9c0aa5bba6,227.2172,lobellan,oe:player:e856a4c24d5611c7f42c5acaefcf841"""

team_data = """date,gameid,side,league,patch,teamname,teamid,result,kills,deaths,assists,egpm,gamelength,ckpm,team_kpm,firstblood,dragons,barons,towers,goldat15,xpat15,csat15,golddiffat15,xpdiffat15,csdiffat15,opponentteam,opponentteamid,opponent_egpm
2024-02-29 18:03:25,LOLTMNT02_60416,Blue,LIT,14.03,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,15,6,26,1337.7632,22.8,0.9211,0.6579,1.0,3.0,1.0,7.0,29030.0,31992.0,543.0,4621.0,3299.0,84.0,ENEMI3S,oe:team:4e32226d1afb63dde4a39d02cb1cd0f,915.6579
2024-02-23 18:52:44,LOLTMNT02_55402,Red,LIT,14.03,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,1,22,16,51,1272.5916,38.2,0.9948,0.5759,1.0,4.0,1.0,10.0,24496.0,28109.0,489.0,-1707.0,-1173.0,-4.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,1005.9162
2024-02-22 20:53:56,LOLTMNT02_53906,Red,LIT,14.03,Dsyre Esports,oe:team:f05dcb5635b1d32962510e26e412218,0,17,27,34,1212.307,35.18333333333333,1.2506,0.4832,0.0,2.0,0.0,7.0,24251.0,28323.0,509.0,-2335.0,-852.0,22.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1429.37"""


class TestEntityStats:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup common test variables."""
        self.sample_player_data = pd.read_csv(StringIO(player_data))
        self.sample_team_data = pd.read_csv(StringIO(team_data))

    def test_calculate_entity_kda(self):
        player_data = calculate_entity_kda(self.sample_player_data)
        team_data = calculate_entity_kda(self.sample_team_data)

        assert "kda" in player_data.columns
        assert "kda" in team_data.columns
        assert player_data["kda"].dtype == float
        assert team_data["kda"].dtype == float
        assert player_data["kda"].min() >= 0
        assert team_data["kda"].min() >= 0

        # Calculate expected KDA values for player and team data
        expected_player_kda = np.where(
            player_data["deaths"] > 0,
            (player_data["kills"] + player_data["assists"]) / player_data["deaths"],
            player_data["kills"] + player_data["assists"],
        )

        expected_team_kda = np.where(
            team_data["deaths"] > 0,
            (team_data["kills"] + team_data["assists"]) / team_data["deaths"],
            team_data["kills"] + team_data["assists"],
        )

        # Assert calculated KDA is equal to expected KDA values
        assert np.allclose(player_data["kda"], expected_player_kda, equal_nan=True)
        assert np.allclose(team_data["kda"], expected_team_kda, equal_nan=True)

    def test_calculate_kill_participation(self):
        player_data = calculate_kill_participation(self.sample_player_data)

        assert "kill_participation" in player_data.columns
        assert player_data["kill_participation"].dtype == float
        assert player_data["kill_participation"].min() >= 0

        # Calculate expected kill participation values for player data
        expected_kill_participation = (
            self.sample_player_data["kills"] + self.sample_player_data["assists"]
        ) / self.sample_player_data.groupby(["gameid", "teamid"])["kills"].transform("sum")

        # Assert calculated kill participation is equal to expected values
        assert np.allclose(
            player_data["kill_participation"],
            expected_kill_participation,
            equal_nan=True,
        )

    def test_elaborate_stats(self):
        player_data = elaborate_stats(self.sample_player_data, "player")
        team_data = elaborate_stats(self.sample_team_data, "team")

        assert "kda" in player_data.columns
        assert "kill_participation" in player_data.columns
        assert "kda" in team_data.columns
        assert "kill_participation" not in team_data.columns
        assert player_data["kda"].dtype == float
        assert player_data["kill_participation"].dtype == float
        assert team_data["kda"].dtype == float
        assert player_data["kda"].min() >= 0
        assert player_data["kill_participation"].min() >= 0
        assert team_data["kda"].min() >= 0

    def test_select_columns_for_entity(self):
        player_columns = select_columns_for_entity("player")
        team_columns = select_columns_for_entity("team")

        true_team_columns = [
            "gamelength",
            "kills",
            "deaths",
            "assists",
            "kda",
            "goldat10",
            "xpat10",
            "csat10",
            "golddiffat10",
            "xpdiffat10",
            "csdiffat10",
            "goldat15",
            "xpat15",
            "csat15",
            "golddiffat15",
            "xpdiffat15",
            "csdiffat15",
            "egpm",
            "ckpm",
            "firstblood",
            "dragons",
            "void_grubs",
            "heralds",
            "barons",
            "elders",
            "towers",
            "turretplates",
            "teamkills",
            "teamdeaths",
            "gspd",
            "team_kpm",
        ]

        true_player_columns = [
            "gamelength",
            "kills",
            "deaths",
            "assists",
            "kda",
            "goldat10",
            "xpat10",
            "csat10",
            "golddiffat10",
            "xpdiffat10",
            "csdiffat10",
            "goldat15",
            "xpat15",
            "csat15",
            "golddiffat15",
            "xpdiffat15",
            "csdiffat15",
            "egpm",
            "ckpm",
            "damageshare",
            "kill_participation",
            "total_cs",
            "earnedgoldshare",
            "damagetochampions",
            "damagetakenperminute",
            "damagemitigatedperminute",
            "controlwardsbought",
            "visionscore",
            "totalgold",
            "gpr",
            "killsat15",
            "assistsat15",
            "deathsat15",
            "dpm",
            "wpm",
            "wcpm",
            "vspm",
            "cspm",
            "gold_efficiency",
            "xp_efficiency",
        ]

        assert set(player_columns) == set(true_player_columns)
        assert set(team_columns) == set(true_team_columns)

    def test_apply_entity_ema_std(self):
        player_data = apply_entity_ema_std(self.sample_player_data, "playerid", ["kills", "deaths"], 5)
        team_data = apply_entity_ema_std(self.sample_team_data, "teamid", ["kills", "deaths"], 5)

        assert "ema_kills_before" in player_data.columns
        assert "ema_kills_after" in player_data.columns
        assert "ema_deaths_before" in player_data.columns
        assert "ema_deaths_after" in player_data.columns

        assert "ema_kills_before" in team_data.columns
        assert "ema_kills_after" in team_data.columns
        assert "ema_deaths_before" in team_data.columns
        assert "ema_deaths_after" in team_data.columns
