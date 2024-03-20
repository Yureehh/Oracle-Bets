from io import StringIO

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.performance_features.egpm import (
    calculate_dominance_metrics,
    calculate_egpm_dominance_ratios,
    egpm_model,
)

team_data = """date,gameid,side,league,patch,teamname,teamid,result,kills,deaths,assists,egpm,gamelength,ckpm,team_kpm,firstblood,dragons,barons,towers,goldat15,xpat15,csat15,golddiffat15,xpdiffat15,csdiffat15,opponentteam,opponentteamid,opponent_egpm,elo_pre_match,elo_pre_match_opponent,elo_win_likelihood,elo,elo_opponent,pl_pre_match_mu,pl_pre_match_sigma,pl_win_likelihood,pl_mu,pl_sigma,trueskill_sum_mu,trueskill_sigma_squared,trueskill_opponent_sum_mu,trueskill_opponent_sigma_squared,trueskill_diff
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,LIT,14.01,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,19,7,47,1259.5975,29.816666666666663,0.872,0.6372,0.0,3.0,2.0,8.0,26226.0,31140.0,498.0,2484.0,2171.0,-15.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,920.5925,1382.428368971944,1006.5206918682358,0.8969604185340606,1385.7256355788543,1003.2234252613257,41.21685425670362,2.3244201356251195,0.9791261268150178,41.308831055129744,2.3211406824661105,133.647,46.301,134.651,53.855,-0.02400000000000002
2024-01-25 18:05:49,LOLTMNT03_34547,Red,LIT,14.01,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,7,19,11,920.5925,29.816666666666663,0.872,0.2348,1.0,1.0,0.0,2.0,23742.0,28969.0,513.0,-2484.0,-2171.0,15.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1259.5975,1006.5206918682358,1382.428368971944,0.10303958146593933,1003.2234252613257,1385.7256355788543,27.38688564124927,2.452479178244336,0.02087387318498224,27.28450850314877,2.447980340781173,134.651,53.855,133.647,46.301,0.02400000000000002"""

player_data = """date,gameid,side,position,league,patch,playername,playerid,teamname,teamid,result,kills,deaths,assists,total_cs,egpm,earnedgoldshare,damagetochampions,dpm,damageshare,damagetakenperminute,wardsplaced,wpm,wardskilled,wcpm,controlwardsbought,visionscore,vspm,totalgold,monsterkills,minionkills,gamelength,ckpm,cspm,team_kpm,goldat15,xpat15,csat15,killsat15,assistsat15,deathsat15,opp_killsat15,opp_assistsat15,opp_deathsat15,golddiffat15,xpdiffat15,csdiffat15,opponentteam,opponentteamid,opponent_egpm,opponentplayername,opponentplayerid,KDA,gold_efficiency,xp_efficiency,kill_participation,kills_volatility,deaths_volatility,kills_growth,deaths_growth,position_bot,position_jng,position_mid,position_sup,position_top,elo_pre_match,elo_pre_match_opponent,elo_win_likelihood,elo,elo_opponent,pl_pre_match_mu,pl_pre_match_sigma,pl_win_likelihood,pl_mu,pl_sigma,trueskill_mu,trueskill_sigma,trueskill_opponent_mu,trueskill_opponent_sigma
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,bot,LIT,14.01,odi11,oe:player:7705566e63eedc7bd6c4c72d43c9104,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,7,0,8,272.0,353.0911,0.280315,18382.0,616.5008,0.271477,363.0184,9.0,0.3018,7.0,0.2348,2.0,37.0,1.2409,14453,21.0,251.0,29.816666666666663,0.872,9.1224,0.6372,6825.0,7071.0,142.0,3.0,1.0,0.0,0.0,0.0,4.0,2076.0,1848.0,17.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,186.5735,Orion,oe:player:915901c2ffafe3ee9245c3ef30ec5b4,15.0,484.72889882615993,9.122414756847402,0.7894736842105263,3.209361307176242,2.0,2.6,3.0,True,False,False,False,False,933.6175790705223,1067.7104736043957,0.3160680793535434,955.503400531209,1045.824652143709,25.379715122567042,5.361198899847011,0.6662400759438727,25.96654314180272,5.326834878828627,25.545,2.782,22.741,2.204
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,jng,LIT,14.01,Lotuss,oe:player:81317166070e5727aa15670f1d996b3,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,0,2,14,159.0,198.3454,0.157462,12867.0,431.5372,0.190028,1089.2566,8.0,0.2683,3.0,0.1006,6.0,31.0,1.0397,9839,142.0,17.0,29.816666666666663,0.872,5.3326,0.6372,4683.0,5333.0,89.0,0.0,4.0,0.0,0.0,1.0,0.0,-51.0,-32.0,-13.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,245.9698,nicolaiy,oe:player:ae50e71f31d35aee5c589498e3c1b17,7.0,329.98323085522645,5.332588038010062,0.7368421052631579,3.7013511046643495,1.7888543819998315,2.8,1.8,False,True,False,False,False,1004.1373585953121,1058.5671087313883,0.42230388209814457,1022.6236343681715,1040.0808329585288,24.888291172458896,5.586355223841543,0.6662400759438727,25.525432553409125,5.547356026747867,25.334,2.986,25.208,3.479
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,mid,LIT,14.01,Nan0,oe:player:a2e1e40c83fe5fd9efb5f50a67d9194,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,8,1,6,254.0,316.8362,0.251532,22873.0,767.1213,0.337803,455.1817,12.0,0.4025,8.0,0.2683,6.0,35.0,1.1738,13372,7.0,247.0,29.816666666666663,0.872,8.5187,0.6372,6078.0,7627.0,146.0,3.0,1.0,0.0,0.0,0.0,2.0,938.0,729.0,9.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,163.9687,Vigil,oe:player:c08f75d1d70bf068013b32f60fe96a5,14.0,448.47400782560095,8.518725544997206,0.7368421052631579,3.1622776601683795,0.5477225575051653,3.0,1.4,False,False,True,False,False,948.476156527719,972.7059244683551,0.4651869805034156,965.5901731516097,955.5919078444643,22.775693312344803,7.065294335420794,0.6662400759438727,23.794760965099396,6.985469072435013,22.137,4.507,30.875,3.637
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,sup,LIT,14.01,Click,oe:player:1e249a5688255bde7f7213b5d257b3c,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,3,0,13,34.0,199.3181,0.158234,6673.0,223.801,0.0985512,248.4181,61.0,2.0458,11.0,0.3689,10.0,116.0,3.8904,9868,0.0,34.0,29.816666666666663,0.872,1.1403,0.6372,4419.0,3997.0,20.0,1.0,4.0,0.0,0.0,0.0,1.0,1153.0,843.0,-4.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,82.1017,Shredder,oe:player:4515c41bfcadfcf2e3c5880242f99a2,16.0,330.9558412520962,1.1403018446059252,0.8421052631578947,2.16794833886788,1.3038404810405286,1.8,0.8,False,False,False,True,False,1384.5360792794074,913.9593130847871,0.9375472928385887,1386.5345659085726,911.9608264556219,37.43110531467705,4.158731303385138,0.6662400759438727,37.784270719453744,4.143241404405361,31.312,2.18,23.368,3.802
2024-01-25 18:05:49,LOLTMNT03_34547,Blue,top,LIT,14.01,Color,oe:player:cb5feb1b7314637725a2e73bdc9f729,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,1,1,4,6,179.0,192.0402,0.152457,6916.0,231.9508,0.10214,583.5998,11.0,0.3689,2.0,0.0671,1.0,16.0,0.5366,9651,0.0,179.0,29.816666666666663,0.872,6.0034,0.6372,4221.0,7112.0,101.0,0.0,0.0,3.0,3.0,0.0,0.0,-1632.0,-1217.0,-24.0,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,242.0123,Matixx,oe:player:89bee56a2a53bdf22737c6d43eedb9c,1.75,323.6780324203466,6.003353828954724,0.3684210526315789,1.2247448713915878,1.3416407864998723,1.0,3.4,False,False,False,False,True,1241.7160120689828,1089.4168502316238,0.7061388368502374,1251.1195692897752,1080.0132930108314,31.03174721424957,4.0699800911991915,0.6662400759438727,31.370005689117882,4.055532268883179,29.319,2.14,32.459,3.035
2024-01-25 18:05:49,LOLTMNT03_34547,Red,bot,LIT,14.01,Orion,oe:player:915901c2ffafe3ee9245c3ef30ec5b4,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,0,5,3,227.0,186.5735,0.202659,14779.0,495.6624,0.288241,486.3387,10.0,0.3354,6.0,0.2012,5.0,17.0,0.5702,9488,0.0,227.0,29.816666666666663,0.872,7.6132,0.2348,4749.0,5223.0,125.0,0.0,0.0,4.0,3.0,1.0,0.0,-2076.0,-1848.0,-17.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,353.0911,odi11,oe:player:7705566e63eedc7bd6c4c72d43c9104,0.6,318.2112912241476,7.6131917272219125,0.42857142857142855,2.5884358211089564,1.7888543819998313,3.2,2.8,True,False,False,False,False,1067.7104736043957,933.6175790705223,0.6839319206464566,1045.824652143709,955.503400531209,24.288124436073993,4.363681189111976,0.3337599240561273,23.89930634018845,4.344620929818347,22.741,2.204,25.545,2.782
2024-01-25 18:05:49,LOLTMNT03_34547,Red,jng,LIT,14.01,nicolaiy,oe:player:ae50e71f31d35aee5c589498e3c1b17,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,3,1,2,204.0,245.9698,0.267179,9602.0,322.0347,0.187272,809.7149,6.0,0.2012,12.0,0.4025,5.0,36.0,1.2074,11259,179.0,25.0,29.816666666666663,0.872,6.8418,0.2348,4734.0,5365.0,102.0,0.0,1.0,0.0,0.0,4.0,0.0,51.0,32.0,13.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,198.3454,Lotuss,oe:player:81317166070e5727aa15670f1d996b3,5.0,377.60760201229743,6.841811067635551,0.7142857142857143,2.1679483388678804,2.073644135332772,2.2,2.6,False,True,False,False,False,1058.5671087313883,1004.1373585953121,0.5776961179018554,1040.0808329585288,1022.6236343681715,29.038278192012744,6.128323995380279,0.3337599240561273,28.271542257686274,6.07378360944747,25.208,3.479,25.334,2.986
2024-01-25 18:05:49,LOLTMNT03_34547,Red,mid,LIT,14.01,Vigil,oe:player:c08f75d1d70bf068013b32f60fe96a5,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,0,6,2,204.0,163.9687,0.178105,9822.0,329.4131,0.191563,560.995,10.0,0.3354,6.0,0.2012,2.0,18.0,0.6037,8814,0.0,204.0,29.816666666666663,0.872,6.8418,0.2348,5140.0,6898.0,137.0,0.0,0.0,2.0,3.0,1.0,0.0,-938.0,-729.0,-9.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,316.8362,Nan0,oe:player:a2e1e40c83fe5fd9efb5f50a67d9194,0.3333333333333333,295.6064840693125,6.841811067635551,0.2857142857142857,4.8166378315169185,1.9999999999999991,3.8,3.0,False,False,True,False,False,972.7059244683551,948.476156527719,0.5348130194965844,955.5919078444643,965.5901731516097,23.545191129481708,6.211687835662674,0.3337599240561273,22.757457318788468,6.154853543815418,30.875,3.637,22.137,4.507
2024-01-25 18:05:49,LOLTMNT03_34547,Red,sup,LIT,14.01,Shredder,oe:player:4515c41bfcadfcf2e3c5880242f99a2,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,0,5,3,36.0,82.1017,0.0891771,3493.0,117.1492,0.0681255,442.1688,46.0,1.5428,14.0,0.4695,11.0,94.0,3.1526,6373,0.0,36.0,29.816666666666663,0.872,1.2074,0.2348,3266.0,3154.0,24.0,0.0,0.0,1.0,1.0,4.0,0.0,-1153.0,-843.0,4.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,199.3181,Click,oe:player:1e249a5688255bde7f7213b5d257b3c,0.6,213.73951928451652,1.2073784237003915,0.42857142857142855,1.4832396974191324,1.2247448713915898,1.8,5.0,False,False,False,True,False,913.9593130847871,1384.5360792794074,0.06245270716141141,911.9608264556219,1386.5345659085726,21.98928538967822,6.123826831092967,0.3337599240561273,21.223674146230454,6.069408433495807,23.368,3.802,31.312,2.18
2024-01-25 18:05:49,LOLTMNT03_34547,Red,top,LIT,14.01,Matixx,oe:player:89bee56a2a53bdf22737c6d43eedb9c,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,0,4,2,1,237.0,242.0123,0.26288,13577.0,455.3494,0.264798,659.6646,10.0,0.3354,2.0,0.0671,0.0,20.0,0.6708,11141,8.0,229.0,29.816666666666663,0.872,7.9486,0.2348,5853.0,8329.0,125.0,3.0,0.0,0.0,0.0,0.0,3.0,1632.0,1217.0,24.0,Macko Esports,oe:team:a2e63aee03585f0b37ff4cad936e349,192.0402,Color,oe:player:cb5feb1b7314637725a2e73bdc9f729,2.5,373.65008384572394,7.9485746226942435,0.7142857142857143,1.8165902124584954,1.2247448713915885,1.6,3.0,False,False,False,False,True,1089.4168502316238,1241.7160120689828,0.2938611631497626,1080.0132930108314,1251.1195692897752,33.268171774377244,5.15353507444175,0.3337599240561273,32.72591361050416,5.121476893984263,32.459,3.035,29.319,2.14"""


class TestEgpm:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup common test variables."""
        self.sample_player_data = pd.read_csv(StringIO(player_data))
        self.sample_team_data = pd.read_csv(StringIO(team_data))

    def test_calculate_egpm_dominance_ratios(self):
        player_data = self.sample_player_data
        rank = player_data["trueskill_mu"]
        opp_rank = player_data["trueskill_opponent_mu"]
        player_data = calculate_egpm_dominance_ratios(player_data, rank, opp_rank)

        assert "egpm_dominance_ratio" in player_data.columns
        assert "egpm_opp_dominance_ratio" in player_data.columns

        team_data = self.sample_team_data
        rank = team_data["trueskill_sum_mu"]
        opp_rank = team_data["trueskill_opponent_sum_mu"]
        team_data = calculate_egpm_dominance_ratios(team_data, rank, opp_rank)

        assert "egpm_dominance_ratio" in team_data.columns
        assert "egpm_opp_dominance_ratio" in team_data.columns

    def test_calculate_dominance_metrics(self):
        team_data = self.sample_team_data
        player_data = self.sample_player_data

        team_data = calculate_egpm_dominance_ratios(
            team_data,
            team_data["trueskill_sum_mu"],
            team_data["trueskill_opponent_sum_mu"],
        )
        player_data = calculate_egpm_dominance_ratios(
            player_data,
            player_data["trueskill_mu"],
            player_data["trueskill_opponent_mu"],
        )

        team_data = calculate_dominance_metrics(team_data, "teamid")
        assert "egpm_dominance_ratio_ema_before" in team_data.columns
        assert "egpm_opp_dominance_ratio_ema_before" in team_data.columns
        assert "egpm_dominance_ratio_ema_after" in team_data.columns
        assert "egpm_opp_dominance_ratio_ema_after" in team_data.columns
        assert "egpm_dominance_diff" in team_data.columns

        player_data = calculate_dominance_metrics(player_data, "playerid")
        assert "egpm_dominance_ratio_ema_before" in player_data.columns
        assert "egpm_opp_dominance_ratio_ema_before" in player_data.columns
        assert "egpm_dominance_ratio_ema_after" in player_data.columns
        assert "egpm_opp_dominance_ratio_ema_after" in player_data.columns
        assert "egpm_dominance_diff" in player_data.columns

    def test_egpm_model(self):
        team_data = self.sample_team_data
        player_data = self.sample_player_data

        team_data = egpm_model(team_data, "team", False)
        assert "egpm_dominance_log_win_perc" in team_data.columns

        player_data = egpm_model(player_data, "player", False)
        assert "egpm_dominance_log_win_perc" in player_data.columns
