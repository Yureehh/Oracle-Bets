from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from moto import mock_s3

from src.data_ingest.oracles_elixir import OraclesElixir, get_opponent
from tests.test_oracle_elixir.s3_utils import create_s3_bucket, get_s3_session, upload_data_to_s3

REGION_NAME = "us-east-1"
BUCKET_NAME = "test_bucket"
DATA_FILE_NAME = "2023_LoL_esports_match_data_from_OraclesElixir.csv"


class TestOraclesElixir:
    @pytest.fixture
    def mock_raw_data(self):
        raw_data = """gameid,datacompleteness,url,league,year,split,playoffs,date,game,patch,participantid,side,position,playername,playerid,teamname,teamid,champion,ban1,ban2,ban3,ban4,ban5,gamelength,result,kills,deaths,assists,teamkills,teamdeaths,doublekills,triplekills,quadrakills,pentakills,firstblood,firstbloodkill,firstbloodassist,firstbloodvictim,team kpm,ckpm,firstdragon,dragons,opp_dragons,elementaldrakes,opp_elementaldrakes,infernals,mountains,clouds,oceans,chemtechs,hextechs,dragons (type unknown),elders,opp_elders,firstherald,heralds,opp_heralds,firstbaron,barons,opp_barons,firsttower,towers,opp_towers,firstmidtower,firsttothreetowers,turretplates,opp_turretplates,inhibitors,opp_inhibitors,damagetochampions,dpm,damageshare,damagetakenperminute,damagemitigatedperminute,wardsplaced,wpm,wardskilled,wcpm,controlwardsbought,visionscore,vspm,totalgold,earnedgold,earned gpm,earnedgoldshare,goldspent,gspd,total cs,minionkills,monsterkills,monsterkillsownjungle,monsterkillsenemyjungle,cspm,goldat10,xpat10,csat10,opp_goldat10,opp_xpat10,opp_csat10,golddiffat10,xpdiffat10,csdiffat10,killsat10,assistsat10,deathsat10,opp_killsat10,opp_assistsat10,opp_deathsat10,goldat15,xpat15,csat15,opp_goldat15,opp_xpat15,opp_csat15,golddiffat15,xpdiffat15,csdiffat15,killsat15,assistsat15,deathsat15,opp_killsat15,opp_assistsat15,opp_deathsat15,void_grubs,gpr
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,1,Blue,top,NuQ,oe:player:ec596621f8c01e1b79c177346b4a158,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,Gwen,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,12,6,3,22,27,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5193,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,1.0,56989.0,1345.1377,0.435756,1366.7349,1240.2911,12.0,0.2832,8.0,0.1888,6.0,53.0,1.251,20574,15113.0,356.7191,0.314582,17950.0,,342.0,325.0,17.0,,,8.0724,3447.0,4709.0,83.0,2838.0,4442.0,65.0,609.0,267.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,4956.0,7479.0,130.0,5140.0,7451.0,113.0,-184.0,28.0,17.0,0.0,0.0,1.0,0.0,1.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,2,Blue,jng,Velja,oe:player:a68581f40d30119bad63673b157e660,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,Wukong,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,2,6,8,22,27,1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.5193,1.1566,,,,,,,,,,,,,,,,,,,1.0,2.0,,,,,,,,0.0,1.0,16642.0,392.8088,0.12725,990.6845,1365.0354,16.0,0.3777,15.0,0.3541,14.0,73.0,1.7231,12365,6904.0,162.9583,0.143706,11850.0,,181.0,29.0,152.0,,,4.2722,3510.0,2951.0,49.0,3159.0,3207.0,66.0,351.0,-256.0,-17.0,2.0,1.0,0.0,0.0,0.0,0.0,4899.0,4939.0,79.0,5139.0,5585.0,107.0,-240.0,-646.0,-28.0,2.0,2.0,0.0,1.0,0.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,3,Blue,mid,Vigil,oe:player:c08f75d1d70bf068013b32f60fe96a5,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,Galio,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,0,5,13,22,27,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5193,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,1.0,15410.0,363.7293,0.11783,795.9087,1097.0653,13.0,0.3068,4.0,0.0944,3.0,27.0,0.6373,12922,7461.0,176.1054,0.1553,14060.0,,269.0,265.0,4.0,,,6.3493,3059.0,4716.0,80.0,3350.0,4917.0,96.0,-291.0,-201.0,-16.0,0.0,0.0,0.0,0.0,0.0,0.0,4662.0,7501.0,130.0,5647.0,8072.0,161.0,-985.0,-571.0,-31.0,0.0,0.0,0.0,0.0,0.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,4,Blue,bot,JaVaaa,oe:player:99b95ef29b1e18c587457d79910c685,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,Zeri,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,6,6,11,22,27,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.5193,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,0.0,33255.0,784.9331,0.254278,805.1377,617.9386,18.0,0.4249,25.0,0.5901,6.0,84.0,1.9827,19395,13934.0,328.8906,0.29004,17500.0,,416.0,403.0,13.0,,,9.819,3635.0,3205.0,83.0,3512.0,2811.0,72.0,123.0,394.0,11.0,1.0,2.0,0.0,1.0,0.0,1.0,5490.0,4832.0,129.0,5065.0,4684.0,117.0,425.0,148.0,12.0,2.0,2.0,0.0,1.0,0.0,1.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,5,Blue,sup,Tinelli,oe:player:a5c70cdfcff6537d4af4e9204ad252c,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,Yuumi,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,2,4,20,22,27,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.5193,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,0.0,8486.0,200.299,0.0648866,205.4209,138.6231,39.0,0.9205,2.0,0.0472,10.0,64.0,1.5106,10091,4630.0,109.284,0.096371,9700.0,,7.0,7.0,0.0,,,0.1652,2147.0,2881.0,1.0,2242.0,2244.0,13.0,-95.0,637.0,-12.0,0.0,3.0,1.0,0.0,1.0,2.0,3105.0,4149.0,1.0,3251.0,3196.0,21.0,-146.0,953.0,-20.0,0.0,4.0,1.0,0.0,1.0,3.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,6,Red,top,StenBosse,oe:player:b366e140eae91b8d8f8c03294238f32,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,Akali,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,4,2,8,27,22,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6373,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,1.0,0.0,26426.0,623.7451,0.193284,650.9599,443.5563,8.0,0.1888,5.0,0.118,4.0,30.0,0.7081,16629,11168.0,263.6035,0.198299,15300.0,,297.0,293.0,4.0,,,7.0102,2838.0,4442.0,65.0,3447.0,4709.0,83.0,-609.0,-267.0,-18.0,0.0,0.0,0.0,0.0,0.0,0.0,5140.0,7451.0,113.0,4956.0,7479.0,130.0,184.0,-28.0,-17.0,0.0,1.0,0.0,0.0,0.0,1.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,7,Red,jng,Stefan,oe:player:752b048c76633399b04f6e87a8b211c,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,Vi,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,7,3,13,27,22,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6373,1.1566,,,,,,,,,,,,,,,,,,,2.0,1.0,,,,,,,,1.0,0.0,22212.0,524.2801,0.162462,980.1101,2405.7828,14.0,0.3304,25.0,0.5901,7.0,83.0,1.9591,16858,11397.0,269.0087,0.202365,14575.0,,244.0,80.0,164.0,,,5.7592,3159.0,3207.0,66.0,3510.0,2951.0,49.0,-351.0,256.0,17.0,0.0,0.0,0.0,2.0,1.0,0.0,5139.0,5585.0,107.0,4899.0,4939.0,79.0,240.0,646.0,28.0,1.0,0.0,0.0,2.0,2.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,8,Red,mid,Dehaste,oe:player:5f4898a107d2948045588584562be3f,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,Cassiopeia,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,8,1,12,27,22,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6373,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,1.0,0.0,34570.0,815.9717,0.252851,744.8544,667.2227,17.0,0.4013,9.0,0.2124,6.0,68.0,1.605,21653,16192.0,382.1873,0.287507,22850.0,,447.0,342.0,105.0,,,10.5507,3350.0,4917.0,96.0,3059.0,4716.0,80.0,291.0,201.0,16.0,0.0,0.0,0.0,0.0,0.0,0.0,5647.0,8072.0,161.0,4662.0,7501.0,130.0,985.0,571.0,31.0,0.0,0.0,0.0,0.0,0.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,9,Red,bot,Ellam,oe:player:6a9dedd375fda0d451a45fa9809875e,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,Draven,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,8,8,13,27,22,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.6373,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,0.0,45074.0,1063.9024,0.329679,749.8348,704.1857,15.0,0.3541,8.0,0.1888,5.0,64.0,1.5106,18382,12921.0,304.9803,0.229426,16225.0,,277.0,264.0,13.0,,,6.5382,3512.0,2811.0,72.0,3635.0,3205.0,83.0,-123.0,-394.0,-11.0,1.0,0.0,1.0,1.0,2.0,0.0,5065.0,4684.0,117.0,5490.0,4832.0,129.0,-425.0,-148.0,-12.0,1.0,0.0,1.0,2.0,2.0,0.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,10,Red,sup,Venzer,oe:player:dd4e48d56d876e846579c9e5c64d965,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,Nautilus,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,0,8,23,27,22,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6373,1.1566,,,,,,,,,,,,,,,,,,,0.0,0.0,,,,,,,,0.0,0.0,8439.0,199.1896,0.0617242,720.0,932.5256,68.0,1.605,11.0,0.2596,24.0,125.0,2.9504,10102,4641.0,109.5437,0.0824027,9950.0,,31.0,31.0,0.0,,,0.7317,2242.0,2244.0,13.0,2147.0,2881.0,1.0,95.0,-637.0,12.0,0.0,1.0,2.0,0.0,3.0,1.0,3251.0,3196.0,21.0,3105.0,4149.0,1.0,146.0,-953.0,20.0,0.0,1.0,3.0,0.0,4.0,1.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,100,Blue,team,,,Cyberground Gaming,oe:team:7894460f171d66e8ffc149c05b056a8,,Maokai,Ashe,Heimerdinger,Syndra,Azir,2542,0,22,27,55,22,27,4.0,0.0,0.0,0.0,1.0,,,,0.5193,1.1566,1.0,1.0,5.0,1.0,4.0,0.0,0.0,0.0,0.0,1.0,0.0,,0.0,1.0,0.0,1.0,1.0,0.0,1.0,2.0,0.0,3.0,11.0,0.0,0.0,3.0,6.0,0.0,5.0,130782.0,3086.9079,,4163.8867,4458.9536,98.0,2.3131,54.0,1.2746,39.0,301.0,7.1046,75347,48041.0,1133.9339,,71060.0,-0.104561,,1029.0,186.0,,,28.6782,15798.0,18462.0,296.0,15101.0,17621.0,312.0,697.0,841.0,-16.0,3.0,6.0,1.0,1.0,1.0,3.0,23112.0,28900.0,469.0,24242.0,28988.0,519.0,-1130.0,-88.0,-50.0,4.0,8.0,2.0,2.0,2.0,4.0,,
ESPORTSTMNT03_3091672,complete,,PGN,2023,Spring,0,2023-01-26 18:03:49,1,13.01,200,Red,team,,,aNc Outplayed,oe:team:cd6b4265231d4ba346274357681c335,,Ryze,Jax,Lucian,Viktor,Sylas,2542,1,27,22,69,27,22,4.0,0.0,0.0,0.0,0.0,,,,0.6373,1.1566,0.0,5.0,1.0,4.0,1.0,0.0,0.0,3.0,1.0,0.0,0.0,,1.0,0.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,11.0,3.0,1.0,1.0,6.0,3.0,5.0,0.0,136721.0,3227.0889,,3845.7592,5153.273,122.0,2.8796,58.0,1.369,46.0,370.0,8.7333,83624,56318.0,1329.2998,,78900.0,0.104561,,1010.0,286.0,,,30.5901,15101.0,17621.0,312.0,15798.0,18462.0,296.0,-697.0,-841.0,16.0,1.0,1.0,3.0,3.0,6.0,1.0,24242.0,28988.0,519.0,23112.0,28900.0,469.0,1130.0,88.0,50.0,2.0,2.0,4.0,4.0,8.0,2.0,,"""
        return pd.read_csv(StringIO(raw_data))

    @mock_s3
    def test_ingest_data(self, mock_raw_data):
        # Create a boto3 session and resource after the mock has started
        session = get_s3_session(region_name=REGION_NAME)
        create_s3_bucket(session, BUCKET_NAME)
        upload_data_to_s3(session, BUCKET_NAME, DATA_FILE_NAME, mock_raw_data)

        # Create the OraclesElixir instance
        oe = OraclesElixir(session=session, bucket="test_bucket")

        # Call the method under test
        oe_data = oe.ingest_data([2023])
        # Check that the returned DataFrame matches the test data
        assert oe_data.equals(mock_raw_data)

    def test_format_data_types(self, mock_raw_data):
        # Create the OraclesElixir instance
        oe = OraclesElixir(session=None, bucket=None)

        # Call the method under test
        formatted_data = oe.format_data_types(mock_raw_data)
        assert formatted_data["date"].dtype == "datetime64[ns]"

        for colname in ["gameid", "playerid", "teamid", "position"]:
            assert not formatted_data[colname].str.contains('""').any()
            assert not formatted_data[colname].str.contains("nan").any()
            assert not formatted_data[colname].str.contains("null").any()

    def test_remove_null_games(self):
        # Setup: Create a DataFrame with one row having a null 'gameid'
        data = {
            "gameid": ["game1", None, "game2"],
            "playername": ["player1", "player2", "player3"],
        }
        df = pd.DataFrame(data)

        # Execution: Call the remove_null_games function
        result_df = OraclesElixir.remove_null_games(df)

        # Assertion: Check that the resulting DataFrame does not contain the row with a null 'gameid'
        expected_data = {
            "gameid": ["game1", "game2"],
            "playername": ["player1", "player3"],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_drop_unknown_entities(self):
        # Setup: Create a DataFrame with "unknown player" and "unknown team"
        data = {
            "gameid": ["game1", "game2", "game3"],
            "playername": ["player1", "unknown player", "player3"],
            "teamname": ["team1", "team2", "unknown team"],
        }
        df = pd.DataFrame(data)

        # Execution: Call the drop_unknown_entities function
        result_df = OraclesElixir.drop_unknown_entities(df)

        # Assertion: Check that the resulting DataFrame does not contain rows with "unknown player" or "unknown team"
        expected_data = {
            "gameid": ["game1"],
            "playername": ["player1"],
            "teamname": ["team1"],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_sort_data(self):
        # Setup: Create a DataFrame with unsorted data
        data = {
            "league": ["L3", "L1", "L2", "L2"],
            "date": ["2023-01-01", "2023-02-01", "2023-01-15", "2023-01-15"],
            "gameid": ["game2", "game3", "game2", "game1"],
            "teamid": [
                "team1",
                "team1",
                "team1",
                "team1",
            ],  # "team" is a subset of "player
            "side": ["Blue", "Red", "Blue", "Blue"],
            "position": ["mid", "top", "bot", "mid"],
        }
        df = pd.DataFrame(data)

        # Execution: Call the sort_data function with 'player' since 'team' is a subset of 'player'
        result_df_player = OraclesElixir.sort_data(df, "player")
        result_df_team = OraclesElixir.sort_data(df, "team")

        # Assertion: Check that the resulting DataFrame is sorted by league, date, gameid, side, and position
        expected_data_player = {
            "league": ["L3", "L2", "L2", "L1"],
            "date": ["2023-01-01", "2023-01-15", "2023-01-15", "2023-02-01"],
            "gameid": ["game2", "game1", "game2", "game3"],
            "teamid": ["team1", "team1", "team1", "team1"],
            "side": ["Blue", "Blue", "Blue", "Red"],
            "position": ["mid", "mid", "bot", "top"],
        }
        expected_data_team = {
            "league": ["L3", "L2", "L2", "L1"],
            "date": ["2023-01-01", "2023-01-15", "2023-01-15", "2023-02-01"],
            "gameid": ["game2", "game1", "game2", "game3"],
            "teamid": ["team1", "team1", "team1", "team1"],
            "side": ["Blue", "Blue", "Blue", "Red"],
            "position": ["mid", "mid", "bot", "top"],
        }
        expected_df_player = pd.DataFrame(expected_data_player)
        expected_df_team = pd.DataFrame(expected_data_team)

        pd.testing.assert_frame_equal(
            result_df_player.reset_index(drop=True),
            expected_df_player.reset_index(drop=True),
        )

        pd.testing.assert_frame_equal(
            result_df_team.reset_index(drop=True),
            expected_df_team.reset_index(drop=True),
        )

    def test_fill_null_team_ids(self):
        # Setup: Create a DataFrame with some null 'teamid' values
        data = {
            "gameid": ["game1", "game2"],
            "teamname": ["Team A", "Team B"],
            "teamid": [np.nan, "teamBID"],
            "position": ["top", "bot"],
        }
        df1 = pd.DataFrame(data)
        df2 = pd.DataFrame(data)

        # Execution: Call the fill_null_team_ids function with 'team'
        result_df_team = OraclesElixir.fill_null_team_ids(df1, "team")
        result_df_player = OraclesElixir.fill_null_team_ids(df2, "player")

        # Assertion: Check that the 'teamid' null values are filled with 'teamname' values
        expected_data_team = {
            "gameid": ["game1", "game2"],
            "teamname": ["Team A", "Team B"],
            "teamid": [np.nan, "teamBID"],
            "position": ["top", "bot"],
        }
        expected_data_player = {
            "gameid": ["game1", "game2"],
            "teamname": ["Team A", "Team B"],
            "teamid": ["Team A", "teamBID"],
            "position": ["top", "bot"],
        }
        expected_df_team = pd.DataFrame(expected_data_team)
        expected_df_player = pd.DataFrame(expected_data_player)

        pd.testing.assert_frame_equal(
            result_df_team.reset_index(drop=True),
            expected_df_team.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            result_df_player.reset_index(drop=True),
            expected_df_player.reset_index(drop=True),
        )

    def test_subset_data(self):
        mock_import_columns = {
            "player": ["gameid", "playername", "position", "teamname"],
            "team": ["gameid", "teamname"],
        }

        with mock.patch("src.data_ingest.oracles_elixir.IMPORT_COLUMNS", new=mock_import_columns):
            data = {
                "gameid": ["game1", "game2"],
                "playername": ["Player A", "Player B"],
                "position": ["mid", "team"],
                "teamname": ["Team A", "Team B"],
                "extra_column": ["Extra Data", "Extra Data"],
            }
            df = pd.DataFrame(data)

            # Execution and Assertion for 'player'
            result_df_player = OraclesElixir.subset_data(df, "player", mock_import_columns)
            result_df_team = OraclesElixir.subset_data(df, "team", mock_import_columns)

            expected_data_player = {
                "gameid": ["game1"],
                "playername": ["Player A"],
                "position": ["mid"],
                "teamname": ["Team A"],
            }
            expected_data_team = {
                "gameid": ["game2"],
                "teamname": ["Team B"],
            }
            expected_df_player = pd.DataFrame(expected_data_player)
            expected_df_team = pd.DataFrame(expected_data_team)

            pd.testing.assert_frame_equal(
                result_df_player.reset_index(drop=True),
                expected_df_player.reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                result_df_team.reset_index(drop=True),
                expected_df_team.reset_index(drop=True),
            )

    def test_subset_data_with_invalid_position(self):
        # Mock IMPORT_COLUMNS to specify expected columns for 'player' and 'team'
        mock_import_columns = {
            "player": ["gameid", "playername", "position", "teamname"],
            "team": ["gameid", "teamname"],
        }

        # Define a DataFrame that includes an invalid position ("caster" is not in mock_import_columns)
        data = {
            "gameid": ["game1", "game2"],
            "playername": ["Player A", "Player B"],
            "position": ["mid", "team"],
            "teamname": ["Team A", "Team B"],
            "extra_column": ["Extra Data", "Extra Data"],
        }
        df = pd.DataFrame(data)

        # Patch the IMPORT_COLUMNS within the OraclesElixir module
        with mock.patch("src.data_ingest.oracles_elixir.IMPORT_COLUMNS", new=mock_import_columns):
            # Expecting a ValueError to be raised due to the invalid 'position'
            with pytest.raises(ValueError):
                OraclesElixir.subset_data(df, "caster", mock_import_columns)

    def test_remove_inconsistent_games(self):
        # Setup: Create a DataFrame with inconsistent game records
        data = {
            "gameid": [
                "game1",
                "game1",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
            ],
            "position": [
                "team",
                "team",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
            ],
        }
        df = pd.DataFrame(data)

        # Execution: Call the remove_inconsistent_games function with 'team'
        team_df = OraclesElixir.remove_inconsistent_games(df, "team")
        player_df = OraclesElixir.remove_inconsistent_games(df, "player")

        # Assertion: Check that 'game2' is removed for 'team' split
        expected_team_data = {
            "gameid": ["game1", "game1"],
            "position": ["team", "team"],
        }
        expected_player_data = {
            "gameid": [
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
                "game2",
            ],
            "position": [
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
                "player",
            ],
        }
        expected_team_data = pd.DataFrame(expected_team_data)
        expected_player_data = pd.DataFrame(expected_player_data)

        pd.testing.assert_frame_equal(
            team_df.reset_index(drop=True),
            expected_team_data.reset_index(drop=True),
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            player_df.reset_index(drop=True),
            expected_player_data.reset_index(drop=True),
            check_like=True,
        )

    def test_enrich_opponent_team_metrics(self):
        # Setup: Create a simplified DataFrame to simulate the enrich process
        data = {
            "gameid": ["game1", "game1"],
            "teamname": ["ABC", "EFG"],
            "teamid": ["teamA", "teamB"],
            "egpm": [1000, 2000],
            "position": ["team", "team"],
        }
        df = pd.DataFrame(data)

        # Execution: Call the enrich_opponent_metrics function with 'team'
        result_df = OraclesElixir.enrich_opponent_metrics(df, "team")

        # Assertion: Check that opponent metrics are correctly added
        df["opponentteam"] = ["EFG", "ABC"]
        df["opponentteamid"] = ["teamB", "teamA"]
        df["opponent_egpm"] = [2000, 1000]

        pd.testing.assert_frame_equal(result_df, df)

    def test_enrich_opponent_player_metrics(self):
        # Setup: Create a simplified DataFrame to simulate the enrich process
        data = {
            "gameid": [
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
                "game1",
            ],
            "playername": [
                "Player A",
                "Player B",
                "Player C",
                "Player D",
                "Player E",
                "Player 1",
                "Player 2",
                "Player 3",
                "Player 4",
                "Player 5",
            ],
            "playerid": [
                "playerA",
                "playerB",
                "playerC",
                "playerD",
                "playerE",
                "player1",
                "player2",
                "player3",
                "player4",
                "player5",
            ],
            "egpm": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            "position": [
                "top",
                "jgl",
                "mid",
                "bot",
                "sup",
                "top",
                "jgl",
                "mid",
                "bot",
                "sup",
            ],
            "teamname": [
                "ABC",
                "ABC",
                "ABC",
                "ABC",
                "ABC",
                "EFG",
                "EFG",
                "EFG",
                "EFG",
                "EFG",
            ],
            "teamid": [
                "teamA",
                "teamA",
                "teamA",
                "teamA",
                "teamA",
                "teamB",
                "teamB",
                "teamB",
                "teamB",
                "teamB",
            ],
        }
        df = pd.DataFrame(data)

        # Execution: Call the enrich_opponent_metrics function with 'player'
        result_df = OraclesElixir.enrich_opponent_metrics(df, "player")

        # Assertion: Check that opponent metrics are correctly added
        df["opponentteam"] = [
            "EFG",
            "EFG",
            "EFG",
            "EFG",
            "EFG",
            "ABC",
            "ABC",
            "ABC",
            "ABC",
            "ABC",
        ]
        df["opponentteamid"] = [
            "teamB",
            "teamB",
            "teamB",
            "teamB",
            "teamB",
            "teamA",
            "teamA",
            "teamA",
            "teamA",
            "teamA",
        ]
        df["opponent_egpm"] = [
            6000,
            7000,
            8000,
            9000,
            10000,
            1000,
            2000,
            3000,
            4000,
            5000,
        ]
        df["opponentplayername"] = [
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4",
            "Player 5",
            "Player A",
            "Player B",
            "Player C",
            "Player D",
            "Player E",
        ]
        df["opponentplayerid"] = [
            "player1",
            "player2",
            "player3",
            "player4",
            "player5",
            "playerA",
            "playerB",
            "playerC",
            "playerD",
            "playerE",
        ]

        pd.testing.assert_frame_equal(result_df, df)

    def test_get_opponent_team(self):
        # Setup: Provide a simple list of team names as input
        column = ["Team A", "Team B"]
        entity = "team"

        # Execution: Call the get_opponent function
        opponent = get_opponent(column, entity)

        # Assertion: Check that the returned list correctly identifies opponents
        expected_opponent = ["Team B", "Team A"]
        assert opponent == expected_opponent, "Opponent function did not return expected values"

    def test_get_opponent_player(self):
        # Setup: Provide a simple list of player names as input
        column = [
            "Player A",
            "Player B",
            "Player C",
            "Player D",
            "Player E",
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4",
            "Player 5",
        ]
        entity = "player"

        # Execution: Call the get_opponent function
        opponent = get_opponent(column, entity)

        # Assertion: Check that the returned list correctly identifies opponents
        expected_opponent = [
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4",
            "Player 5",
            "Player A",
            "Player B",
            "Player C",
            "Player D",
            "Player E",
        ]
        assert opponent == expected_opponent, "Opponent function did not return expected values"

    def test_clean_data(self, mock_raw_data):
        # Sample data to test cleaning function
        sample_data = mock_raw_data

        oe = OraclesElixir(session=None, bucket=None)
        team_data = oe.clean_data(oe_data=sample_data, split_on="team")
        player_data = oe.clean_data(oe_data=sample_data, split_on="player")

        # Check that the returned DataFrames are not empty
        assert not team_data.empty
        assert not player_data.empty
