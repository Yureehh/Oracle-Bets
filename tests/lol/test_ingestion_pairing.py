import pytest
from lol_bets.data_generation.ingestion.oracles_elixir import (
    OraclesElixir,
    OraclesElixirError,
)
from oracle_bets_core.pd import pd

POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def _full_raw_game(gameid: str, league: str = "LCK", patch: str = "15.1") -> list[dict]:
    rows = [
        {
            "date": "2025-01-01",
            "gameid": gameid,
            "league": league,
            "patch": patch,
            "playoffs": 0,
            "game": 1,
            "side": side,
            "position": position,
            "result": result,
            "teamname": team,
            "teamid": team_id,
            "playername": f"{team}-{position}",
            "playerid": f"{team_id}-{position}",
            "gamelength": 1800,
        }
        for side, team, team_id, result in [
            ("Blue", "T1", "t1", 1),
            ("Red", "T2", "t2", 0),
        ]
        for position in POSITIONS
    ]
    for side, team, team_id, result in [
        ("Blue", "T1", "t1", 1),
        ("Red", "T2", "t2", 0),
    ]:
        rows.append(
            {
                "date": "2025-01-01",
                "gameid": gameid,
                "league": league,
                "patch": patch,
                "playoffs": 0,
                "game": 1,
                "side": side,
                "position": "team",
                "result": result,
                "teamname": team,
                "teamid": team_id,
                "playername": team,
                "playerid": pd.NA,
                "gamelength": 1800,
            }
        )
    return rows


def test_enrich_opponent_metrics_pairs_teams_without_block_order():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "league": ["LCK", "LCK"],
            "gameid": ["g1", "g1"],
            "side": ["Red", "Blue"],
            "teamname": ["T2", "T1"],
            "teamid": ["t2", "t1"],
        }
    )

    out = OraclesElixir.enrich_opponent_metrics(df, "team").set_index("side")

    assert out.loc["Blue", "opponentteam"] == "T2"
    assert out.loc["Red", "opponentteam"] == "T1"


def test_enrich_opponent_metrics_pairs_players_by_position():
    rows = [
        {
            "date": pd.Timestamp("2025-01-01"),
            "league": "LCK",
            "gameid": "g1",
            "side": side,
            "position": position,
            "teamname": team,
            "teamid": team_id,
            "playername": f"{team}-{position}",
            "playerid": f"{team_id}-{position}",
        }
        for side, team, team_id in [("Red", "T2", "t2"), ("Blue", "T1", "t1")]
        for position in POSITIONS
    ]

    out = OraclesElixir.enrich_opponent_metrics(pd.DataFrame(rows), "player")
    blue_top = out[(out["side"] == "Blue") & (out["position"] == "top")].iloc[0]

    assert blue_top["opponentplayername"] == "T2-top"


def test_remove_buggy_games_drops_invalid_full_game_before_split():
    rows = _full_raw_game("good") + _full_raw_game("bad")
    raw = pd.DataFrame(rows)
    raw = raw[
        ~(
            (raw["gameid"] == "bad")
            & (raw["position"] == "sup")
            & (raw["side"] == "Red")
        )
    ]

    cleaned = OraclesElixir._remove_buggy_games(raw)

    assert set(cleaned["gameid"]) == {"good"}


def test_validate_game_composition_rejects_partial_post_filter_games():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2"],
            "side": ["Blue", "Red", "Blue"],
        }
    )

    with pytest.raises(OraclesElixirError, match="Invalid team game composition"):
        OraclesElixir.validate_game_composition(df, "team")


def test_fill_null_patch_value_fills_middle_nulls():
    df = pd.DataFrame({"patch": ["15.1", None, "15.2"]})

    out = OraclesElixir.fill_null_patch_value(df)

    assert out["patch"].tolist() == ["15.1", "15.1", "15.2"]


def test_fill_null_patch_value_rejects_leading_nulls():
    df = pd.DataFrame({"patch": [None, "15.1"]})

    with pytest.raises(OraclesElixirError, match="Patch values still contain"):
        OraclesElixir.fill_null_patch_value(df)
