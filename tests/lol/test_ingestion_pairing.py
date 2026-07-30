import pytest
from lol_bets.data_generation.ingestion.oracles_elixir import (
    OraclesElixir,
    OraclesElixirError,
)
from oracle_bets_core.pd import pd

POSITIONS = ["top", "jng", "mid", "bot", "sup"]
EXPECTED_IDENTIFIED_TEAMS = 2


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


def test_ingest_data_prefers_local_csv_files(tmp_path):
    for year in (2024, 2026):
        path = tmp_path / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        path.write_text(f"gameid,date\ng{year},2025-01-01\n")

    result = OraclesElixir(local_data_dir=tmp_path).ingest_data(years=[2024, 2026])

    assert result["gameid"].tolist() == ["g2024", "g2026"]


def test_ingest_data_requires_every_requested_local_file(tmp_path):
    path = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
    path.write_text("gameid,date\ng2024,2024-01-01\n")

    with pytest.raises(OraclesElixirError, match="Missing required local"):
        OraclesElixir(local_data_dir=tmp_path).ingest_data(years=[2024, 2026])


def test_local_drive_rejects_empty_placeholder(tmp_path):
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    path.touch()

    with pytest.raises(OraclesElixirError, match="is empty"):
        OraclesElixir._read_local_csv(path)


def test_local_drive_read_errors_include_recovery_guidance(monkeypatch, tmp_path):
    path = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
    path.touch()

    def fail_read_csv(*_args, **_kwargs):
        raise OSError(70, "Stale NFS file handle")

    monkeypatch.setattr(
        "lol_bets.data_generation.ingestion.oracles_elixir.pd.read_csv",
        fail_read_csv,
    )

    with pytest.raises(
        OraclesElixirError,
        match="available offline",
    ):
        OraclesElixir._read_local_csv(path)


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


def test_subset_data_renames_first_pick_for_team_rows():
    raw = pd.DataFrame(
        [
            {
                "position": "team",
                "gameid": "g1",
                "side": "Red",
                "teamname": "T1",
                "teamid": "t1",
                "firstPick": 1,
            }
        ]
    )

    out = OraclesElixir.subset_data(raw, "team", columns={"team": ["first_pick"]})

    assert out.iloc[0]["first_pick"] == 1


def test_validate_game_composition_rejects_partial_post_filter_games():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1", "g2"],
            "side": ["Blue", "Red", "Blue"],
        }
    )

    with pytest.raises(OraclesElixirError, match="Invalid team game composition"):
        OraclesElixir.validate_game_composition(df, "team")


def test_validate_game_composition_rejects_duplicate_team_sides():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "side": ["Blue", "Blue"],
            "result": [1, 0],
        }
    )

    with pytest.raises(OraclesElixirError, match="side composition"):
        OraclesElixir.validate_game_composition(df, "team")


def test_validate_game_composition_rejects_duplicate_player_roles():
    raw = pd.DataFrame(_full_raw_game("g1"))
    players = raw[raw["position"] != "team"].copy()
    players.loc[
        (players["side"] == "Blue") & (players["position"] == "sup"),
        "position",
    ] = "bot"

    with pytest.raises(OraclesElixirError, match="role composition"):
        OraclesElixir.validate_game_composition(players, "player")


def test_validate_game_composition_rejects_games_without_one_winning_team():
    df = pd.DataFrame(
        {
            "gameid": ["g1", "g1"],
            "side": ["Blue", "Red"],
            "result": [1, 1],
        }
    )

    with pytest.raises(OraclesElixirError, match="result composition"):
        OraclesElixir.validate_game_composition(df, "team")


def test_fill_null_patch_value_fills_middle_nulls():
    df = pd.DataFrame({"patch": ["15.1", None, "15.2"]})

    out = OraclesElixir.fill_null_patch_value(df)

    assert out["patch"].tolist() == ["15.1", "15.1", "15.2"]


def test_fill_null_patch_value_rejects_leading_nulls():
    df = pd.DataFrame({"patch": [None, "15.1"]})

    with pytest.raises(OraclesElixirError, match="Patch values still contain"):
        OraclesElixir.fill_null_patch_value(df)


def test_fill_null_patch_value_rejects_unsorted_dates():
    # Forward-fill on rows that are not date-ordered can copy a patch from an
    # unrelated era; the guard must refuse to fill in that case.
    df = pd.DataFrame(
        {
            "patch": ["15.2", None, "15.1"],
            "date": pd.to_datetime(["2025-03-01", "2025-01-15", "2025-01-01"]),
        }
    )

    with pytest.raises(OraclesElixirError, match="date-ordered"):
        OraclesElixir.fill_null_patch_value(df)


def test_fill_null_team_ids_uses_teamname_and_drops_identityless_rows():
    df = pd.DataFrame(
        {
            "teamname": ["T1", "Gen.G", None],
            "teamid": [None, "oe:team:geng", None],
        }
    )

    out = OraclesElixir.fill_null_team_ids(df)

    # Null teamid backfills from teamname instead of becoming the string "nan".
    assert out.loc[0, "teamid"] == "T1"
    assert out.loc[1, "teamid"] == "oe:team:geng"
    # Rows with neither identity are removed, not kept as "nan".
    assert len(out) == EXPECTED_IDENTIFIED_TEAMS
    assert not out["teamid"].str.casefold().isin({"nan", "<na>", "none", ""}).any()
