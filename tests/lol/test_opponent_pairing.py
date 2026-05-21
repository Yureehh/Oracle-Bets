import pytest
from lol_bets.data_generation.ingestion.oracles_elixir import get_opponent


def test_get_opponent_pairs_team_rows():
    assert get_opponent(["blue", "red"], "team").tolist() == ["red", "blue"]


def test_get_opponent_pairs_player_blocks():
    players = ["b1", "b2", "b3", "b4", "b5", "r1", "r2", "r3", "r4", "r5"]

    assert get_opponent(players, "player").tolist() == [
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
    ]


def test_get_opponent_rejects_unknown_entity():
    with pytest.raises(RuntimeError):
        get_opponent(["a", "b"], "map")
