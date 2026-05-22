import pytest
from oracle_bets_discord.predictions.lol import resolve_first_pick


def test_resolve_first_pick_marks_only_named_team():
    assert resolve_first_pick("T1", "Gen.G", "Gen.G") == (False, True)
    assert resolve_first_pick("T1", "Gen.G", "T1") == (True, False)
    assert resolve_first_pick("T1", "Gen.G", None) == (None, None)


def test_resolve_first_pick_rejects_unknown_team():
    with pytest.raises(ValueError, match="First-pick team"):
        resolve_first_pick("T1", "Gen.G", "G2")
