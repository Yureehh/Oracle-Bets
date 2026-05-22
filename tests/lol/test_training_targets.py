import pytest
from lol_bets.training import parse_training_targets


def _targets(selector: str) -> list[str]:
    return [cfg.target_column for cfg in parse_training_targets(selector)]


def test_training_target_parser_supports_all_outcome_props_and_commas():
    assert _targets("all") == ["result", "gamelength", "total_kills", "total_towers"]
    assert _targets("outcome") == ["result"]
    assert _targets("props") == ["gamelength", "total_kills", "total_towers"]
    assert _targets("total_kills,total_towers") == ["total_kills", "total_towers"]


def test_training_target_parser_rejects_unknown_targets():
    with pytest.raises(ValueError, match="Unknown training target"):
        parse_training_targets("barons")
