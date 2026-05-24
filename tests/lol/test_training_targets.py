import pytest
from lol_bets.training import parse_training_targets
from oracle_bets_core.cli import build_parser

SELECTED_FEATURE_COUNT = 90


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


def test_training_parser_supports_feature_set_and_max_features():
    args = build_parser().parse_args(
        [
            "lol",
            "train",
            "--targets",
            "outcome",
            "--feature-set",
            "selected",
            "--max-features",
            str(SELECTED_FEATURE_COUNT),
        ]
    )

    assert args.feature_set == "selected"
    assert args.max_features == SELECTED_FEATURE_COUNT
