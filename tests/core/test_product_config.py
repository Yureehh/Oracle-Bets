from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from oracle_bets_core.config import (
    ProductConfig,
    ProductConfigError,
    load_product_config,
)
from oracle_bets_core.paths import PRODUCT_CONFIG

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FIXTURE_WINDOW_HOURS = 36
EXPECTED_MAJOR_MAP_TRIGGER = 20
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_VALID_MAP_TRIGGER = 50


def _raw_config() -> dict:
    return json.loads((ROOT / "config/product/product.json").read_text())


def test_product_config_contains_only_runtime_and_safety_rules():
    config = load_product_config()

    assert PRODUCT_CONFIG == ROOT / "config/product/product.json"
    assert config.schema_version == EXPECTED_SCHEMA_VERSION
    assert config.timezone == "Europe/Rome"
    assert config.fixture_window_hours == EXPECTED_FIXTURE_WINDOW_HOURS
    assert config.market.read_only is True
    assert config.leagues.profile == "tier1_plus_erls"
    assert config.training.new_valid_maps_trigger == EXPECTED_VALID_MAP_TRIGGER
    assert config.training.new_major_maps_trigger == EXPECTED_MAJOR_MAP_TRIGGER
    assert config.promotion.automatic is False


def test_config_round_trip_is_stable():
    raw = _raw_config()

    config = ProductConfig.from_dict(raw)

    assert config.to_dict() == raw


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("market", "read_only", False, "read-only"),
        ("leagues", "profile", "research_all_supported", "tier1_plus_erls"),
        ("training", "new_valid_maps_trigger", 1, "triggers"),
        ("promotion", "automatic", True, "automatic"),
    ],
)
def test_config_rejects_values_that_break_operating_rules(section, key, value, message):
    raw = deepcopy(_raw_config())
    raw[section][key] = value

    with pytest.raises(ProductConfigError, match=message):
        ProductConfig.from_dict(raw)


def test_config_rejects_unknown_fields():
    raw = _raw_config()
    raw["market"]["wallet"] = "forbidden"

    with pytest.raises(ProductConfigError, match="Unknown"):
        ProductConfig.from_dict(raw)
