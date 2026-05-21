from pathlib import Path

from setuptools import find_namespace_packages

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_src_tree_removed():
    assert not (ROOT / "src").exists()


def test_package_discovery_excludes_legacy_src():
    package_roots = [
        ROOT / "packages/oracle-bets-core/src",
        ROOT / "packages/lol-bets/src",
        ROOT / "packages/oracle-bets-discord/src",
    ]
    packages = {
        package
        for root in package_roots
        for package in find_namespace_packages(str(root))
    }

    assert "oracle_bets_core" in packages
    assert "lol_bets" in packages
    assert "oracle_bets_discord" in packages
    assert "utils" not in packages
    assert "data_generation" not in packages
    assert "prediction_models" not in packages
    assert "discord_predictions" not in packages
