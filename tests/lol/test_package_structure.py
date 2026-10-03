import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_src_tree_removed():
    assert not (ROOT / "src").exists()


def test_package_discovery_excludes_legacy_src():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_roots = [
        ROOT / path
        for path in config["tool"]["setuptools"]["packages"]["find"]["where"]
    ]
    packages = {
        ".".join(package.parent.relative_to(root).parts)
        for root in package_roots
        for package in root.rglob("__init__.py")
    }

    assert "oracle_bets_core" in packages
    assert "lol_bets" in packages
    assert "oracle_bets_discord" in packages
    assert "utils" not in packages
    assert "data_generation" not in packages
    assert "prediction_models" not in packages
    assert "discord_predictions" not in packages
