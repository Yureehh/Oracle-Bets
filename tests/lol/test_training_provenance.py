from __future__ import annotations

import subprocess

import pytest
from lol_bets.operations.provenance import (
    DirtyWorktreeError,
    require_clean_repository,
)


def _git(root, *args):
    return subprocess.run(  # noqa: S603
        ["/usr/bin/git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_clean_repository_returns_exact_head(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "-m", "initial")

    provenance = require_clean_repository(tmp_path)

    assert provenance.clean
    assert provenance.revision == _git(tmp_path, "rev-parse", "HEAD").stdout.strip()


def test_dirty_or_untracked_source_blocks_registered_training(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "-m", "initial")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("NEW = True\n", encoding="utf-8")

    with pytest.raises(DirtyWorktreeError, match=r"M source\.py") as error:
        require_clean_repository(tmp_path)

    assert "?? new.py" in str(error.value)
