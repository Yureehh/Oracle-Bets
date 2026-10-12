"""Reuse a feature build only when its complete output contract is unchanged."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from oracle_bets_core.io_utils import atomic_write_text

from lol_bets.operations.training_inputs import file_evidence


def record_feature_build(
    receipt: Path, *, paths: dict[str, Path], source: dict[str, Any], code: str
) -> None:
    atomic_write_text(
        receipt,
        json.dumps(
            {
                "schema_version": 1,
                "source": source,
                "code_sha256": code,
                "files": file_evidence(paths),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def feature_build_matches(
    receipt: Path, *, paths: dict[str, Path], source: dict[str, Any], code: str
) -> bool:
    try:
        saved = json.loads(receipt.read_text())
        return (
            bool(paths)
            and saved.get("schema_version") == 1
            and saved.get("source") == source
            and saved.get("code_sha256") == code
            and saved.get("files") == file_evidence(paths)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
