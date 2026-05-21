"""
Pandas shim with FireDucks acceleration when available.

Usage: ``from oracle_bets_core.pd import pd`` in place of ``import pandas as pd``.
Falls back to pandas if FireDucks is not installed.
"""

from __future__ import annotations

import importlib
import os
from typing import Any


def _load_pd() -> Any:
    """
    Prefer FireDucks for speed but fall back to pandas if unavailable, opted
    out, or missing required functionality (reshape internals, groupby
    transform with custom lambdas).
    """
    if os.getenv("USE_FIREDUCKS", "1").lower() in {"0", "false", "no"}:
        import pandas as _pd  # type: ignore[import-untyped]  # noqa: ICN001

        return _pd

    try:
        import fireducks.pandas as _fd_pd  # type: ignore[import-not-found]
    except Exception:
        import pandas as _pd  # type: ignore[import-untyped]  # noqa: ICN001

        return _pd

    # Ensure FireDucks exposes the pandas reshape internals we rely on and that
    # key groupby/transform behaviors match pandas. If any check fails, fall
    # back to pandas for correctness.
    try:
        _extracted_from__load(_fd_pd)
    except Exception:
        import pandas as _pd  # type: ignore[import-untyped]  # noqa: ICN001

        return _pd

    return _fd_pd


def _extracted_from__load(_fd_pd):
    importlib.import_module("pandas.core.reshape.reshape")
    # Sanity check a tiny pivot to catch missing reshape pieces early.
    _fd_pd.DataFrame({"a": [1], "b": [2]}).pivot_table(values="b", index="a")

    # Sanity check groupby.transform with outer-scope mask (mirrors usage in
    # FeatureGenerator.compute_win_loss_metrics).
    tmp = _fd_pd.DataFrame(
        {
            "playerid": [1, 1, 2],
            "season": ["2024", "2024", "2024"],
            "result": [1, 0, 1],
            "kills": [3, 4, 5],
        }
    )
    mask = tmp["result"].eq(1)
    grp = tmp.groupby(["playerid", "season"], sort=False, observed=True)
    _ = grp["kills"].transform(
        lambda s: s.where(mask.loc[s.index], 0.0).cumsum().shift()
    )


pd = _load_pd()

__all__ = ["pd"]
