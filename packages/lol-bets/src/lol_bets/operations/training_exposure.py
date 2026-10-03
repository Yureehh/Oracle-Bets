"""Durable exposure records, independent of disposable training reports."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import TYPE_CHECKING

from oracle_bets_core.io_utils import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path


def utc_date(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def initialize_history(runs_root: Path, *, now: dt.datetime | None = None) -> dict:
    """Freeze a future evaluation boundary; missing history never means pristine."""
    root = runs_root.parent / "holdout-exposure"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "baseline.json"
    if not path.exists():
        payload = {
            "schema_version": 1,
            "history_status": "legacy_coverage_unknown",
            "baseline_cutoff": (now or dt.datetime.now(dt.UTC)).isoformat(),
        }
        # Exclusive creation prevents simultaneous callers moving the cutoff.
        try:
            with path.open("x", encoding="utf-8") as output:
                json.dump(payload, output, sort_keys=True)
        except FileExistsError:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    utc_date(payload["baseline_cutoff"])
    return payload


def record_exposure(runs_root: Path, payload: dict) -> None:
    """Append content-addressed evidence; changed inputs retain their old exposure."""
    initialize_history(runs_root)
    normalized = payload | {"date_max": utc_date(payload["date_max"]).isoformat()}
    text = json.dumps(normalized, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    atomic_write_text(runs_root.parent / "holdout-exposure" / f"{digest}.json", text)


def read_exposures(runs_root: Path) -> list[dict]:
    initialize_history(runs_root)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((runs_root.parent / "holdout-exposure").glob("*.json"))
        if path.name != "baseline.json"
    ]


def reserve_exposure(run_root: Path, *, target: str, date_max: str) -> None:
    """Record a conservative upper bound before training can inspect any outcome."""
    record_exposure(
        run_root.parent,
        {
            "run_id": run_root.name,
            "target": target,
            "date_max": date_max,
            "kind": "reservation",
            "review_input_sha256": None,
        },
    )
