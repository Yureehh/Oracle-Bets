"""Contracts shared by Oracle Bets prediction modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ArtifactCheck:
    """Health result for one required artifact."""

    name: str
    path: str
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ArtifactHealth:
    """Aggregate health result for a prediction module."""

    module_id: str
    checks: tuple[ArtifactCheck, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def raise_if_unhealthy(self) -> None:
        if self.ok:
            return
        failed = [
            f"{check.name}: {check.path} ({check.reason or 'missing/unreadable'})"
            for check in self.checks
            if not check.ok
        ]
        msg = f"Artifact health check failed for {self.module_id}: " + "; ".join(failed)
        raise RuntimeError(msg)


class PredictionModule(Protocol):
    """Minimal surface every sport/e-sport prediction module should expose."""

    id: str

    def artifact_health(self) -> ArtifactHealth:
        """Return required-artifact health without mutating state."""

    def predict_match(self, *args: Any, **kwargs: Any) -> Any:
        """Predict a match outcome."""


class PropsPredictionModule(PredictionModule, Protocol):
    """Optional extension for modules that expose secondary markets/props."""

    def predict_props(self, *args: Any, **kwargs: Any) -> Any:
        """Predict secondary markets/props."""
