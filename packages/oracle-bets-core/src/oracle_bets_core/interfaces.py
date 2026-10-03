"""Contracts shared by Oracle Bets prediction modules."""

from __future__ import annotations

from dataclasses import dataclass, field


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
