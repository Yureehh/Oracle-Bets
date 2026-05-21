"""Prediction-module registry used by the Discord layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_bets_core.interfaces import PredictionModule


class ModuleRegistry:
    """Small in-process registry for prediction modules."""

    def __init__(self) -> None:
        self._modules: dict[str, PredictionModule] = {}

    def register(self, module: PredictionModule) -> None:
        self._modules[module.id] = module

    def get(self, module_id: str) -> PredictionModule:
        try:
            return self._modules[module_id]
        except KeyError:
            msg = f"Prediction module not registered: {module_id}"
            raise KeyError(msg) from None

    def all(self) -> tuple[PredictionModule, ...]:
        return tuple(self._modules.values())


def default_registry() -> ModuleRegistry:
    from lol_bets.module import LoLBetsModule

    registry = ModuleRegistry()
    registry.register(LoLBetsModule())
    return registry
