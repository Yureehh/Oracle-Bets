# Runtime state

This folder separates mutable/append-only operational state from generated LoL
datasets.

- `oracle_bets.db`: the single append-only evidence database for predictions,
  market observations, paper decisions, settlement, and performance.
- `model-registry/lol/`: immutable, checksum-verified model bundles and the
  champion pointer. This is intentionally distinct from `models/lol/`, which is
  the mutable training workspace.
Database files and model bundles are generated and ignored. The README is the
only required tracked file in a fresh clone.
