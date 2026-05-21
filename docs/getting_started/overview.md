# Getting Started

This project is driven by one CLI and three packages:

1. `oracle-bets-core` - shared paths, logging, CLI, betting math, and markets.
2. `lol-bets` - League of Legends ingestion, features, ratings, training, and inference.
3. `oracle-bets-discord` - Discord bot shell and command delivery.

The LoL module owns scoped folders under `config/lol/`, `data/lol/`,
`models/lol/`, `notebooks/lol/`, `reports/lol/`, `logs/lol/`, and `tests/lol/`.
Future sports or esports modules can sit beside it without sharing artifacts or
configuration files.

If you are new to the codebase, start with `packages/lol-bets/src/lol_bets/pipeline.py`,
then `packages/lol-bets/src/lol_bets/training.py`, then
`packages/lol-bets/src/lol_bets/inference/match_predictor.py`.
