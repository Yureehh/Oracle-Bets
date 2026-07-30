# Architecture

Oracle Bets has three small Python packages:

- `oracle_bets_core`: CLI, paths, product config, evidence database, health,
  settlement, performance, backups, and read-only market adapters;
- `lol_bets`: LoL ingestion, features, ratings, training, calibration,
  inference, lifecycle, and daily orchestration;
- `oracle_bets_discord`: one-way LoL message formatting.

There is no dashboard package, API server, interactive bot, watcher, live
execution package, TabNet path, or WHR experiment.

## End-to-end flow

```text
Google Drive Desktop (Oracle's Elixir 2024–2026)
  -> history reconciliation and quarantine
  -> team/player feature and rating tables
  -> validated supervised tables
  -> explicit retune candidates OR routine refit
  -> temporal calibration and model reports
  -> mutable models/lol workspace
  -> optional immutable model registry/champion

PandaScore schedule + expected lineups
  -> conservative team resolution
  -> canonical matchup features
  -> winner/series/prop predictions
  -> read-only Polymarket candidate comparison
  -> JSON + Markdown report
  -> optional one-way Discord webhook
  -> append-only evidence and later settlement/performance review
```

## Storage boundaries

- `data/lol/`: generated source and feature datasets.
- `models/lol/`: mutable complete training workspace.
- `data/state/oracle_bets.db`: canonical append-only operational evidence.
- `data/state/model-registry/lol/`: immutable checksum-verified model bundles.
- `reports/lol/`: review artifacts; notebooks read these.
- `logs/lol/oracle-bets.log`: bounded rotating runtime log.

The registry is not a duplicate models folder: the workspace is replaceable,
while the registry preserves exactly what was reviewed and served.
