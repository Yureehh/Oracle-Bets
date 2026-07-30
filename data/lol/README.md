# LoL generated data

- `raw/raw_data.parquet`: reconciled 2024–2026 Oracle's Elixir source history.
- `interim/`: enriched team/player tables.
- `processed/`: validated supervised training tables and flattened inference state.
- `raw/oracles_elixir`: local symlink to Google Drive Desktop; never committed.

Run `oracle-bets lol reconcile-history` for a full source refresh or
`oracle-bets lol ingest` for the normal incremental pipeline.
