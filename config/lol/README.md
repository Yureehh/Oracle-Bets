# config/lol

League of Legends configuration for `lol-bets`.

- `data_ingestion/`: Oracle's Elixir columns, league filters, taxonomy, and cleanup rules.
- `training/`: full and compact feature lists for training and flattened inference tables.
- `hyperparameters/`: default and tuned rating/model parameters.

Example:

```bash
TRAINING_CONFIG_VARIANT=compact uv run oracle-bets lol ingest
```
