# training

Feature-column configurations for LoL training and inference artifacts.

- `training_*_config.json`: columns used for supervised training.
- `training_compact_*_config.json`: smaller experimental feature sets.
- `flattened_*_config.json`: columns materialized for inference state.

Example:

```bash
TRAINING_CONFIG_VARIANT=compact uv run oracle-bets lol ingest
```
