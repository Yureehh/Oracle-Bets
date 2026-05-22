# training

Feature-column configurations for LoL training and inference artifacts.

- `training_*_config.json`: broad supervised-training columns, favoring explicit `diff_ema_*` comparison features.
- `training_compact_*_config.json`: curated compact baselines for faster experiments.
- `flattened_*_config.json`: own-team/player state materialized for inference-time diff construction.

Example:

```bash
TRAINING_CONFIG_VARIANT=compact uv run oracle-bets lol ingest
uv run oracle-bets lol train --model-type lightgbm --feature-selection report
```
