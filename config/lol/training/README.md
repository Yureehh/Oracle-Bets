# training

Feature-column configurations for LoL training and inference artifacts.

- `training_*_config.json`: broad supervised-training columns, favoring explicit `diff_ema_*` comparison features.
- `training_compact_*_config.json`: curated compact baselines for faster experiments; feature-selection reports recommend changes but do not overwrite them.
- `flattened_*_config.json`: own-team/player state materialized for inference-time diff construction.

The team configs include focused style metrics for snowballing, closing speed,
team vision, objective conversion, and lead conversion. Comparable stats are
usually consumed as `diff_ema_*`; closing-speed EMAs stay as own-team context.

Example:

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol train --model-type lightgbm --feature-selection report
uv run oracle-bets lol train --model-type lightgbm --feature-set compact
uv run oracle-bets lol train --model-type lightgbm --feature-set selected --max-features 120
```
