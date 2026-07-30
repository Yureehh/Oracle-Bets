# prediction_models

Model preprocessing, training, selection, and observability.

- `data_preprocessor.py`: joins team and player artifacts.
- `gbdt_model.py`: shared GBM training/evaluation pipeline and explicit EMA-diff construction.
- `feature_selector.py`: feature-selection helpers and compact recommendation reports.
- `lightgbm_model.py`: primary baseline model.

Known model targets are owned in `data_preprocessor.py` so target cleanup stays
beside the training-table assembly that uses it.

Enabled targets are:

- `outcome`: map winner classification.
- `gamelength`: expected map duration in minutes.
- `total_kills`: expected combined champion kills.
- `total_towers`: expected combined towers destroyed.

Classification stores a probability calibrator when validation data supports it.
Regression stores residual summaries for Discord over/under line pricing.

Example:

```bash
uv run oracle-bets lol train --targets all --feature-selection report
```
