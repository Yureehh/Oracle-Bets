# prediction_models

Model preprocessing, training, selection, and observability.

- `data_preprocessor.py`: joins team and player artifacts.
- `gbdt_model.py`: shared GBM training/evaluation pipeline.
- `lightgbm_model.py`: primary baseline model.
- `tabnet_model.py`: optional neural baseline.

Example:

```bash
uv run oracle-bets lol train --model-type lightgbm --feature-selection none
```
