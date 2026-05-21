# reports/lol

Generated League of Legends evaluation outputs.

- `ingestion/`: generated league/team metadata from raw ingestion data.
- `evaluation_insights/`: metrics, predictions, model cards, feature importance, and cohort tables.
- `figures/`: plots for accuracy, calibration, SHAP, and cohort diagnostics.

Example:

```bash
uv run oracle-bets lol train --model-type lightgbm
```
