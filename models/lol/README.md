# models/lol

Generated League of Legends model artifacts for `lol-bets`.

- `OutcomePrediction_<ModelType>/`: trained outcome model and preprocessing artifacts.
- `league_elo.parquet`: league strength ratings.
- `team_league_mapping.parquet`: latest team-to-league mapping.
- `league_strength_priors.json`: calibrated league prior offsets.

Example:

```bash
uv run oracle-bets lol train --model-type lightgbm
```
