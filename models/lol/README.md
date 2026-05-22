# models/lol

Generated League of Legends model artifacts for `lol-bets`.

- `OutcomePrediction_<ModelType>/`: trained outcome model and preprocessing artifacts.
- `league_elo.parquet`: learned per-league Elo plus learned `strength_pool` Elo.
- `team_league_mapping.parquet`: latest team-to-home-league mapping with pool metadata.

Example:

```bash
uv run oracle-bets lol train --model-type lightgbm
```
