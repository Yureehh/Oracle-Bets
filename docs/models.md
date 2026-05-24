# Models

The project blends rating systems with supervised models.

## Ratings

Computed for both teams and players:

- Elo
- Glicko-2
- Plackett-Luce
- TrueSkill

Ratings are used directly as features and as win-likelihood signals.

## Supervised models

`oracle-bets lol train --model-type lightgbm --targets all` trains Gradient
Boosting models for:

- Outcome prediction (classification)
- Gamelength prediction (regression)
- Total kills prediction (regression)
- Total towers prediction (regression)

Trained models and metadata are stored under `models/lol/<ModelName>/`.

Useful target selectors:

```bash
uv run oracle-bets lol train --targets outcome
uv run oracle-bets lol train --targets props
uv run oracle-bets lol train --targets total_kills,total_towers
```

Use `--force-retune` to ignore cached best hyperparameters and run Optuna again
for the selected targets:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets all --force-retune
```

Feature-set controls:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets outcome --feature-set full
uv run oracle-bets lol train --model-type lightgbm --targets outcome --feature-set compact
uv run oracle-bets lol train --model-type lightgbm --targets outcome --feature-set selected --max-features 120
```

`full` keeps the full generated surface. `compact` applies the curated compact configs at train time. `selected` uses the latest feature-selection report and keeps mandatory anchor features before filling the remaining slots up to `--max-features`. Prop models train one row per map because game length, total kills, and total towers are game-level targets.

Classification artifacts include a validation-fitted probability calibrator when
the validation split has enough samples. Regression artifacts include residual
summaries with sigma, MAE, RMSE, and percentiles; these power Discord over/under
pricing for kills, towers, and game length.
