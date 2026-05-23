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

Classification artifacts include a validation-fitted probability calibrator when
the validation split has enough samples. Regression artifacts include residual
summaries with sigma, MAE, RMSE, and percentiles; these power Discord over/under
pricing for kills, towers, and game length.
