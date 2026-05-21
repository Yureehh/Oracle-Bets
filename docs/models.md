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

`oracle-bets lol train` trains Gradient Boosting models (LightGBM) for:

- Outcome prediction (classification)
- Gamelength prediction (regression)
- Total kills prediction (regression)
- Total towers prediction (regression)

Trained models and metadata are stored under `models/lol/<ModelName>/`.
