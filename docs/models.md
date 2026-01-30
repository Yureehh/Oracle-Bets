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

`src/2_models_training.py` trains Gradient Boosting models (LightGBM) for:

- Outcome prediction (classification)
- Gamelength prediction (regression)
- Total kills prediction (regression)
- Total towers prediction (regression)

Trained models and metadata are stored under `models/<ModelName>/`.
