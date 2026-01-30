# Predictions

## Match predictor

`src/discord_predictions/match_predictor.py` assembles inference features to produce:

- Rating-based probabilities
- Model-based outcome predictions
- Auxiliary regressions (gamelength, totals)

It mirrors training-time transforms to avoid feature drift.

## Discord bot

`src/3_oracle_bot.py` uses the predictor to serve predictions via Discord. It expects:

- `DISCORD_TOKEN` in the environment
- Trained model artifacts under `models/`

The bot also exposes utility commands like Kelly criterion suggestions.
