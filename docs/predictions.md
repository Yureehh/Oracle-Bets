# Predictions

## Match predictor

`packages/lol-bets/src/lol_bets/inference/match_predictor.py` assembles inference features to produce:

- Rating-based probabilities
- Model-based outcome predictions
- Auxiliary regressions (gamelength, totals)

It mirrors training-time transforms to avoid feature drift.

## Discord bot

`packages/oracle-bets-discord/src/oracle_bets_discord/bot.py` uses the predictor to serve predictions via Discord. It expects:

- `DISCORD_TOKEN` in the environment
- Trained model artifacts under `models/lol/`

The bot also exposes utility commands like Kelly criterion suggestions.
