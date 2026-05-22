# Predictions

## Match predictor

`packages/lol-bets/src/lol_bets/inference/match_predictor.py` assembles inference features to produce:

- Rating-based probabilities
- Model-based outcome predictions
- Auxiliary regressions (gamelength, totals)

It mirrors training-time transforms to avoid feature drift.

Outcome predictions use a validation-fitted calibration artifact when one is
available. Prop predictions use regression residual artifacts to convert a mean
projection into an over/under probability for a betting line.

## Discord bot

`packages/oracle-bets-discord/src/oracle_bets_discord/bot.py` uses the predictor to serve predictions via Discord. It expects:

- `DISCORD_TOKEN` in the environment
- Trained model artifacts under `models/lol/`

The bot also exposes utility commands like Kelly criterion suggestions.

Market-ready LoL commands live under `!lol`:

```text
!lol predict "Team WE" "LNG Esports" --bo5
!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85
!lol edge "Team WE" "LNG Esports"
```

See [LoL Market Predictions](lol_market_predictions.md) for command examples
and definitions of probability, fair odds, edge, and half-Kelly.
