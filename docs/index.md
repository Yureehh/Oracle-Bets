# Oracle Bets

Oracle Bets is a modular sports and esports prediction suite. Its first module,
`lol-bets`, ingests League of Legends match data, computes ratings and features,
trains prediction models, and serves inference through shared Discord commands.

## What it does

- Ingests historical match data from Oracle's Elixir (S3 CSVs).
- Builds league, team, and player ratings (Elo, Glicko2, Plackett-Luce, TrueSkill).
- Generates leak-free feature tables for training and inference.
- Trains Gradient Boosting models for outcomes and regression targets.
- Produces match predictions and optional Discord bot outputs.

## Data sources

- Oracle's Elixir match data (primary historical source).
- PandaScore API for upcoming schedules.

## Disclaimer

This project is for research and analytics. It is not financial advice, and any betting or wagering use is at your own risk.

## Where to start

- Getting Started for setup and quickstart steps.
- Pipeline for how data moves from ingestion to features.
- Models for how ratings and GBDT models are trained.
- Predictions for inference and Discord bot usage.
