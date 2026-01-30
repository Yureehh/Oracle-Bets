# Getting Started

This project has four main runnable scripts in `src/`:

1. `src/0_schedule_update.py` - fetch upcoming schedules from PandaScore.
2. `src/1_data_generator.py` - ingest and prepare historical data.
3. `src/2_models_training.py` - train prediction models.
4. `src/3_oracle_bot.py` - run the Discord bot.

The data pipeline writes intermediate and final artifacts into `data/` and `models/`. Configuration lives in `config/`.

If you are new to the codebase, start with `src/1_data_generator.py` to understand how data is ingested and prepared, then `src/2_models_training.py` for model training, and `src/discord_predictions/match_predictor.py` for inference logic.
