# Oracle Bets

Oracle Bets is a League of Legends match modeling pipeline. It ingests match data, computes ratings and features, trains prediction models, and powers inference workflows (including a Discord bot).

## Features

- Oracle's Elixir data ingestion and cleaning
- League, team, and player ratings (Elo, Glicko2, Plackett-Luce, TrueSkill)
- Leak-free feature engineering and EMA metrics
- Gradient Boosting models for outcomes and regression targets
- Match predictor and Discord bot

## Quickstart

### Install

```bash
uv sync --extra dev
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure environment

Create a `.env` file (do not commit) with:

```bash
BUCKET_NAME=your-oracles-elixir-bucket
ACCESS_ID=your-aws-access-key
SECRET_ID=your-aws-secret
PANDASCORE_API_KEY=your-pandascore-token
DISCORD_TOKEN=your-discord-bot-token
```

### Run the pipeline

```bash
python src/1_data_generator.py
python src/2_models_training.py
```

Optional:

```bash
python src/0_schedule_update.py
python src/3_oracle_bot.py
```

## Documentation

Docs are built with MkDocs Material.

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000/` in a browser.

## Project layout

- `src/0_schedule_update.py` - PandaScore schedule fetcher
- `src/1_data_generator.py` - data ingestion and feature pipeline
- `src/2_models_training.py` - model training
- `src/3_oracle_bot.py` - Discord bot

## License

Apache-2.0. If you want a different license, update `LICENSE` and `pyproject.toml`.

## Acknowledgements

Please visit and support [Oracle's Elixir](https://www.oracleselixir.com), which provides the backbone data source behind this project. Thanks to Tim Sevenhuysen, BuckeyeSundae, TZero, Addie Thompson, and the Oracle's Elixir Data Science community for their guidance and feedback.

## Disclaimer

This project is for research and analytics. It is not financial advice, and any betting or wagering use is at your own risk.
