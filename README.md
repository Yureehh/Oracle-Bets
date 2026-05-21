# Oracle Bets

Oracle Bets is a modular sports and e-sports prediction suite. The first module, `lol-bets`, ingests League of Legends data, computes ratings and features, trains prediction models, and serves inference through a shared Discord bot.

## Features

- Oracle's Elixir data ingestion and cleaning
- League, team, and player ratings (Elo, Glicko2, Plackett-Luce, TrueSkill)
- Leak-free feature engineering and EMA metrics
- Gradient Boosting models for outcomes and regression targets
- Match predictor and Discord bot
- Read-only prediction-market discovery plus edge and half-Kelly sizing signals

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
oracle-bets lol ingest
oracle-bets lol train
```

Optional:

```bash
oracle-bets lol health
oracle-bets discord run
```

Legacy `src/` entrypoints and import shims have been removed. Use the package
imports and `oracle-bets` CLI above.

### Scoped Files

LoL-owned files live under module subdirectories: `config/lol/`, `data/lol/`,
`models/lol/`, `notebooks/lol/`, `reports/lol/`, `logs/lol/`, and `tests/lol/`.
Override the suite root with `ORACLE_BETS_HOME`, the LoL root with
`ORACLE_BETS_LOL_HOME`, or a specific directory with `ORACLE_BETS_DATA_DIR`,
`ORACLE_BETS_MODELS_DIR`, `ORACLE_BETS_REPORTS_DIR`, or `ORACLE_BETS_LOGS_DIR`.

## Documentation

Docs are built with MkDocs Material.

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000/` in a browser.

## Project layout

- `packages/oracle-bets-core/` - shared paths, logging, pandas/FireDucks shim, module contracts, betting math, market adapters, and CLI
- `packages/lol-bets/` - League of Legends ingestion, feature engineering, ratings, training, and inference
- `packages/oracle-bets-discord/` - Discord bot, formatting, command routing, and module registry
- `docs/audits/` - model and data-quality audits that guide algorithm changes
- `config/lol/` - League of Legends configuration
- `data/lol/`, `models/lol/`, `reports/lol/`, `logs/lol/` - generated LoL artifacts
- `tests/lol/` - current unit and smoke tests

## License

Apache-2.0. If you want a different license, update `LICENSE` and `pyproject.toml`.

## Acknowledgements

Please visit and support [Oracle's Elixir](https://www.oracleselixir.com), which provides the backbone data source behind this project. Thanks to Tim Sevenhuysen, BuckeyeSundae, TZero, Addie Thompson, and the Oracle's Elixir Data Science community for their guidance and feedback.

## Disclaimer

This project is for research and analytics. It is not financial advice, and any betting or wagering use is at your own risk.
