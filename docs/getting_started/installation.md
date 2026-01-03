# Installation

## Requirements

- Python 3.11+
- One of: uv (recommended) or pip

## Install with uv

```bash
uv sync --extra dev
```

## Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment variables

Create a `.env` file (not committed) with the required keys:

```bash
BUCKET_NAME=your-oracles-elixir-bucket
ACCESS_ID=your-aws-access-key
SECRET_ID=your-aws-secret
PANDASCORE_API_KEY=your-pandascore-token
DISCORD_TOKEN=your-discord-bot-token
```

Only the first three are required for the data pipeline. The PandaScore and Discord tokens are needed for schedule updates and the bot.

## Pre-commit (optional)

```bash
pre-commit install
```
