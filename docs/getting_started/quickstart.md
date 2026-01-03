# Quickstart

This assumes you have a `.env` file with the required keys and have installed dependencies.

## 1) Build data artifacts

```bash
python src/1_data_generator.py
```

This writes processed data to `data/processed/` and rating artifacts to `models/artifacts/`.

## 2) Train models

```bash
python src/2_models_training.py
```

Models are stored in `models/artifacts/<ModelName>/`.

## 3) Update schedule (optional)

```bash
python src/0_schedule_update.py
```

Requires `PANDASCORE_API_KEY`.

## 4) Run the Discord bot (optional)

```bash
python src/3_oracle_bot.py
```

Requires `DISCORD_TOKEN` and trained model artifacts.

## Compact training config

To use the compact training feature set, set:

```bash
export TRAINING_CONFIG_VARIANT=compact
```
