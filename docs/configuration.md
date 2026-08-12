# Configuration

`config/product/product.json` defines `Europe/Rome`, the fixture window, the
read-only market boundary, training/actionable league profiles, and retraining
triggers. Training and ratings use `research_all_supported`; daily owner output
uses `tier1_plus_erls` with LCP and CBLOL excluded.

`config/lol/data_ingestion/team_aliases.json` separates current provider aliases
from reviewed historical identity merges. Unknown identity never justifies
fabricated history. `import_columns.json` is the source allowlist. Training
configs retain reconstruction metadata such as split/game but the persisted
feature pipeline excludes identifiers, targets, side, draft, and post-start
state from the direct prematch model.

Default rating/search priors live under `config/lol/hyperparameters/defaults/`.
Reviewed production values live under `hyperparameters/tuned/ratings/` and
`hyperparameters/tuned/lightgbm/`. Routine training requires reviewed files and
never starts Optuna. Optuna output remains isolated until owner review and
`lol promote-tuning`.

Use the ignored `0600` `.env` for:

- `PANDASCORE_API_KEY`;
- `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID`, `DISCORD_OWNER_USER_ID`;
- `DISCORD_DELIVERY_MODE=gateway|off`;
- optional `OPENAI_API_KEY` and `OPENAI_MODEL`;
- optional `ORACLES_ELIXIR_LOCAL_DIR`.

AWS and Discord webhook variables are obsolete. Never commit secrets, private
keys, or wallet material. Rotate any credential exposed in chat, screenshots,
logs, or commits.
