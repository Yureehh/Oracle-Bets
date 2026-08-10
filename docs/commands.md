# Command runbook

Run every command from the repository root. `uv run` loads the project
environment and the ignored `.env`; inspect any command with
`uv run oracle-bets <domain> <command> --help`.

## Day-by-day playbook

| When | Run | Purpose |
| --- | --- | --- |
| Normal day, around 00:15 Rome time | `uv run oracle-bets daily lol` | Refresh history, conditionally retrain, validate, predict, read public markets, write one report pair, and notify Discord. |
| Before trusting a changed setup | `uv run oracle-bets daily lol --dry-run` | Exercise schedule, prediction, quotes, gates, and reports without ingestion, training, evidence writes, model promotion, or Discord. |
| Monday | Commands printed by the daily reminder | Review open positions, system health, and previous-week performance. |
| Thursday | Commands printed by the daily reminder | Review open positions, current-week performance, exposure, and market failures. |
| First calendar day | Commands printed by the daily reminder | Run the previous-month audit and paper-performance review. |
| After routine training | `model list`, then `model review <id>` | Confirm whether the candidate was safely promoted or which gate retained the champion. |
| After Optuna retuning | Review the run, then `lol promote-tuning <run-id>` only if justified | Retuning never promotes parameters or a model automatically. Retrain afterward. |
| After accepting a paper proposal | Keep the Gateway bot running or inspect `paper list --state open` | Capture closing evidence and later settle manually. |

Monday, Thursday, and first-of-month reminders use `Europe/Rome`. When they
overlap, the daily report and Discord receive one combined advisory. Reminders
never run audits, promote models, suppress predictions, or settle positions.

## Installation and configuration

```bash
uv sync --extra discord-bot
uv run oracle-bets --help
```

Required network credentials are `PANDASCORE_API_KEY` for schedules and
`DISCORD_TOKEN`, `DISCORD_CHANNEL_ID`, and `DISCORD_OWNER_USER_ID` for the
interactive bot. `DISCORD_WEBHOOK_URL` enables one-way reports.
`OPENAI_API_KEY` is optional: without it, deterministic output is unchanged.
Set exactly one `DISCORD_DELIVERY_MODE=gateway|webhook|off`. Gateway is the
interactive bot and is the intended mode for this installation; webhook sends
one-way reports only. The workflow never uses both transports in one run.
Keep `.env` ignored and run `chmod 600 .env`. Public Polymarket discovery and
books need no trading credentials. No command places orders, signs payloads,
uses wallets, or moves funds.

## LoL data, schedules, and models

| Command | When and effects | Important options |
| --- | --- | --- |
| `lol health` | Read-only check of training and serving artifacts. No network. | None. |
| `lol source-check` | Read-only preflight for the local 2024–2026 Oracle's Elixir snapshot. Blocks stale, missing, placeholder, or actively changing files before rebuilding. | `--format table|json`. |
| `lol ingest` | Routine local Google Drive Oracle's Elixir refresh for `research_all_supported`; writes raw/interim/processed data and manifests. Does not train or tune. | None. |
| `lol reconcile-history` | Destructive full reconciliation of retained source history after source/schema changes; writes data artifacts. Does not train or tune. | None. |
| `lol sync-identities` | Rebuild canonical player/team/league/series/map identity evidence from retained raw data. Writes evidence. | None. |
| `lol schedule` | Fetch and print PandaScore fixtures only; does not write the schedule or train. | `--days N`, `--leagues LCK,LEC`. |
| `lol validate-data` | Read-only validation of generated supervised tables and feature contracts. | None. |
| `lol market-check` | Public, read-only Gamma/CLOB check. Reports typed match, token orientation, minimum shares, hypothetical cost, executable odds, and isolated token failures. Two observations use one shared 45-second wait. | `--match-key <pandascore-key>` checks that fixture's supported typed market. |
| `lol train` | Routine refit with reviewed parameters; retrains weights and calibrators, registers an immutable candidate, and may auto-promote only a healthy non-inferior non-Optuna bundle. It does not run Optuna. | `--targets all|outcome|props|<names>`, `--feature-set full|compact|selected`, `--max-features N`, `--feature-selection none|importance|cumulative|report`, calibration/split options. |
| `lol retune` | Explicit Optuna research search. Writes isolated tuning reports and artifacts; never updates reviewed parameters or champion automatically. | `--targets`, `--feature-set`, `--max-features`. |
| `lol promote-tuning <run-id>` | Owner promotion of one reviewed, complete tuned-parameter bundle. Writes production parameter files, not model weights. Run `lol train` afterward. | Complete run ID. |

Examples:

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol validate-data
uv run oracle-bets lol schedule --days 2
uv run oracle-bets lol market-check --match-key <pandascore-key>

# Routine retraining: no Optuna
uv run oracle-bets lol train --targets all --feature-set compact

# Rare research retuning, manual parameter promotion, then compatible retrain
uv run oracle-bets lol retune --targets all --feature-set compact
uv run oracle-bets lol promote-tuning <run-id>
uv run oracle-bets lol train --targets all --feature-set compact
```

Ingestion, retraining, and retuning are separate operations. Retraining updates
model weights/calibrators on newer data using fixed reviewed hyperparameters.
Retuning searches hyperparameters and carries greater overfitting risk.

## Candidate registry

| Command | Effect |
| --- | --- |
| `model status [--registry PATH]` | Read champion and registry health. |
| `model list [--format table|json] [--registry PATH]` | List immutable candidates. |
| `model review <model-id> [--format table|json]` | Show manifest, evidence, gates, and artifacts. |
| `model register-run <run-id|latest>` | Confirm an automatically registered training run. |
| `model register-current <id> --code-version <sha> --metric name=value [...]` | Freeze the current complete inference tree as a manual candidate; optional `--target`, `--random-seed`, `--registry`. |
| `model promote <id> --reason <text>` | Explicitly move the champion pointer after owner review. |
| `model rollback <id> --reason <text>` | Explicitly restore a prior healthy bundle. |

The first champion is manual. Later routine candidates may auto-promote only
after sealed-row non-inferiority, calibration, cohort, regression-target, and
artifact gates pass. Optuna-derived candidates always remain manual.

## Daily workflow

`daily lol` is the only daily workflow command.

```bash
# Full normal workflow; network + data/model/evidence/report writes + optional Discord
uv run oracle-bets daily lol

# No ingestion/training/evidence/promotion/Discord; writes one JSON/Markdown pair
uv run oracle-bets daily lol --dry-run

# Diagnostics when public markets are intentionally unavailable
uv run oracle-bets daily lol --dry-run --skip-market-search

# Normal run without conditional routine retraining
uv run oracle-bets daily lol --skip-retrain
```

Useful options are `--horizon-hours`, `--leagues`, `--webhook-url`,
`--targets`, `--feature-set`, `--max-features`, `--skip-market-search`,
`--ai-review`/`--no-ai-review`, and `--openai-model`. The product default is a
36-hour `tier1_plus_erls` prediction universe excluding configured non-actionable
leagues. Training still uses `research_all_supported`.

Every run writes exactly one pair under `reports/lol/daily/`. The report keeps
contract matching in `market_reviews` and outcome quotes/gates in
`market_actions`. Valid odds remain visible for blocked and no-edge outcomes;
one token failure cannot erase other quotes.

The daily command never settles paper positions. It only reports the open
count and points to the manual owner workflow.

## Evidence database

| Command | Effect |
| --- | --- |
| `evidence init [--database PATH]` | Create/migrate the append-only evidence schema. |
| `evidence health [--database PATH]` | Read-only integrity/coverage check. |
| `evidence backup [--database PATH] [--output DIR]` | Create a consistent backup. |
| `evidence export [--database PATH] [--output DIR] [--format json|csv]` | Export review copies. |
| `evidence restore-verify <backup.db>` | Verify a backup without replacing the active database. |

Back up evidence before destructive rebuilds:

```bash
uv run oracle-bets evidence health
uv run oracle-bets evidence backup
uv run oracle-bets evidence export --format json
```

## Paper decisions, closing lines, and manual settlement

```bash
uv run oracle-bets paper list --state pending --format table
uv run oracle-bets paper show <proposal-or-position-id> --format json
uv run oracle-bets paper decide \
  --proposal-id <id> --decision accept --reason "owner paper review"
uv run oracle-bets paper list --state open

uv run oracle-bets paper settle \
  --position-id <id> \
  --result win \
  --source-reference <result-url-or-id> \
  --note "optional verification note"
```

Settlement results are `win`, `loss`, `push`, and `void`. Win PnL is
`stake × (odds - 1)`, loss is `-stake`, and push/void are zero. Odds and stake
come from the immutable position; do not re-enter them. Every settlement is
`owner_verified`, requires a source reference, is idempotent for the same
result/source, and rejects conflicting attempts. There is deliberately no
`paper reconcile` command and no provider automatically closes positions.

Other paper commands:

| Command | Effect |
| --- | --- |
| `paper list` | Filter lifecycle rows with `--state pending|open|settled`, `--target`, `--league`, `--format`, or `--database`. Read-only. |
| `paper show <id>` | Read one proposal or position. |
| `paper quote-prop --forecast-id ... --line ... --over-odds ... --under-odds ... --source ...` | Write a research-only scalar prop proposal using the forecast's exact calibrator. |
| `paper decide --proposal-id ... --decision accept|reject` | Append an owner decision; acceptance opens one paper position. Options: `--reason`, `--actor-id`, `--database`. |
| `paper settle ...` | Append one manual settlement. Options: `--note`, `--actor-id`, `--settled-at`, `--database`; `--dry-run` writes nothing. |
| `paper capture-closing [--window-minutes 15] [--dry-run] [--format table|json]` | Read public books and write fully fillable pre-start closing observations for open Polymarket positions. Never settles or trades. |
| `paper performance [--since ISO] [--target X] [--league X] [--format table|json]` | Read ROI, CLV, drawdown, calibration, uncertainty, and cohorts. |

For manual props:

```bash
uv run oracle-bets paper quote-prop \
  --forecast-id <id> --line 26.5 \
  --over-odds 1.91 --under-odds 1.91 --source bookmaker
```

## Discord

| Command | Effect |
| --- | --- |
| `discord doctor` | Validate local configuration without network calls. |
| `discord doctor --live` | Read the bot/channel identity and verify channel-history access for crash recovery; sends no message. |
| `discord run` | Run the always-on Gateway bot for owner-only proposal and settlement controls and closing captures. |
| `discord publish --run-id latest|<id>` | Send one saved report through the configured one-way webhook. |

The Gateway bot provides Accept/Reject and Win/Loss/Push/Void controls.
Settlement buttons open a required source-reference modal. Controls persist
across restarts and disable after settlement. The LLM never sees or influences
settlement. A webhook cannot host controls; webhook-only users settle by CLI.
Do not run the separate closing-line launchd job when the Gateway bot is doing
the same capture.
Before each new card is sent, the bot records a durable intent and embeds a
deterministic marker. After a crash it searches bounded channel history and
records the existing message instead of sending a duplicate. A process lock
also rejects a second Gateway instance.

## Health and audit

```bash
uv run oracle-bets health system
uv run oracle-bets health system --record --output reports/health
uv run oracle-bets audit monthly --period 2026-07
```

`health system` reads operational checks; `--record` appends evidence and
`--output` writes a report. `audit monthly` writes and records the light owner
audit; `--database` and `--output` override locations. Neither command promotes
models, settles positions, or changes market state.

## Tests and documentation

```bash
uv run pytest tests/core tests/lol
uv run ruff check packages tests
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-data
uv run oracle-bets lol market-check
uv run oracle-bets evidence health
uv run oracle-bets discord doctor
```

`lol market-check` uses public network data and waits once between observations.
The other health checks are read-only. Do not use ingestion, training, or
retuning as routine code-change verification unless the change actually
requires regenerating those artifacts.

## launchd and recovery

Supported macOS templates and installation commands are in `ops/README.md`.
Use the Gateway-bot template for interactive
controls or the closing-line template for webhook-only operation, never both.

The clean research rebuild sequence is in
[Getting started](getting_started.md#first-clean-research-rebuild). It deletes
generated data, models, reports, logs, evidence, and registry state, so back up
`data/state/` first. A clean rebuild is disaster recovery/research setup, not a
normal retrain.
