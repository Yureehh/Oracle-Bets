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
interactive bot.
`OPENAI_API_KEY` is optional: without it, deterministic output is unchanged.
Set `DISCORD_DELIVERY_MODE=gateway` (or `off` for local diagnostics). Gateway is
the sole delivery path, so a legacy webhook cannot duplicate messages.
Keep `.env` ignored and run `chmod 600 .env`. Public Polymarket discovery and
books need no trading credentials. No command places orders, signs payloads,
uses wallets, or moves funds.

## LoL data, schedules, and models

| Command | When and effects | Important options |
| --- | --- | --- |
| `lol health` | Read-only check of training and serving artifacts. No network. | None. |
| `lol source-refresh` | Request a temporary anonymous ZIP for the reviewed 2024–2026 public Google Drive files and promote validated CSVs into an atomic local cache. Network + cache/manifest writes; never uses AWS, credentials, or Drive mutations. | `--format table|json`. |
| `lol source-check` | Read-only preflight for the managed 2024–2026 Oracle's Elixir cache. Blocks stale, missing, placeholder, or actively changing files before rebuilding. | `--format table|json`. |
| `lol ingest` | Refresh the public source, then run routine `research_all_supported` ingestion; writes raw/interim/processed data and manifests. Does not train or tune. | None. |
| `lol reconcile-history` | Destructive full reconciliation of retained source history after source/schema changes; writes data artifacts. Does not train or tune. | None. |
| `lol sync-identities` | Rebuild canonical player/team/league/series/map identity evidence from retained raw data. Writes evidence. | None. |
| `lol schedule` | Fetch and print PandaScore fixtures only; does not write the schedule or train. | `--days N`, `--leagues LCK,LEC`. |
| `lol validate-data` | Read-only validation of generated supervised tables and feature contracts. | None. |
| `lol build-series` | Reconstruct deterministic complete BO1/BO3/BO5 series, write one frozen prematch row per accepted series plus the next-map shadow dataset, and write an explicit rejection manifest. Run after ingestion and before Winner V2 training. | None. |
| `lol validate-winner-model` | Fail-closed validation of the promoted direct-series bundle: actionability, ten-member ensemble, direct rating families, train/serve parity, canonical swap contract, and forbidden-feature absence. | `--format table|json`. |
| `lol market-check` | Public, read-only Gamma/CLOB check. Reports typed match, token orientation, minimum shares, hypothetical cost, executable odds, and isolated token failures. Two observations use one shared 45-second wait. | `--match-key <pandascore-key>` checks that fixture's supported typed market. |
| `lol market-watch` | Write one JSON/Markdown pair and append read-only series-winner price/book observations for timing/CLV research. Intended for an hourly scheduler; never creates a proposal. | `--format table|json`. |
| `lol train` | Routine refit with reviewed parameters; retrains weights and calibrators, registers an immutable candidate, and may auto-promote only a healthy non-inferior non-Optuna V2 bundle. It does not run Optuna. | `--targets all|series_winner|next_map_winner|props|<names>`, `--feature-set full|compact|selected`, `--max-features N`, report-only feature selection, and calibration/split options. |
| `lol retune` | Explicit Optuna research search. Writes isolated tuning reports and artifacts; never updates reviewed parameters or champion automatically. | `--targets`, `--feature-set`, `--max-features`. |
| `lol promote-tuning <run-id>` | Owner promotion of the complete target set requested by one reviewed tuning run. Writes only those production parameter files, not model weights. Run a complete `lol train` afterward. | Complete run ID. |
| `lol review-tuning <run-id>` | Compare a Winner V2 Optuna result with its predeclared rating baseline on the untouched holdout. Writes `tuning_review.json`; blocked reviews cannot be promoted. | Completed Winner V2 tuning run ID. |

Examples:

```bash
uv run oracle-bets lol source-refresh
uv run oracle-bets lol source-check
uv run oracle-bets lol ingest
uv run oracle-bets lol validate-data
uv run oracle-bets lol build-series
uv run oracle-bets lol validate-winner-model
uv run oracle-bets lol schedule --days 2
uv run oracle-bets lol market-check --match-key <pandascore-key>
uv run oracle-bets lol market-watch

# Routine retraining: no Optuna
uv run oracle-bets lol train --targets all --feature-set compact

# Winner V2 research retune; continue only if review status is approved
uv run oracle-bets lol retune --targets series_winner --feature-set full
uv run oracle-bets lol review-tuning <run-id> --format json
uv run oracle-bets lol promote-tuning <run-id>

# Only after series and next-map targets both have reviewed fixed parameters
uv run oracle-bets lol train --targets all --feature-set compact
```

The experimental next-map target does not yet have the independent tuning
review required for parameter promotion. Do not use a combined retune to bypass
that missing gate. The August 14, 2026 series study
`20260813T233204_354446Z` is blocked and must not be promoted.

Source refresh, ingestion, retraining, and retuning are separate operations.
The daily workflow refreshes the public cache before ingestion. Retraining updates
model weights/calibrators on newer data using fixed reviewed hyperparameters.
Retuning searches hyperparameters and carries greater overfitting risk.

The source adapter mirrors Drive's public **Download all** behavior through its
anonymous bulk-export endpoint. That endpoint is undocumented and may change;
the adapter therefore trusts only Google Storage archive URLs, extracts only
the three reviewed filenames, validates ZIP/CSV structure and sizes, and fails
closed without replacing the previous cache.

## Candidate registry

| Command | Effect |
| --- | --- |
| `model status [--registry PATH]` | Read champion and registry health. |
| `model list [--format table|json] [--registry PATH]` | List immutable candidates. |
| `model review <model-id> [--format table|json]` | Show manifest, evidence, gates, and artifacts. |
| `model register-run <run-id|latest>` | Confirm an automatically registered training run. A complete LoL bundle includes direct series, experimental next-map, legacy map diagnostic, three shadow props, their calibrators/uncertainty, and shared rating lookups. |
| `model register-current <id> --code-version <sha> --metric name=value [...]` | Freeze the current complete inference tree as a manual candidate; optional `--target`, `--random-seed`, `--registry`. |
| `model promote <id> --reason <text>` | Explicitly move the champion pointer after owner review. |
| `model quarantine <id> --reason <text>` | Immediately make a model research-only; diagnostic predictions remain available but it cannot create paper proposals. |
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

Useful options are `--horizon-hours`, `--leagues`, `--discord-delivery-mode`,
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
uv run oracle-bets paper requote <proposal-id> --format json
uv run oracle-bets paper decide \
  --proposal-id <id> --decision accept --requote-token <120-second-token> \
  --reason "owner confirmed fresh quote"
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
| `paper requote <proposal-id>` | Capture two fresh executable books, rerun every Winner V2 gate, and issue a confirmation token valid for 120 seconds. Read-only market access; no position is opened. |
| `paper record-map-state --series-id ... --map-number ... --winner ... --source-reference ...` | Append owner-verified completed-map state for the separate reactive shadow experiment. Maps must be sequential and conflicting evidence is rejected. It never creates an actionable position. |
| `paper expire --strategy-version <version|all> --reason <text>` | Append invalidation corrections to matching undecided proposals without deleting evidence. |
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

The Gateway bot provides Accept/Reject and Win/Loss/Push/Void controls. Accept
first fetches a fresh two-observation quote; only a separate Confirm within 120
seconds opens the flat one-unit paper position.
Settlement buttons open a required source-reference modal. Controls persist
across restarts and disable after settlement. The LLM never sees or influences
settlement. Webhook delivery is disabled.
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
uv run ty check packages
uv run mkdocs build --strict
uv run oracle-bets lol health
uv run oracle-bets lol validate-data
uv run oracle-bets lol validate-winner-model
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
Use the Gateway-bot template for interactive controls. Do not install a second
message-delivery job.

The clean research rebuild sequence is in
[Getting started](getting_started.md#first-clean-research-rebuild). It deletes
generated data, models, reports, logs, evidence, and registry state, so back up
`data/state/` first. A clean rebuild is disaster recovery/research setup, not a
normal retrain.
