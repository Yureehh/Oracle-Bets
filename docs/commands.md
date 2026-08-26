# Command and Discord runbook

This is the canonical usage guide. Run commands from the repository root:

```bash
cd /Users/yureeh/dev/oracle_bets
source .venv/bin/activate
```

Oracle Bets predicts independently of market prices. It only reads Polymarket,
never contacts Thunderpick automatically, and never places a bet.

## Normal order of use

### Start the owner console

```bash
uv run oracle-bets evidence init
uv run oracle-bets discord doctor
uv run oracle-bets discord doctor --live
uv run oracle-bets discord run
```

Only one Gateway process may run. If launchd already owns it, stop it before a
manual run:

```bash
launchctl bootout gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.discord-bot.plist
```

Restart the managed service after code/configuration changes:

```bash
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.oracle-bets.discord-bot.plist
```

### Review markets and record a bet

1. In Discord, type `/oracle`.
2. Use **Schedule** to inspect the next 14 days.
3. Use **Review Markets** and paste one or two Polymarket/Thunderpick links.
4. For Thunderpick, select the fixture and manually enter visible lines as
   `target | selection | decimal odds | line | game number`.
5. Read the compact comparison and attached Markdown report.
6. Use **Record Bet**, choose paper/real and currency, and enter the actual
   accepted odds, bankroll, stake percentage, amount, and note.
7. Review the confirmation screen. Confirm only after manually placing a real
   bet outside Oracle Bets.
8. Use **Open Bets** to settle Win/Loss/Push/Void with a result reference.
9. Use **Closed Bets** and **Performance** to review evidence.

The hub layout is:

- Row 1: **Review Markets**, **Record Bet**, **Open Bets**, **Closed Bets**.
- Row 2: **Schedule**, **Performance**, **Health**.

The same core workflow is available from the CLI:

```bash
uv run oracle-bets lol schedule --days 14
uv run oracle-bets lol market-review "<POLYMARKET_URL> <THUNDERPICK_URL>"
uv run oracle-bets bet list --state open --mode paper
uv run oracle-bets bet show <BET_ID>
uv run oracle-bets bet settle \
  --bet-id <BET_ID> --result win --source-reference <RESULT_URL_OR_ID>
uv run oracle-bets bet performance --mode paper
```

## Thunderpick CLI input

Thunderpick is link-plus-manual-lines only. Create a JSON file:

```json
[
  {
    "target": "total_kills_mean",
    "selection": "Over",
    "decimal_odds": 1.9,
    "line": 27.5,
    "game_number": 1,
    "note": "visible Thunderpick line"
  }
]
```

Then run:

```bash
uv run oracle-bets lol market-review \
  "<THUNDERPICK_URL>" --manual-lines /path/to/lines.json
```

Accepted targets are `series_winner`, `map_winner`, `series_total_maps`,
`series_handicap`, `gamelength_mean`, `total_kills_mean`, and
`total_towers_mean`. Unsupported targets remain recordable evidence but have no
model probability.

## Unified bet ledger

The active commands are `bet list`, `bet show`, `bet record`, `bet settle`, and
`bet performance`.

```bash
uv run oracle-bets bet list --state open --mode paper
uv run oracle-bets bet list --state settled --mode real --format json
uv run oracle-bets bet show <BET_ID> --format json

uv run oracle-bets bet record \
  --review-id <REVIEW_ID> \
  --market-id <MARKET_ID> \
  --mode paper \
  --currency EUR \
  --bankroll-before 1000 \
  --stake-percent 1 \
  --accepted-odds 1.95 \
  --reason "owner review"

uv run oracle-bets bet settle \
  --bet-id <BET_ID> \
  --result win \
  --source-reference <RESULT_URL_OR_ID>

uv run oracle-bets bet performance --mode paper --format table
uv run oracle-bets bet performance --mode real --format json
```

Stake amount must equal bankroll × stake percentage within currency rounding.
Win PnL is `stake × (odds - 1)`, loss is `-stake`, and push/void are zero.
Paper and real results are separate; different currencies are never summed.
Settlement is always manual and append-only.
`bet record --idempotency-key <KEY>` is only for retrying the same failed
record request; omit it for a genuinely separate bet, even when the terms are
identical. `--opened-at <ISO_TIMESTAMP>` records the actual entry time.

## Data and model lifecycle

These are CLI-only operations:

| Command | When to use it |
| --- | --- |
| `lol source-check` | Verify local Oracle's Elixir 2024–2026 files and freshness. |
| `lol source-refresh` | Refresh the public source cache. Network/write. |
| `lol ingest` | Refresh and incrementally rebuild history/features/ratings. No Optuna. |
| `lol reconcile-history` | Full retained-history rebuild. No training. |
| `lol sync-identities` | Synchronize historical identity links into evidence. |
| `lol validate-data` | Validate generated data before training. |
| `lol build-series` | Reconstruct BO1/3/5 direct-series rows. |
| `lol train` | Retrain with reviewed fixed parameters. Never runs Optuna. |
| `lol retune` | Explicit isolated Optuna research. Never auto-promotes. |
| `lol review-tuning` | Review one tuning run on sealed evidence. |
| `lol promote-tuning` | Promote reviewed parameter files, not model weights. |
| `lol health` | Check current LoL data/model artifacts. |
| `lol validate-winner-model` | Check symmetry, lineage, leakage, calibration, and serving parity. |
| `lol validate-market-strategies` | Validate map-path totals/handicaps and experimental map evidence. |
| `lol market-check` | Read-only public Polymarket connectivity/book diagnostic. |
| `lol market-review` | Review one fixture from one/two exact owner links; writes one report pair and evidence. |
| `lol schedule` | Fetch/save PandaScore fixtures; normally use `--days 14`. |

Routine retraining updates weights and calibrators with fixed reviewed
hyperparameters. Retuning runs Optuna and needs separate review. Neither action
is required for each market review.

## Model registry

| Command | Purpose |
| --- | --- |
| `model status` | Show current champion/registry state. |
| `model list` | List immutable candidates. |
| `model review` | Review a candidate; `--refresh` replays evidence. |
| `model register-run` | Confirm an automatically registered training run. |
| `model register-current` | Manually freeze a complete artifact bundle. |
| `model promote` | Move the champion pointer with an owner reason. |
| `model quarantine` | Make an unsafe model research-only. |
| `model rollback` | Restore a prior healthy bundle. |

## Evidence, health, and audit

| Command | Purpose |
| --- | --- |
| `evidence init` | Create or transactionally migrate the append-only database. |
| `evidence health` | Verify schema and SQLite integrity. |
| `evidence backup` | Create a consistent database backup. |
| `evidence export` | Export JSON or CSV review copies. |
| `evidence restore-verify` | Verify a backup without replacing live state. |
| `health system` | Write/read combined system health. |
| `audit monthly` | Create a monthly operational/evidence review. |

## Daily and Discord

`daily lol` handles data refresh, validation, schedule, optional fixed-parameter
retraining, and one daily report pair. It does not discover markets, create
proposals, record bets, or settle bets.

```bash
uv run oracle-bets daily lol --dry-run
uv run oracle-bets daily lol --skip-retrain
```

Public commands are `discord doctor` and `discord run`. The first is diagnostic;
`--live` adds read-only Discord API checks. The second starts the owner-only
Gateway console.

## Verification before paper testing

```bash
uv run pytest tests/core tests/lol
uv run ruff check packages tests
uv run ty check packages
uv run mkdocs build --strict
uv run oracle-bets lol validate-data
uv run oracle-bets lol validate-winner-model
uv run oracle-bets lol validate-market-strategies
uv run oracle-bets lol health
uv run oracle-bets evidence health
uv run oracle-bets discord doctor
uv run oracle-bets discord doctor --live
```

## Generated state

- `data/state/oracle_bets.db`: canonical ledger; keep and back up.
- `data/state/model-registry/lol/`: immutable candidates/champion history.
- `models/lol/`: replaceable training workspace; `.staging` is crash-safe and
  should be empty when no training runs.
- `reports/lol/`: generated audit reports.
- `logs/lol/`: bounded rotating pipeline/schedule/Discord logs.
- `site/`, `.hypothesis/`, `.pytest_cache/`, `.ruff_cache/`: disposable output.
- `ops/launchd/`: optional macOS Gateway service templates.
- `notebooks/lol/`: cleared, read-only analysis notebooks.
