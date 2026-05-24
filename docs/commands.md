# Command Reference

Run commands from the repository root:

```bash
cd /Users/yureeh/Dev/Oracle-Bets
```

Use `uv run ...` unless your shell already has the editable package installed.

## LoL Data Pipeline

### Health

```bash
uv run oracle-bets lol health
```

Checks required LoL artifacts. It returns non-zero while required trained models or generated artifacts are missing.

### Ingest

```bash
uv run oracle-bets lol ingest
```

Builds processed LoL team/player data, ratings, flattened inference tables, and training tables.

Parameters: none.

## LoL Training

Base command:

```bash
uv run oracle-bets lol train [options]
```

Options:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--model-type` | `lightgbm`, `tabnet` | `lightgbm` | Model family to train. Use `lightgbm` for current production work. |
| `--targets` | `all`, `outcome`, `props`, `gamelength`, `total_kills`, `total_towers`, comma-separated targets | `all` | Which models to train. `props` means game length, total kills, and total towers. |
| `--feature-selection` | `none`, `importance`, `cumulative`, `rfecv`, `boruta`, `report` | `none` | Whether to produce or apply feature-selection logic. Use `report` to generate recommendations without auto-overwriting compact configs. |
| `--feature-set` | `full`, `compact`, `selected` | `full` | Feature surface used for training. `compact` uses curated compact configs. `selected` uses feature-selection reports. |
| `--max-features` | integer | `120` | Feature cap for `--feature-set selected`. |
| `--force-retune` | flag | off | Ignore cached model hyperparameters and run Optuna again. |

Common commands:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets outcome
uv run oracle-bets lol train --model-type lightgbm --targets props
uv run oracle-bets lol train --model-type lightgbm --targets total_kills,total_towers
uv run oracle-bets lol train --model-type lightgbm --targets all --feature-selection report
uv run oracle-bets lol train --model-type lightgbm --targets outcome --feature-set selected --max-features 120
uv run oracle-bets lol train --model-type lightgbm --targets props --feature-set selected --max-features 120
uv run oracle-bets lol train --model-type lightgbm --targets all --force-retune
```

Daily/default retrain after ingestion should usually omit `--force-retune` so cached hyperparameters are reused.

## Discord Bot

```bash
uv run oracle-bets discord run
```

Requires `DISCORD_TOKEN` and trained artifacts.

The bot can start with outcome-only artifacts. In that state `!lol predict`,
`!schedule`, `!markets`, and `!bet` work, while `!lol props` requires trained
`gamelength`, `total_kills`, and `total_towers` artifacts.

### Prediction And Market Commands

```text
!lol
!lol predict "Team WE" "LNG Esports"
!lol predict "Team WE" "LNG Esports" --side Blue --first-pick "Team WE"
!lol predict "Team WE" "LNG Esports" --bo3
!lol predict "Team WE" "LNG Esports" --bo5
!lol props "Team WE" "LNG Esports"
!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85 --kills-under-odds 1.95
!lol props "Team WE" "LNG Esports" --towers-line 12.5 --towers-over-odds 1.90
!lol props "Team WE" "LNG Esports" --length-line 31.5
!lol edge "Team WE" "LNG Esports"
!markets Team WE LNG kills
!edge 2.10 55%
!kelly 2.10 55%
!odds 55%
!prob 2.10
```

Polymarket yes/no prices are probabilities. A `44c` price means about `44%`
implied probability and fair decimal odds of `1 / 0.44 = 2.27`. A `57c` price
means about `57%` implied probability and fair decimal odds of `1 / 0.57 = 1.75`.

For a Polymarket winner market:

```text
!lol predict "Team Vitality" "Movistar KOI"
!edge 2.27 48%
!bet record --event "Team Vitality vs Movistar KOI" --market winner --selection "Team Vitality" --odds 2.27 --stake 25 --book polymarket --prob 0.48 --league LEC
```

Replace `48%` with the model probability shown by `!lol predict`. Use the
Polymarket price converted to decimal odds for `!edge` and `!bet record`.

### Handicap Markets (+1.5 / -1.5)

`!lol predict` automatically outputs a **Handicap Markets** section alongside
the scorelines. No extra command is needed.

**What the lines mean**

| Line | Covers |
|------|--------|
| `Team +1.5` (BO3) | Team wins OR loses 1–2 (not swept 0–2) |
| `Team -1.5` (BO3) | Team must sweep 2–0 |
| `Team +1.5` (BO5) | Team wins series OR loses 2–3 |
| `Team -1.5` (BO5) | Team wins 3–0 or 3–1 |
| `Team +2.5` (BO5) | Team not swept 0–3 |
| `Team -2.5` (BO5) | Team wins 3–0 |

**Example at 50/50 BO3** — model gives both teams 50% single-game win rate:

```
• Vitality +1.5 (not swept): 75.00% → fair odds 1.33
• KOI +1.5 (not swept):      75.00% → fair odds 1.33
• Vitality -1.5 (must sweep): 25.00% → fair odds 4.00
• KOI -1.5 (must sweep):      25.00% → fair odds 4.00
```

If Polymarket offers `Vitality +1.5` at `1.58`:

```text
!edge 1.58 75%
# edge +18.5%

!kelly 1.58 75%
# half-Kelly sizing (halve again for quarter-Kelly)

!bet record --event "Team Vitality vs Movistar KOI" --market handicap --selection "Vitality +1.5" --odds 1.58 --stake 40 --kelly 0.08 --book polymarket --prob 0.75 --edge 0.185 --league LEC
```

### Bet Ledger Commands

These commands record bets you manually placed. They do not place bets, send orders, or touch a wallet.

```text
!bet
!bet record --event "Team WE vs LNG Esports" --market kills --selection over --line 26.5 --odds 1.85 --stake 25 --book polymarket --prob 0.557 --edge 0.031 --map 1 --league LPL
!bet record --event "Team WE vs LNG Esports" --market towers --selection under --line 12.5 --odds 1.95 --stake 10 --book thunderpick --map 1 --league LPL
!bet record --event "Team WE vs LNG Esports" --market winner --selection "LNG Esports" --odds 1.76 --stake 50 --book polymarket --prob 0.568 --edge 0.000
!bet record --event "Team Vitality vs Movistar KOI" --market handicap --selection "Vitality +1.5" --odds 1.58 --stake 40 --kelly 0.08 --book polymarket --prob 0.75 --edge 0.185 --league LEC
!bet list --status open
!bet list --status settled --limit 20
!bet settle 12 --result win
!bet settle 12 --result loss
!bet settle 12 --result push
!bet delete 12
!bet bankroll --balance 1250 --note "after LPL slate"
```

`!bet record` options:

| Option | Required | Meaning |
|---|---:|---|
| `--event` | yes | Match/event name stored in the ledger. |
| `--market` | yes | Market family, e.g. `winner`, `kills`, `towers`, `length`. |
| `--selection` | yes | Your bet selection, e.g. `over`, `under`, `"Team WE"`. |
| `--odds` | yes | Decimal odds at which you placed the bet. |
| `--stake` | no | Stake amount. If omitted, the bet is tracked without stake/PnL sizing. |
| `--book` | no | Bookmaker or venue, e.g. `polymarket`, `thunderpick`. |
| `--prob` | no | Model probability for the selected outcome, as `0.557` for 55.7%. |
| `--edge` | no | Expected edge as decimal, e.g. `0.031` for +3.1%. |
| `--kelly` | no | Half-Kelly fraction as decimal, e.g. `0.012` for 1.2% bankroll. |
| `--line` | no | O/U line such as `26.5`. |
| `--map` | no | Map number for map-specific props. |
| `--bo` | no | Series format: `1`, `2`, `3`, or `5`. |
| `--league` | no | League/tournament label. |
| `--sport` | no | Defaults to `League of Legends`. |
| `--side` | no | Side/context label: `Blue`, `Red`, `Home`, `Away`, `Over`, `Under`. |
| `--ref` | no | Bet slip/reference id. Used to prevent duplicates. |
| `--smoke` | no | Paper bet flag. |
| `--live` | no | Live/in-play bet flag. |
| `--note` / `--notes` | no | Free-text note. |
| `--tags` | no | Comma-separated tags. |

`!lol props` emits copy-paste `!bet record ...` lines for priced positive-edge props when market odds are supplied. Copy one only after you actually place the bet yourself.

## Dashboard

```bash
uv run oracle-bets dashboard init
uv run oracle-bets dashboard run
```

The dashboard is a local bet ledger and analytics UI. It is separate from model training.

Ledger DB location:

```text
data/ledger.db
```

Dashboard pages:

| Page | Purpose |
|---|---|
| Overview | Real/smoke KPI summary, PnL curve, recent bets. |
| Record Bet | Manual single/parlay bet entry. |
| Ledger | Filter, inspect, settle, and delete bets. |
| Analytics | Performance by sport, market, bookmaker, timing, and edge bucket. |
| Bankroll | Balance history, deposits, withdrawals, and risk metrics. |

## Daily Workflow

```bash
uv run oracle-bets lol ingest
uv run oracle-bets lol train --model-type lightgbm --targets all --feature-set selected --max-features 120
uv run oracle-bets discord run
uv run oracle-bets dashboard run
```

Inside Discord:

```text
!schedule LPL
!lol predict "Team WE" "LNG Esports" --bo5
!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85 --kills-under-odds 1.95
!markets Team WE LNG kills
!bet record --event "Team WE vs LNG Esports" --market kills --selection over --line 26.5 --odds 1.85 --stake 25 --book polymarket --prob 0.557 --edge 0.031 --map 1 --league LPL
!bet settle 12 --result win
```

Do not use `--force-retune` for daily retraining unless model logic, target shape, or data volume changed enough to justify a fresh Optuna run.

## Supported Markets

Supported v1 model-backed markets:

| Market | Command | Notes |
|---|---|---|
| Map winner / series winner | `!lol predict` | Outcome model with calibrated probability when available. |
| Total kills | `!lol props --kills-line` | Game-level total, one row per map in training. |
| Total towers | `!lol props --towers-line` | Game-level total, one row per map in training. |
| Game length | `!lol props --length-line` | Game-level duration in minutes. |

Not yet model-backed:

| Market | Status |
|---|---|
| Team-specific kills/towers | Needs separate side/team-POV targets such as `team_kills`. |
| Player kills | Needs player-level target rows and role/player-specific features. |
| Dragons, inhibitors, first blood, baron, penta/quadra | Intentionally excluded from v1 because they are more volatile and need separate modeling. |

## Betting Terms

| Term | Meaning |
|---|---|
| Model probability | Oracle Bets estimate for the selected outcome. |
| Implied probability | Market probability from decimal odds: `1 / odds`. |
| Fair odds | No-vig decimal odds implied by model probability: `1 / probability`. |
| Edge | Expected value versus market price: `model_probability * market_odds - 1`. |
| Half-Kelly | Conservative bankroll fraction from Kelly sizing, multiplied by 0.5. |
| ROI | Settled PnL divided by staked amount. |
| PnL | Profit/loss after settlement. A loss is `-stake`; a win is `stake * (odds - 1)`. |

Positive edge is not a guaranteed bet. Liquidity, limits, model confidence, stale rosters, and market movement still matter.

## Prop Evaluation

Current prop models should be judged by:

- MAE/RMSE for the central prediction.
- Residual sigma and residual percentiles.
- Over/under calibration by probability bucket.
- Hit rate and ROI by edge bucket when historical market lines/odds are available.
- Per-league metrics, especially LPL when sample size is sufficient.
- Residual cohorts by league, patch, game number, and line band.

For betting, RMSE alone is not enough. Once historical lines are available, line backtesting and closing-line value should decide whether a model is deployable.

## Docs

```bash
uv run mkdocs serve
uv run mkdocs build
```

Serve opens local docs at `http://127.0.0.1:8000/`.

## Environment Overrides

| Variable | Purpose |
|---|---|
| `ORACLE_BETS_HOME` | Override suite root. |
| `ORACLE_BETS_LOL_HOME` | Put all LoL-owned directories under one alternate root. |
| `ORACLE_BETS_DATA_DIR` | Override LoL data directory. |
| `ORACLE_BETS_MODELS_DIR` | Override LoL model directory. |
| `ORACLE_BETS_REPORTS_DIR` | Override LoL reports directory. |
| `ORACLE_BETS_LOGS_DIR` | Override LoL logs directory. |
