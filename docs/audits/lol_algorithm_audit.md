# LoL Algorithm Audit

This audit records the current state of `lol-bets` after the Oracle Bets
modularization. It is intentionally decision-oriented: each section states what
is sound, what is risky, and what should change before we judge profitability.

## Current State

The model stack has the right broad shape:

- Oracle's Elixir is a strong free data source and supports repeated refreshes.
- The pipeline separates team and player rows, computes temporal ratings, builds
  `*_before` training features, and writes flattened `*_after` inference state.
- LightGBM is the right first production baseline for tabular esports data.
- Existing headline metrics around 0.59-0.61 accuracy, AUC near 0.70, and Brier
  near 0.218 are plausible for match-winner prediction, but they are not enough
  to prove betting edge.

The immediate operational blocker is artifact completeness. Local team artifacts
exist, but player training and flattened artifacts do not:

- `data/lol/processed/teams/training_teams.parquet`
- `data/lol/processed/teams/flattened_teams.parquet`
- missing `data/lol/processed/players/training_players.parquet`
- missing `data/lol/processed/players/flattened_players.parquet`

Until those player artifacts are regenerated, Discord inference and supervised
training should remain unhealthy.

The evidence points to an artifact-materialization gap rather than missing
configuration: raw data and interim player data exist, and player training/
flattened configs exist, but `data/lol/processed/players/` contains no parquet files.
That usually means the prior pipeline run stopped before processed player output
or those files were removed after generation. The next ingest run should confirm
whether player feature generation itself fails or simply needs to be rerun.

## Ingestion

### Findings

- Full game composition checks are mostly present: the ingestion layer removes
  games that do not have 12 total rows, 10 unique players, and 2 teams.
- League filtering is explicit through the active profile in `considered_leagues.json`.
- Date coercion and game length normalization are handled early.
- Patch values are forward-filled, which is practical for Oracle's Elixir dumps
  but should be validated after every upstream schema change.
- Opponent pairing previously depended on exact sorted row blocks. This was
  fragile because any row-order drift could silently assign the wrong opponent.

### Changes Made

- Team/player opponent enrichment now pairs by `gameid`, `side`, and for players
  `position`, using the opposite side. This removes the most fragile block-order
  dependency from ingestion.

### Next Decisions

- Keep `get_opponent` only as a low-level mirror helper for already-sorted
  two-side feature arrays. Prefer merge-based pairing for raw dataframe work.
- Add a pipeline validation report after ingestion with counts for dropped games,
  missing dates, missing patches, unknown entities, league exclusions, and games
  with non-standard position composition.

## Feature Engineering

### Findings

- The feature design correctly distinguishes training `*_before` features from
  flattened inference `*_after` state.
- Several historical features used grouped cumulative calculations followed by a
  global `shift()`. That can leak the final value from one group into the first
  row of the next group.
- Atakhan fields should be removed for 2026 onward because Atakhan, Blood Roses,
  and Feats of Strength left Summoner's Rift. Historical 2025 Atakhan data is
  now a patch-specific artifact rather than a stable skill signal.
- Oracle's Elixir currently provides 15/25-minute checkpoint columns, not
  14-minute columns. Treat 15 minutes as the practical post-plates mid-game
  marker and 25 minutes as the late-game marker; avoid carrying 10/20-minute
  checkpoints unless an ablation proves they add calibrated out-of-time signal.
- Void Grubs now have one spawn cycle, so `void_grubs` remains usable as a
  capped early-objective count, but old assumptions about two grub waves should
  not be encoded.
- Head-to-head is conceptually useful but sparse. It should be gated by prior
  sample count and probably treated as low-confidence below 2-3 prior games.
- Player aggregates are valuable, but they can inject roster noise if flattened
  rosters lag actual lineups.

### Changes Made

- Fixed grouped shift leakage in:
  - player win/loss expanding metrics,
  - generic expanding means,
  - head-to-head win counts.
- Added tests that prove group boundaries reset correctly.

### Next Experiments

Run these feature sets with the same walk-forward folds:

1. `full`: current full config after leakage and 2026 objective cleanup.
2. `compact`: existing compact config.
3. `rating_only`: Elo, Glicko2, PL, TrueSkill, league Elo, side/patch/season rates.
4. `team_only`: remove player pivot features.
5. `player_only_plus_meta`: player form plus league/side/patch metadata.
6. `no_late_game`: remove 25-minute stats and any high-missing objective fields.
7. `no_h2h`: remove head-to-head features.
8. `no_league_priors`: remove calibrated league-prior features.

Accept a feature family only if it improves out-of-time log loss/Brier, not just
accuracy.

## Ratings And Hyperparameters

Ratings are not just features; they encode strong priors about team strength and
data recency. Their hyperparameters deserve walk-forward tuning.

### Elo

Current risks:

- Fixed starting values may be too neutral for major/minor league imbalance.
- K-factor should not be one-size-fits-all across teams, players, roles, and
  cross-league events.
- Elo divisor controls probability sharpness and directly affects calibration.
- Seasonal decay and position reset are important but easy to overtune.

Tune:

- base team Elo by league tier,
- base player Elo by role and league tier,
- K-factor for teams and players,
- Elo divisor,
- seasonal decay factor,
- promoted/relegated league prior,
- position reset factor.

Objective:

- primary: walk-forward log loss,
- secondary: Brier, calibration slope/intercept, ROI simulation.

### Glicko2

Current risks:

- RD/phi and volatility can encode uncertainty better than Elo, but bad defaults
  can make new or inactive teams too volatile.
- Uncertainty should decay through inactivity and reset across seasons.

Tune:

- initial mu, phi, sigma,
- volatility constraint,
- inactivity RD growth,
- team aggregation method for players.

### TrueSkill

Current risks:

- TrueSkill is useful for lineup aggregation, but default beta/tau/draw settings
  come from generic games and may not match LoL.
- LoL has no draw for game winner, but series markets can have draw-like outcomes
  in BO2 formats.

Tune:

- mu,
- sigma,
- beta,
- tau,
- draw probability only for BO2/series models, not single-game winner.

### Plackett-Luce

Current risks:

- PL is attractive for multi-entity lineup strength but may duplicate TrueSkill
  signal.
- It should survive ablation before keeping it in compact configs.

Tune:

- initial uncertainty,
- update scale,
- player-to-team aggregation,
- role weights.

### EMA Features

Current risks:

- One global half-life is unlikely to be optimal. Patch form, side win rate, team
  form, and player KDA-style stats decay at different speeds.

Tune:

- team-form half-life,
- player-form half-life,
- patch half-life,
- side half-life,
- minimum games before using rate features,
- neutral prior strength.

## Modeling And Evaluation

### Current Model

LightGBM remains the preferred baseline. It handles mixed feature scales,
missingness, nonlinear interactions, and modest data sizes better than a neural
model by default.

### Main Risks

- F1 threshold tuning is useful for binary labels but not the betting objective.
  Betting needs calibrated probabilities.
- Accuracy can improve while expected value gets worse.
- Validation must be chronological; random/grouped splits overstate performance
  when team strength drifts.
- Feature selection reports for importance/cumulative modes do not currently
  retrain on the selected subset, so they are diagnostics rather than true
  feature-reduced experiments.

### Required Validation

Every accepted model run should report:

- accuracy,
- log loss,
- Brier,
- ROC AUC,
- calibration slope/intercept,
- expected calibration error,
- league cohort metrics,
- patch cohort metrics,
- side cohort metrics,
- market edge simulation.

### Calibration Plan

1. Train LightGBM on the training window.
2. Fit calibration on validation only:
   - isotonic when validation sample size is large enough,
   - sigmoid/Platt when validation is smaller.
3. Evaluate calibrated probabilities on the test window.
4. Store uncalibrated and calibrated metrics separately.
5. Use calibrated probability for Discord edge/fair-odds output.

### Market Simulation

The current market layer should remain read-only. For historical evaluation, add
a fixture-based simulator that accepts model probability and market implied
probability, then reports:

- flat stake ROI,
- half-Kelly ROI,
- max drawdown,
- bet count by edge threshold,
- performance by league and patch,
- sensitivity to slippage/liquidity assumptions.

No wallet keys, signing, order placement, or automated execution belongs in this
phase.

## Recommended Roadmap

1. Finish artifact and leakage correctness.
2. Regenerate full team/player artifacts.
3. Reproduce the existing LightGBM baseline.
4. Add calibrated LightGBM evaluation.
5. Run feature ablations.
6. Tune rating hyperparameters with walk-forward validation.
7. Only then evaluate TabNet/neural models.
8. Add props only where labels are reliable and artifacts exist: game length,
   total kills, towers/objectives.

## Acceptance Criteria

Do not promote a new algorithm unless it beats the baseline out-of-time on:

- lower log loss,
- lower Brier,
- acceptable calibration,
- stable cohort behavior,
- positive simulated edge after conservative friction assumptions.
