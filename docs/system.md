# System and long-term operation

Oracle Bets turns historical LoL results and future fixtures into independent
forecasts and auditable paper research. Market prices never enter the models.
The owner confirms any later entry and sources its close/result. There is no
wallet, signing, order submission, fund movement, or automatic wagering path.
Start with [daily operation](daily-operation.md); use [commands](commands.md)
for the complete CLI reference and [status](pipeline-status.md) for current metrics.

## Pipeline from input to evaluation

| Stage | What it does | Where to read |
| --- | --- | --- |
| Source | Validates public Oracle’s Elixir exports; publishes versioned cache/history and quarantine. | `lol_bets/data_generation/ingestion/` |
| Features | Computes chronological team/player strength, form, inactivity, roster and opponent state before results update it. | `lol_bets/data_generation/feature_engineering/` |
| Targets | Builds map inputs and inferred historical BO1/BO3/BO5 series with explicit rejected groups. | `lol_bets/data_generation/series.py` |
| Research | Fits preprocessing inside temporal folds; runs separate Optuna studies and sealed diagnostics. | `lol_bets/training.py`, `prediction_models/` |
| Registry | Freezes artifacts and hashes, reviews candidates, retains promotion/quarantine history. | `lol_bets/operations/models.py` |
| Serving | Pins one immutable model and paired feature generation; uses state available before the decision. | `lol_bets/inference/` |
| Daily | Refreshes, enrolls the 25-competition/14-day population, forecasts, discovers listings and reports gaps. | `lol_bets/daily.py` |
| Market review | Validates exact event/period/selection/line/rules and executable quote evidence. | `lol_bets/operations/manual_market.py`, `market_actions.py` |
| Capital/ledger | Checks declared native equity and capped Kelly; appends owner decisions, closes, results and corrections. | `oracle_bets_core/operations/`, `evidence/` |
| Discord | Presents owner-only paginated controls and reports; it does not invent decisions. | `oracle_bets_discord/ui/` |

## Packages and configuration ownership

`oracle_bets_core` owns CLI, configuration, evidence storage, pricing primitives,
capital, portfolio research, backup and health. `lol_bets` owns sport-specific
identity, data, features, models and market interpretation. `oracle_bets_discord`
owns the Gateway and interaction controls. These boundaries have real callers;
reducing them to one module would not make the system simpler to operate.

Product configuration schema 5 sets the 336-hour horizon and `paper_selected`
profile. `config/lol/data_ingestion/` owns league profiles/taxonomy and team
aliases; training still uses `research_all_supported`. Feature materialization
and reviewed parameter JSONs have live consumers and are retained. The paper
protocol freezes the experiment; the capital declaration freezes allocation,
conversion and costs before entries. No old parameter file is assumed to be
active merely because it remains in Git.

## What the model predicts

The active bundle `lol-20261010-paper-v1` contains the fresh October refit’s
five target artifacts. Series winner is a direct symmetric, calibrated model.
Map winner is prematch; it is not draft-aware or an in-play next-map signal.
Duration, kills and towers are regression targets whose calibrated distributions
can price supported lines. A mean alone is not an Over/Under probability.

Series totals/handicaps enumerate legal map paths and are explicitly derived.
Paths stop at the required wins. Total-map conservative bounds inspect the
interior 50/50 extremum as well as interval endpoints. Derived assumptions and
disagreement with the direct series model remain visible.

Prop pricing loads the calibrator belonging to the forecast’s immutable model
ID. It cannot reuse an earlier champion’s calibrator after promotion. Missing
identity fails closed. Unsupported player props, first blood, Baron/dragon,
multikills and other specials have no implied model support.

The August recipe’s look-ahead defect invalidates it as a clean benchmark;
conventional overfitting and the size of score inflation were not proved.
The replacement passed first-model noninferiority against its causal rating
baseline, but calibration and conservative-bound coverage warnings remain.
Every strategy is exploratory or display-only; recommendations are disabled.

## Temporal and structural safeguards

Training transformations fit only eligible development rows. Calibration,
uncertainty and evaluation partitions are separated by timestamp. Prior holdout
exposure is recorded; a reused historical test cannot become untouched evidence.
Winner inputs exclude prices, draft and post-start observations. Team swapping
complements winner probabilities and preserves invariant context.

Serving snapshots retain after-game histories, completion lower bounds and
actual publication time. Later reconstruction is not prospective availability.
The paired generation pins both teams. Source/config/code fingerprints and
output hashes bind the build. Reuse retains the original publication timestamp;
it does not backdate or freshly relabel the snapshot.

Historical series format is inferred from observed scores where authoritative
format/completion metadata is absent. An unfinished BO5 source tail at 2–0 can
be indistinguishable from a completed BO3. This remains a research limitation:
obtain trusted format/completion evidence before future retraining or claiming
those inferred labels are authoritative. Frozen weights are preserved here.

## Discovery, executable prices and portfolio sizing

Public listing discovery returns candidates, not approved exact contracts or
executable quotes. Public Polymarket metadata and fresh books serve different
purposes. Thunderpick lines are entered manually with source, period, selection,
line, rules and timestamp. Naive timestamps and malformed inputs fail visibly.
Unmatched/unsupported contracts and provider failures remain in coverage.

An entry later requires a fresh stake-sized quote including costs. Polymarket
checks book depth; the owner attests net costs and available terms. Quotes expire
under the existing 120-second rule. No historical price or minimum-order display
can stand in for the confirmed stake’s executable price.

Quarter Kelly is provisional. Ticket, fixture and total exposure caps reserve
unresolved positions and prevent counting correlated markets as independent
capital. Each venue has a separate declared native bankroll derived from initial
capital plus corrected settled PnL; inputs cannot inflate it. Native currencies
are not summed. A combined EUR report needs explicit valuation/conversion evidence.

Portfolio research compares capped eighth, quarter and half Kelly against the
locked balanced target. It uses fixture-aware temporal splits and settlement
order; joint optimization requires supported joint scenarios, otherwise explicit
caps apply. The existing evaluator’s 50-fixture/8-block support defaults are
research minimums, not evidence that any policy is optimal. No policy wins by
being selected and evaluated on the same outcomes.

## Evidence and maintenance

SQLite is append-only. Corrections supersede facts and must not form cycles or
move a settlement to another bet. Results/settlements cannot precede entry or be
future-dated. Equal-time settlements are netted before realized drawdown updates.
Partial fixture settlement leaves unresolved ticket components open.

Keep `data/state/oracle_bets.db` and immutable model-registry bundles. Backups use
consistent SQLite snapshots and restore verification. Generated data/models,
reports and logs are ignored by Git; logs rotate. `site/`, pytest/Ruff/Hypothesis
caches are disposable, while research receipts are useful evidence. Use guarded
reset commands only when a reset is actually intended—not as routine cleanup.

The local data tree is large because it retains reproducible generations and
paired historical serving states. Do not delete old evidence just to make GitHub
look smaller: it is already outside the tracked repository.

## Longer-term decisions

Keep October 11–November 30 weights, calibration and uncertainty fixed. Daily
feature state may advance with completed games available before decisions.
Evaluate probability quality, coverage, CLV, return, exposure and drawdown by
league and target after the window. Insufficient support is inconclusive.
Calibration/coverage improvements belong on development data and a separately
registered candidate/window, not repeated selection on this exposed holdout.

Before moving to tennis, demonstrate the shared quote-to-close-to-settlement
workflow and review the available evidence; profitable returns are not a
prerequisite for that engineering assessment. The [tennis proposal](plans/2026-09-06-0050-feat-tennis-betting-module-plan.md)
is future design, not an implemented module. Hosting and automated scheduling
can come later; they do not imply automatic wager approval.
