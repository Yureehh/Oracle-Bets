# Pipeline status

_Local verification snapshot: 2026-10-09. Generated datasets, model bundles,
quotes, and the evidence database are local state; cloning GitHub does not
include them._

The numbers below are milestones in the workflow, not a model-quality score or
an estimate of profitability. Research and paper collection can proceed in
parallel, but a later milestone cannot compensate for a failed earlier gate.

| Milestone | What happens | Current evidence | Next gate |
| --- | --- | --- | --- |
| 0 — Source and configuration | Install locked dependencies; keep provider credentials outside Git. | The serving-parity and pytest security fixes are merged, with 730 tests passing and 1 skipped locally; CI, lint, formatting, and type checks passed for those changes. | Check setup on a second machine. |
| 10 — Ingest | Refresh the public Oracle's Elixir source and retain a versioned raw snapshot. | The validated source is current through 2026-10-07 11:58 UTC. | Continue checking source freshness before each rebuild. |
| 20 — Clean data and identities | Quarantine bad rows; normalize teams, players, maps, and series. | The October 8 rebuild has 24,430 games, 10,196 accepted series, and 1,770 quarantined groups. | Review quarantined groups when source or rules change. |
| 30 — Features and targets | Build chronological team/player features, ratings, and paired serving history. | Map generation `d0335193`, series generation `e088b017`, and serving snapshot `features-e9cfedc2dc0f9424b5f20252` passed source, code, and table lineage checks. | Preserve paired lineage on every refresh. |
| 40 — Research splits | Fit preprocessing inside temporal development folds; reserve sealed evaluation and record prior exposure. | Leakage and lineage guards are implemented and covered by tests; the October 8 studies used sealed temporal splits and recorded prior exposure. | Keep the split and exposure checks in each new generation. |
| 50 — Fresh search | Run a separate Optuna study for each supported target. | Five independent 100-trial studies completed on clean commit `3e64767`, one each for map winner, series winner, game length, kills, and towers. All use the October 8 source and map generation; the series study also records the matching series generation. | Keep the study parameters research-only until review. |
| 60 — Full refit | Apply the selected study parameters to research-only models. | Refit `20261008T170701_385063Z` completed all five targets with no failures. Its manifest is clean, research-only, and explicitly non-promotable. | Review calibration, baseline comparisons, and serving replay before considering any separate candidate promotion. |
| 70 — Replay and model review | Compare sealed feature values with the real serving path; inspect calibration and cohort gates. | A registered temporary candidate passed bundle integrity and data-lineage checks, but the 16-row-per-target serving replay **failed**. Several historical rosters lack earlier serving statistics; one non-actionable game crossing midnight has feature differences. The live champion remains the August bundle. | Resolve replay availability and cross-midnight parity, then repeat the full replay on a lineage-valid candidate. Do not promote this research refit. |
| 80 — Prospective fixture population | Enroll scheduled fixtures before forecasts and prices. | The October 8 daily report enrolled two LCS fixtures before their scheduled starts. Earlier fixture cohorts remain recorded. | Continue complete enrollment and record missing markets and no-bet decisions. |
| 90 — Market and paper cycle | Capture exact Polymarket/Thunderpick contracts, owner-confirm quotes, size positive-edge paper entries, record close and result. | The evidence database passes integrity checks, but contains zero bets and zero settlements. The latest local market observations are from October 4; the October 8 Polymarket check found no supported open LoL market from Italy. No verified Thunderpick line or complete quote-to-settlement cycle exists. | Obtain an owner-observed executable contract and price, then record a real prospective paper cycle, including the closing quote and sourced result. |
| 100 — Decision evidence | Evaluate all enrolled fixtures, net paper returns, calibration, closing-line value, exposure, and drawdown before extending to tennis or real stakes. | **Pending.** No observed earnings claim is justified. | Accumulate enough prospective, settled, representative samples and compare locked policies on later data. |

The live paper workflow does not place bets. The owner chooses and confirms any
entry. A model can load successfully while its market strategy remains
exploration-only. Current map, totals, handicaps, and props must retain that
label until target-specific sealed and paper evidence supports stronger use.

## How to operate and learn from it

1. Follow the [command runbook](commands.md) to check source, data, model,
   evidence, and Discord health. Back up the evidence database before a reset.
2. Inspect the daily schedule and prospectively enrolled cohort. For one
   fixture, supply exact provider URLs and manually transcribe visible
   Thunderpick odds, line, map/series period, selection, and settlement terms.
3. Read the full comparison, including missing and unsupported markets. A
   positive model probability difference is only a hypothesis until the quote
   is executable and the contract matches exactly.
4. If a paper proposal survives the checks, confirm its current price and
   capped stake. Later record the closing observation, official result, and
   settlement source. Leave missing stages marked missing.
5. Review calibration, Brier/log loss, coverage, net return, drawdown, and
   closing-line value together. Compare the preregistered eighth, quarter, and
   half-Kelly policies on development data, then evaluate the chosen policy on
   later observations. Do not tune sizing against the same outcomes used to
   claim its performance.

Historical prediction accuracy is not betting profit. Odds, fees, failed
captures, correlated positions, changed lines, and losses all affect realized
returns. This project currently has no verified live edge or income stream.

## Current research results

The October 8 sealed evaluations returned map-winner accuracy 65.5%, AUC
0.714, and Brier 0.215 on 3,663 games; series-winner accuracy 71.3%, AUC
0.786, and Brier 0.188 on 1,528 series. Game length, kills, and towers had
mean absolute errors of 4.245 minutes, 7.532 kills, and 1.711 towers. Each
beat its constant baseline, but towers improved by only 0.019 towers and had
R² 0.005. None of the prop reports contains a historical market-line backtest.
These test windows overlap prior research, so they are development evidence,
not a fresh claim of market edge or earnings.

## Performance work after correctness

The October 8 full rebuild took 14,819.6 seconds wall time, including
13,768.3 seconds in enrichment; the host may have slept, so profile an awake
run before using those timings as a benchmark. An earlier daily run with no
changed history still spent about 30 minutes rebuilding features. The next
optimization is a no-change shortcut
guarded by source identity, code/configuration fingerprint, manifest integrity,
and output hashes. Benchmark an awake before/after run and keep the full path
for changed or damaged inputs. Serving also verifies and reads the paired
snapshot for both teams; measure fixture latency before caching a verified
read-only pair. Profile memory and CPU before changing the dataframe backend or
adding infrastructure. FireDucks remains opt-in and is not in the locked
installation while its dependency pins an Arrow version flagged by the
repository's dependency review.

See the [research readiness plan](plans/2026-09-24-1117-fix-paper-research-readiness-plan.md)
for detailed gates and the separate
[tennis plan](plans/2026-09-06-0050-feat-tennis-betting-module-plan.md) for
the later expansion.
