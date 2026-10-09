# Pipeline status

_Local verification snapshot: 2026-10-09. Generated datasets, model bundles,
quotes, and the evidence database are local state; cloning GitHub does not
include them._

The numbers below are milestones in the workflow, not a model-quality score or
an estimate of profitability. Research and paper collection can proceed in
parallel, but a later milestone cannot compensate for a failed earlier gate.

| Milestone | What happens | Current evidence | Next gate |
| --- | --- | --- | --- |
| 0 — Source and configuration | Install locked dependencies; keep provider credentials outside Git. | The serving and rating corrections pass 760 tests, with 1 skipped locally; lint, formatting, and types pass. | Require green final PR checks and check setup on a second machine. |
| 10 — Ingest | Refresh the public Oracle's Elixir source and retain a versioned raw snapshot. | The validated source is current through 2026-10-07 11:58 UTC. | Continue checking source freshness before each rebuild. |
| 20 — Clean data and identities | Quarantine bad rows; normalize teams, players, maps, and series. | The October 8 rebuild has 24,430 games, 10,196 accepted series, and 1,770 quarantined groups. | Review quarantined groups when source or rules change. |
| 30 — Features and targets | Build chronological team/player features, ratings, and paired serving history. | The October 9 rating-unit rebuild changed four model-input tables. Review then found that entity ratings imported end-of-history league Elo, leaking later results into earlier inputs. Default entity priors are now neutral in training and serving; a new signed rebuild is required. | Verify source, code, and all output hashes on the corrected numerical generation. |
| 40 — Research splits | Fit preprocessing inside temporal development folds; reserve sealed evaluation and record prior exposure. | Leakage and lineage guards are implemented and covered by tests; the October 8 studies used sealed temporal splits and recorded prior exposure. | Keep the split and exposure checks in each new generation. |
| 50 — Fresh search | Run a separate Optuna study for each supported target. | Five independent 100-trial studies completed on October 8. They predate the rating-unit and causal league-prior corrections and are archived research evidence. | Run five fresh studies on the corrected generation. |
| 60 — Full refit | Apply selected study parameters to research-only models. | October 8 refit `20261008T170701_385063Z` completed five targets. Its registered candidate is now research-only because it predates the rating-unit and causal league-prior corrections. | Fit all five targets from the new studies; keep training success separate from promotion. |
| 70 — Replay and model review | Compare sealed feature values with the real serving path; inspect calibration and cohort gates. | Wider calculation checks exposed season, team-transfer, role, league-context, and source-ID defects. Regression fixes pass; the August champion and pre-correction October candidate are disabled for owner-facing use. | Run signed replay on the new candidate and require a passing sealed-row review before promotion. |
| 80 — Prospective fixture population | Enroll scheduled fixtures before forecasts and prices. | Earlier fixture cohorts remain recorded. The owner will choose relevant leagues and fixtures after model validation. | Confirm the prospective population and collect complete coverage, including missing markets and no-bet decisions. |
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

## Archived research results

The results below predate the October 9 rating-unit and causal league-prior corrections. Their
models are research-only and are not approved for owner-facing paper proposals.
New generation results must replace them after refitting and review.

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
for changed or damaged inputs. Serving retains checksum validation on every snapshot read and caches at most
two decoded tables, returning isolated copies. A ten-team construction profile
improved from 4.76 to 3.92 seconds; this is not a benchmark of the full daily run. Profile memory and CPU before changing the dataframe backend or
adding infrastructure. FireDucks remains opt-in and is not in the locked
installation while its dependency pins an Arrow version flagged by the
repository's dependency review.

See the [research readiness plan](plans/2026-09-24-1117-fix-paper-research-readiness-plan.md)
for detailed gates and the separate
[tennis plan](plans/2026-09-06-0050-feat-tennis-betting-module-plan.md) for
the later expansion.

## Causal entity-rating priors

Elo, Glicko, Plackett–Luce, and TrueSkill use neutral, predeclared league
priors by default. They no longer load the final league-Elo table, whose
results include games later than historical training rows. Serving uses the
same neutral transfer priors. League changes still reset skill uncertainty;
separate chronological league-strength features remain available. Explicit
caller-supplied priors must themselves be fixed before the evaluated games.
Four regression tests verify that modifying only the final league table cannot
change earlier default entity ratings. Freshness guards stopped the queued
search before any trials because the cached source had aged past its limit;
refresh the source before rebuilding and starting the replacement studies.
