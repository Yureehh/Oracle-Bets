# Pipeline status

_Local verification snapshot: 2026-10-10 (Europe/Rome). Generated datasets, model bundles,
quotes, and the evidence database are local state; cloning GitHub does not
include them._

The numbers below are milestones in the workflow, not a model-quality score or
an estimate of profitability. Research and paper collection can proceed in
parallel, but a later milestone cannot compensate for a failed earlier gate.

| Milestone | What happens | Current evidence | Next gate |
| --- | --- | --- | --- |
| 0 — Source and configuration | Install locked dependencies; keep provider credentials outside Git. | The serving and rating corrections pass 761 tests, with 1 skipped locally; lint, formatting, and types pass. | Require green final PR checks and check setup on a second machine. |
| 10 — Ingest | Refresh the public Oracle's Elixir source and retain a versioned raw snapshot. | The validated source is current through 2026-10-08 22:56 UTC. | Continue checking source freshness before each rebuild. |
| 20 — Clean data and identities | Quarantine bad rows; normalize teams, players, maps, and series. | The corrected October 9 generation has 24,458 games, 10,203 accepted series, and 1,770 quarantined groups. | Review quarantined groups when source or rules change. |
| 30 — Features and targets | Build chronological team/player features, ratings, and paired serving history. | The corrected generation uses neutral entity-rating league priors, signed training tables, and a paired serving snapshot. All 581 available comparisons in the 128-per-target replay matched; 59 sampled rows were unavailable because of insufficient roster history. | Verify source, code, and all output hashes on the corrected numerical generation. |
| 40 — Research splits | Fit preprocessing inside temporal development folds; reserve sealed evaluation and record prior exposure. | Leakage and lineage guards pass. The completed recency experiment preserved the exact uncertainty window and all 1,529 final series test IDs. | Keep the exposure checks; the reused historical holdout is research evidence, not untouched prospective validation. |
| 50 — Fresh search | Run a separate Optuna study for each supported target. | All five replacement 100-trial studies completed on the frozen, corrected generation. The earlier five studies also remain archived. | The one-cycle research cap is exhausted. Develop further changes on development data and predeclare a later evaluation window. |
| 60 — Full refit | Apply selected study parameters to research-only models. | Replacement refit `20261009T231405_688954Z` completed all five targets, with no failures. Its candidate is research-only after promotion failed. | Keep successful fitting separate from model and production-parameter approval. |
| 70 — Replay and model review | Compare sealed feature values with the real serving path; inspect calibration and cohort gates. | Replacement replay passed all 581 available comparisons, with 59 unavailable samples. Series log loss improved to 0.562790 but remained worse than the 0.556720 benchmark; Brier and calibration error also regressed. August and all three October candidates are disabled for owner-facing use. | A new champion remains unapproved. Require the same improvement, calibration, replay, and readiness gates on a future candidate. |
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
change earlier default entity ratings. Freshness guards stopped an earlier queued
search before any trials because the cached source had aged past its limit.
The subsequent refresh and rebuild passed these guards; both corrected research
cycles used the retained source current through October 8.

## Winner-recency research cycle

The corrected October 9 run is fresh as an artifact, but its original 55%
training partition stops the series base learner on May 13, 2025. Its
calibration then spans much of the following year. The candidate passed
calculation parity but failed model promotion: series log loss was 6.14% worse
than the benchmark, with tier-one/ERL regression. No new champion was promoted.

One replacement research cycle is preregistered with timestamp-atomic
73/2/2.5/2.5/5/15 partitions for training, tuning, calibration fitting,
calibration selection, uncertainty fitting, and testing. On the retained
source this moves series training to 7,447 rows through February 1, 2026.
The original uncertainty window and all 1,529 final series test IDs remain
exactly unchanged. The calibration periods become smaller and more recent;
minimum support, calibration diversity, timestamp separation, and all model
promotion and market-readiness checks still apply. This is an experiment,
not an assurance of improved performance.

The research cap is one new five-target search/refit/replay cycle. If it still
fails the unchanged promotion gates, record that failure rather than repeatedly
selecting against the same test outcomes. The reused historical holdout has
known prior exposure and cannot establish an independent earning claim.
Prospective owner-selected fixtures, executable quotes, closes, and settled
results remain necessary before stronger strategy claims.

The splitter also now reserves a timestamp for every remaining partition,
fixing its previous failure on the documented six-timestamp minimum.
The running Discord worker was refreshed silently after the code fixes.

## Completed replacement review

All five replacement studies and the full refit completed. The immutable
candidate is `lol-20261009T231405_688954Z`; its base series learner trains
through February 1, 2026, rather than May 2025. The serving rebuild verified
that the numerical input files were byte-identical to the preceding corrected
generation. The recipe changed; the source, uncertainty window, and final
series test population did not.

| Target | Study run | Refit test result |
| --- | --- | --- |
| Map winner | `20261009T222436_913278Z` | Log loss 0.632618; accuracy 65.09%; calibration slope 0.720. |
| Series winner | `20261009T222945_303975Z` | Log loss 0.562790; accuracy 71.42%; calibration slope 1.135. |
| Game length | `20261009T223509_551567Z` | MAE 4.246 minutes; R² 0.061. |
| Total kills | `20261009T224609_486040Z` | MAE 7.520 kills; R² 0.090. |
| Total towers | `20261009T230527_940573Z` | MAE 1.715 towers; R² 0.00033. |

The actual 128-per-target serving replay matched 117 map rows, 113 series
rows, and 117 rows for each prop target: **581 matches, zero mismatches**.
The other 59 samples lacked sufficient roster history and were unavailable,
not counted as passes. This reconstructs historical calculation values; it
does not prove that the later-published snapshot was available prospectively.

The unchanged Optuna promotion review **blocked** the replacement:

- Series log loss was 1.09% worse than the archived benchmark, not the required
  proven improvement. The paired-bootstrap improvement lower bound was −2.095%.
- Series Brier score was 0.190816 versus 0.188261.
- Series calibration error was 0.035164 versus 0.024885.
- The tier-one/ERL cohort had log loss 0.566670 versus 0.558482 on 1,147 series.
  This remained worse, although it did not trigger the separate cohort-regression gate.

The series parameter review was also blocked by insufficient improvement,
failed conservative-probability coverage, and prior holdout exposure. No
replacement parameters were published as production settings. The registry
still points to the August artifact for historical identity, but that artifact
is quarantined and cannot issue owner-facing forecasts. Neither failed October
replacement was promoted. A healthy checksum is not model approval.

Local receipts are retained under `reports/lol/training/replay/`, including
`recency-research-cycle.json`, `recency-validation-outcome.json`, and
`lol-20261009T231405_688954Z-corrected-20261009T234448Z.json`. The sealed review
is under `data/state/model-registry/lol/reviews/`. These generated files are
not distributed by a GitHub clone.

The remaining model work is to diagnose calibration and conservative-bound
coverage using development partitions, then preregister a new candidate and
a later evaluation window before viewing its outcomes. Do not weaken gates or
repeat selection on this exposed test set. Meaningful owner-selected fixtures
and complete prospective evidence remain pending; there are still no paper
bets, settlements, or demonstrated earnings. A second-machine setup check also
remains unverified.
