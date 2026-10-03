# Pipeline status

_Local verification snapshot: 2026-10-03. Generated datasets, model bundles,
quotes, and the evidence database are local state; cloning GitHub does not
include them._

The numbers below are milestones in the workflow, not a model-quality score or
an estimate of profitability. Research and paper collection can proceed in
parallel, but a later milestone cannot compensate for a failed earlier gate.

| Milestone | What happens | Current evidence | Next gate |
| --- | --- | --- | --- |
| 0 — Source and configuration | Install locked dependencies; keep provider credentials outside Git. | Lock, CLI, and CI contracts exist. | Check setup on a second machine. |
| 10 — Ingest | Refresh the public Oracle's Elixir source and retain a versioned raw snapshot. | Source check ready through 2026-10-02; full retained history rebuilt. | Continue checking source freshness before each rebuild. |
| 20 — Clean data and identities | Quarantine bad rows; normalize teams, players, maps, and series. | 24,329 games and 10,160 accepted series in the local rebuild; identity sync completed. | Review quarantined groups when source or rules change. |
| 30 — Features and targets | Build chronological team/player features, ratings, and paired serving history. | Map and series training tables and a paired serving snapshot were built and validated. | Keep source, code, and table hashes aligned after any source edit. |
| 40 — Research splits | Fit preprocessing inside temporal development folds; reserve sealed evaluation and record prior exposure. | Leakage and lineage guards are implemented and covered by tests. | Verify them on the fresh training run. |
| 50 — Fresh search | Run a separate Optuna study for each supported target. | **Pending.** No new study has run on this generation. | Start only from a clean, reviewed source commit. |
| 60 — Full refit | Apply the selected study parameters to research-only models. | **Pending.** Existing champion artifacts are from August. | Complete all studies, then use `lol refit-research` without auto-promotion. |
| 70 — Replay and model review | Compare sealed feature values with the real serving path; inspect calibration and cohort gates. | Comparator works and correctly rejects the old candidate as a different data generation. | Replay the new candidate, resolve mismatches, and review before any promotion. |
| 80 — Prospective fixture population | Enroll scheduled fixtures before forecasts and prices. | One Cloud9–LYON cohort was enrolled for 2026-10-03; daily workflow and owner bot ran. | Keep future cohorts complete, including missing markets and no-bet decisions. |
| 90 — Market and paper cycle | Capture exact Polymarket/Thunderpick contracts, owner-confirm quotes, size positive-edge paper entries, record close and result. | Code and required-market checklist exist; **no complete live cycle** is verified. Polymarket timed out in the latest read-only check. | Obtain owner-observed URLs, lines, prices, periods, and terms; then run one complete paper cycle. |
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

## Performance work after correctness

The first full historical rebuild was expensive, as expected. More
significantly, a later daily run with no changed history still spent roughly
30 minutes rebuilding features. The next optimization is a no-change shortcut
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
