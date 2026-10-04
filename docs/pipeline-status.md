# Pipeline status

_Local verification snapshot: 2026-10-04. Generated datasets, model bundles,
quotes, and the evidence database are local state; cloning GitHub does not
include them._

The numbers below are milestones in the workflow, not a model-quality score or
an estimate of profitability. Research and paper collection can proceed in
parallel, but a later milestone cannot compensate for a failed earlier gate.

| Milestone | What happens | Current evidence | Next gate |
| --- | --- | --- | --- |
| 0 — Source and configuration | Install locked dependencies; keep provider credentials outside Git. | Lock, CLI, and CI pass on merged `main`; the latest local suite passed 721 tests with 1 skipped. | Check setup on a second machine. |
| 10 — Ingest | Refresh the public Oracle's Elixir source and retain a versioned raw snapshot. | Source check ready through 2026-10-04 08:57 UTC. | Continue checking source freshness before each rebuild. |
| 20 — Clean data and identities | Quarantine bad rows; normalize teams, players, maps, and series. | Latest local rebuild: 24,386 games, 10,184 accepted series, and 1,770 quarantined groups. | Review quarantined groups when source or rules change. |
| 30 — Features and targets | Build chronological team/player features, ratings, and paired serving history. | Paired map, series, and serving snapshots were built. A later evidence-code commit changed the broad code fingerprint, so the tables require a new generation before training on current `main`. | Rebuild after the serving-feature corrections, then validate source, code, and table hashes together. |
| 40 — Research splits | Fit preprocessing inside temporal development folds; reserve sealed evaluation and record prior exposure. | Leakage and lineage guards are implemented and covered by tests; the October 4 research run used sealed temporal splits. | Keep the split and exposure checks in the next generation. |
| 50 — Fresh search | Run a separate Optuna study for each supported target. | Five 100-trial studies completed on commit `cc93624`, one each for map winner, series winner, game length, kills, and towers. They are stale after code and source changes. | Repeat only after serving-feature parity is corrected. |
| 60 — Full refit | Apply the selected study parameters to research-only models. | Research-only refit `20261004T153158_723319Z` completed all five targets with no failures and no promotion; it is stale against current `main`. | Review the sealed evidence, then refit on the corrected generation. |
| 70 — Replay and model review | Compare sealed feature values with the real serving path; inspect calibration and cohort gates. | The candidate lineage check was fixed in PR #5. Diagnostic replay reached the values and found broad mismatches: 229 of 243 model-input columns differed on the first sampled map row. No candidate was promoted. | Align training and serving transforms, run the official replay on a fresh generation, and review calibration before promotion. |
| 80 — Prospective fixture population | Enroll scheduled fixtures before forecasts and prices. | Cloud9–LYON was enrolled for October 3. Team Liquid–LYON was enrolled at 16:34 UTC before its October 4 20:00 UTC start after PR #6 corrected a shared provider-series ID collision. | Continue complete enrollment and record missing markets and no-bet decisions. |
| 90 — Market and paper cycle | Capture exact Polymarket/Thunderpick contracts, owner-confirm quotes, size positive-edge paper entries, record close and result. | The October 4 Polymarket review captured 55 contracts and 42 quote observations before start; four required market families have observed quotes. All priced actions remain exploration-only. No verified Thunderpick line, paper entry, close, or settlement exists. | Obtain owner-observed executable lines and terms; then complete a paper entry-to-settlement cycle. |
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

The October 4 full rebuild took 1,847.4 seconds; enrichment alone took
1,661.7 seconds. An earlier daily run with no changed history also spent about
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
