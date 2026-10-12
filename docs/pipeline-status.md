# Pipeline status

_Verification updated October 12, 2026, Europe/Rome. Data refreshed October 12;
historical replay retains its October 10 observation date. Datasets, model bundles, quote records,
and the evidence database are local state; a GitHub clone does not include them._

The active paper champion is **`lol-20261010-paper-v1`**. The August artifact is
quarantined. Its recorded recipe contained a look-ahead defect in entity-rating
priors, so its better historical score cannot establish a clean benchmark.
Conventional overfitting and the size of any score inflation have not been proved.
The [locked protocol](lol-paper-protocol.md) explains the comparison and selection.

| Milestone | Current evidence | Remaining gate |
| --- | --- | --- |
| 0 — Setup | Current local suite passes 839 tests, 1 skipped; lint, formatting, package types, compilation and strict docs pass. The reviewed code commit passed all four GitHub checks; live refresh and verified feature reuse pass. See [PR #23](https://github.com/Yureehh/Oracle-Bets/pull/23) for final check receipts. | Second-machine setup remains unverified. |
| 10 — Ingestion | Retained source extends through October 11, 2026, 15:58:23 UTC; October 12 source freshness passes. | Refresh and check source freshness before daily decisions. |
| 20 — Cleaning | 24,544 games, 10,234 accepted series, 1,770 quarantined groups. | Review new source and quarantine changes. |
| 30 — Features | Neutral causal rating priors, signed input tables and paired serving snapshot `features-5f5eaf01d697dac50a2fde67` rebuilt October 12; input and artifact health pass. | Validate lineage after each changed source/code generation. |
| 40 — Splits | Temporal safeguards and exposure guards; winner partitions are 73/2/2.5/2.5/5/15, with series base training through February 1, 2026. | Reused historical evaluation is diagnostic, not untouched prospective evidence. |
| 50 — Search | Five fresh 100-trial Optuna studies completed. | Do not repeat selection against the exposed holdout. |
| 60 — Full refit | All five targets completed in `20261009T231405_688954Z`. Paper packaging preserves every trained artifact checksum. | Repackaging is not another training run or production parameter approval. |
| 70 — Replay and selection | 581 matches, zero mismatches; 59 sampled rows unavailable. Structural winner and market validation pass. Normal registry promotion selected the paper package against the causal baseline. | Calibration-intercept and conservative-bound coverage warnings keep every strategy exploratory/display-only. |
| 80 — Prospective population | Model/window locked; 25 selected competitions and 336-hour horizon implemented. October 12 run enrolled three provider fixtures before inference; seven forecasts were recorded for one supported fixture, with two unsupported-team gaps retained. | Collect complete prospective coverage; missing stages remain in the denominator. |
| 90 — Paper cycle | Evidence integrity/schema checks pass; October 12 daily discovery recorded six observations with zero provider failures. Executable quote snapshots remain unchanged; no paper tickets, settlements, or complete prospective cycle yet. | Owner-confirm exact executable quotes, capped stakes, closes, and sourced settlements. |
| 100 — Decision evidence | No representative settled paper sample or verified earnings. | Evaluate the locked population and policy after the window; insufficient evidence is inconclusive. |

Use the [daily operating guide](daily-operation.md), then resume the
[owner checklist](paper-owner-checklist.md) at point 4. The €500 capital declaration
is implemented; Polymarket conversion evidence remains owner-supplied before entry. Model weights,
parameters, calibration, and uncertainty are frozen for fixtures scheduled
October 11 through November 30, Europe/Rome. Daily features can advance using
completed games available before the decision. Recommendation and real-stake
activation remain disabled; the active registry scope is `paper_research_only`.

## Current model results and approval limits

The paper package retains weights from `lol-20261009T231405_688954Z`:

| Target | Fresh study | Historical test result |
| --- | --- | --- |
| Map winner | `20261009T222436_913278Z` | Log loss 0.632618; accuracy 65.09%; calibration slope 0.720. |
| Series winner | `20261009T222945_303975Z` | Log loss 0.562790; accuracy 71.42%; Brier 0.190816; calibration error 0.035164; calibration slope 1.135. |
| Game length | `20261009T223509_551567Z` | MAE 4.246 minutes; R² 0.061. |
| Total kills | `20261009T224609_486040Z` | MAE 7.520 kills; R² 0.090. |
| Total towers | `20261009T230527_940573Z` | MAE 1.715 towers; R² 0.00033. |

The existing first-valid-model review compared the corrected series model with
its frozen causal rating baseline: log loss **0.562790 versus 0.566831**. The
paired-bootstrap degradation upper bound was **0.839%**, below its existing
1% noninferiority limit. The review returned `manual_review_required` with no
blocking reasons; the owner-authorized normal promotion completed. This does
not prove a greater than 1% improvement or market edge.

The earlier comparison against August remains archived and blocked: candidate
log loss 0.562790 versus 0.556720, Brier 0.190816 versus 0.188261, and calibration
error 0.035164 versus 0.024885. The separate Optuna parameter review also remains
blocked by insufficient improvement, conservative coverage, and prior holdout
exposure. No production parameter configuration was published. Replacing an
invalid legacy comparator does not remove strategy-readiness warnings.

A coverage-report bug aligned sliced dataframe indices with reset positional
columns, silently omitting cohort membership. The fix resets metadata indices
before assembling the report; two offset-index regressions reproduce and guard
the defect. Corrected reports expose the league failures and preserve original
reports. Probabilities and trained weights did not change.

Actual serving replay sampled 128 rows per target: 117 map, 113 series, and 117
for each prop matched. The other 59 samples lacked enough roster history and
are not counted as passes. Replay verifies calculation parity, not historical
availability of the later-published snapshot. Sealed predictions exported for
the paper package are byte-identical to the original full-refit exports and
explicitly identified as an evaluation export rather than another training run.

All 13 structural winner checks pass. Market validation covers direct series,
map parity, legal series paths, and BO3/BO5 derived markets. Derived backtests
contain 835 BO3 and 361 BO5 series; these are probability diagnostics, not
executable-price backtests or demonstrated profits. Target-specific map,
handicap, totals, and prop readiness remains exploratory. Next-map remains
display-only. There are zero recommendation-active strategy cells.

## Local evidence and operational boundary

Receipts are under `reports/lol/training/replay/`:

- `legacy-comparator-audit.json` records the legacy recipe defect and limits.
- `paper-serving-rebuild-receipt.json` records the rebuilt input equivalence.
- `lol-20261010-paper-v1-serving-replay.json` records actual parity and unavailable rows.
- `lol-20261010-paper-v1-manual-selection.json` preserves the initial transition.
- `lol-20261010-paper-v1-normal-promotion-confirmation.json` records normal promotion.

Registry reviews live under `data/state/model-registry/lol/reviews/`; the
immutable package contains its protocol/provenance receipt under `_evaluation`.
The October 12 daily receipt is
`reports/lol/daily/20261012T182128_260428Z.json`. All twelve steps passed. Its three
provider fixtures are CBLOL/LCS/LCP promotion events, not evidence that all major
tournaments have published their fixtures. Empty provider lineups were retained;
the supported fixture used disclosed historical roster evidence. Public candidates
require exact owner review. The 90 immutable champion files retain their hashes.

A new sealed replay attempt correctly refused the advanced data generation with
`Candidate was trained from a different data generation`. Do not bypass that
guard or count the attempt as parity success. The retained 581/0/59 replay belongs
to the original generation; current-data serving was exercised by the daily run.
A future sealed replay requires its candidate-matched input generation.

These generated files are not distributed by a GitHub clone. The evidence
health check reports integrity `ok` and schema version 7. No new fixtures,
executable quotes, paper entries, closes, or results were invented during this
engineering work. The owner selects the population and supplies that evidence.

## Further engineering work

The October 12 awake feature rebuild took **2,068.8 seconds**, including
**1,853.7 seconds** in enrichment; the October 10 rebuild took 1,790.1 seconds.
A previous daily run
rebuilt features even when history had not changed. The implemented no-change
shortcut is guarded by source identity, code/config fingerprint, manifest integrity,
and complete output hashes. The unchanged October 12 repeat verified
zero added/updated rows and all hashes, reducing feature preparation (including
ingestion) from **2,068.8 to 148.4 seconds**, avoiding about 32 minutes of rebuilding.
This is a local single-run comparison, not a whole-daily-run benchmark. Changed or damaged inputs still require
the full rebuild. A ten-team serving construction profile improved from 4.76
to 3.92 seconds using bounded decoded-snapshot caching; that is not a benchmark
of the full daily run. Profile CPU and memory before changing the dataframe
backend or infrastructure. FireDucks remains opt-in outside the locked install.

Develop calibration and conservative-bound improvements on development data,
then preregister a separate candidate and evaluation epoch. Do not alter this
paper model based on the current window's outcomes. Tennis comes after the
operational cycle and evidence review; see the [tennis plan](plans/2026-09-06-0050-feat-tennis-betting-module-plan.md).
