# Repository review and verification

Reviewed October 12, 2026. The audit covered the 225 tracked files present at
its baseline, including source, tests, configuration, documentation, notebooks,
CI, and dependency metadata. New implementation files were checked during
integration. This records coverage and limits; it is not a bug-free guarantee.

## Coverage

The core audit read 51 assigned files (31 source, 15 tests, five build/config).
The Discord audit read its 16 assigned files and connected callers. The LoL
audit read all 137 assigned files: 61 source, 50 tests, 22 configuration files,
and all cells/metadata in four notebooks. The host reviewed remaining docs,
configuration and repository metadata. Dependency lock entries were inspected;
package download hashes were not independently re-downloaded for this audit.
Generated datasets, immutable models, reports and ledger were inventoried and
retained. Their full historical contents were not manually read row by row.

## Changes justified by findings

- Preserve zero probability bounds and bind prop calibration to the forecast's
  immutable model, including model switching and changed-content detection.
- Validate cross-page fixture identity; remove interrupted download archives;
  reject ambiguous manual observation timestamps and retain provider failures.
- Declare €500 capital split across venues. Verify currency conversion, derive
  bankroll from corrected realized PnL, freeze declaration after first entry,
  and reserve venue-specific open exposure.
- Reject correction cycles, cross-ticket settlement replacements, and impossible
  event times. Net simultaneous settlements before calculating drawdown.
- Label monthly activity and lifetime summaries separately.
- Paginate Discord choices, allow recording a closing quote for an open ticket,
  correct the managed-worker path, and expose paper scope/readiness warnings.
- Remove the unused row-wise head-to-head matchup computation. A prior measured
  comparison on 48,916 team rows produced identical frames and took 0.402 vs
  0.175 seconds; this is a local step measurement, not total pipeline speed.
- Reuse a feature generation only with identical source identity, code/config,
  manifests, paired snapshot and complete output hashes. Missing or changed
  artifacts take the full rebuild.
- Add prospective daily forecasts and bounded public listing observations;
  preserve observation time and distinguish absence from provider failure.
- Consolidate operation into the daily and system guides; remove the superseded
  September readiness plan, retaining its Git history. Keep the tennis design
  explicitly marked as deferred.

## Verification and remaining limits

The integrated suite passed 825 tests, one skipped after the final regressions.
Live-run results are recorded in the pipeline status when available. Lint and
package type checking pass. Notebooks and test monkeypatches are outside the
configured CI type-check scope; a wider exploratory type check reports issues
there and is not represented as a passing check.

The dedicated final code-review attempt failed before starting because the
selected model was at capacity. Code review: skipped (ce-code-review unavailable).
The host performs a manual diff scan; this is not independent reviewer approval.
Simplification used the loaded reuse, quality and efficiency rubrics locally
because review-agent dispatch was unavailable. The pass retained the shared total-map
probability-bound helper and existing lineage/file-hash utilities, removed duplicate
daily message-summary updates, and kept all validation guards. No speculative
architecture or extra dependency was introduced. The manual diff scan caught and
fixed unreported predictor-initialization failures; a regression verifies that the
daily report retains suppression and the failure instead of raising out of the run.

Historical series reconstruction still infers BO3 from observed 2–0/2–1 games
without authoritative format/completeness metadata. A partially ingested BO5
can look complete. Fixing that requires trusted metadata or a separately
reviewed tail-quarantine policy; do not silently retrain the frozen champion.
Global seasonal rating reset loops merit profiling before optimization.
Second-machine installation, live Discord button interaction, exact provider
contract/price review, and a complete quote-to-settlement cycle remain unverified.
There is no proven market edge, optimized Kelly policy, or earnings evidence.
