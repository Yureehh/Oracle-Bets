# Roadmap and status

## Implemented and verified

- Independent symmetric direct-series champion plus experimental prematch map,
  legal BO3/BO5 totals, handicaps, and scalar prop forecasts.
- Owner-selected one/two-link review for Polymarket and Thunderpick.
- Public read-only Polymarket metadata/books; no unsupported-contract book
  fan-out; no Thunderpick automation.
- Exact-semantic provider comparison with model probability, fair odds, implied
  probability, EV, and best provider.
- Compact Discord/Markdown output and detailed normalized JSON evidence.
- One owner-only `/oracle` console: Review Markets, Record Bet, Open Bets,
  Closed Bets, Schedule, Performance, and Health.
- Versioned target/cohort readiness, deterministic
  `recommended|exploration|not_comparable` policy, and frozen flat/full/half/
  quarter-Kelly paths on one ticket.
- Unified append-only paper/real ledger with result facts, manual settlement,
  correction events, cohort coverage, calibration, drawdown, CLV, and
  fixture-clustered performance.
- Guarded external archive, restore verification, exclusive reset lock, and
  short-lived single-use reset confirmation.
- No automatic acceptance, automatic settlement, betting, wallet,
  signing, bookmaker login, scraping, or fund movement.

## Current work: bootstrap the fresh paper epoch

Code verification alone does not create a valid paper epoch. The current
generated state must be archived and reset through `ops reset-plan` /
`ops reset-apply`, then current 2024–2026 history, research studies, reviewed
tuning, fixed-parameter models, registry champion, and smoke evidence must be
rebuilt from one clean commit. Any stale source, dirty provenance, failed
symmetry/train-serve check, or failed archive restore blocks the bootstrap.

After bootstrap, the owner should review every scheduled fixture in each
preregistered league-week cohort, keep recommendation and exploration evidence
separate, capture official results, settle manually, and inspect calibration,
CLV, drawdown, and equal-fixture ROI rather than raw ticket ROI alone.

Real-money recommendations remain blocked until at least 200 unique fixture
clusters from consecutively shadowed direct-series recommendation opportunities
show at least 90% schedule-review coverage, 100% result capture, healthy
calibration, positive CLV, no material cohort failure, and a positive
multiplicity-aware lower confidence bound for fixture-clustered ROI at a fixed
monthly decision date. Owner-accepted tickets are reported separately. Each
experimental target needs its own independent 200-fixture activation review.

## Next

1. Complete and record the guarded fresh-epoch bootstrap.
2. Accumulate enough settled evidence to compare direct series, prematch map,
   derived totals/handicaps, and calibrated props separately.
3. Add controlled closing-price capture for CLV only after the manual ledger has
   enough entries to justify the operational cost.
4. Improve Discord integration testing at the interaction boundary, including
   fixture selection, confirmation expiry, pagination, and restart behavior.
5. Add correlation-aware portfolio research before any real-money exposure
   recommendation.

## Later

- Draft-aware map research with timestamped champions, bans, sides, patch,
  player identities, source references, exact swap symmetry, and a separate
  evidence cohort.
- Reactive next-map research using frozen prematch features plus manually
  sourced series state; no martingale or loss chasing.
- Counter-Strike and real-sport adapters only after LoL paper evidence is
  credible.
