# Roadmap and status board

| Status | Priority | Capability | Next gate |
| --- | --- | --- | --- |
| Done | P0 | Legacy champion containment | Champion quarantined; undecided legacy proposals corrected/expired |
| Done | P0 | Winner V2 preprocessing correctness | Signed deltas retained, stale opponent deltas overwritten, production correlation/CV pruning removed |
| Done | P0 | Direct historical series reconstruction | Sequential BO1/3/5 rules, frozen Map-1 features, rejection manifest |
| Done | P0 | Feature lineage and temporal partitions | Source/availability/swap/eligibility lineage; timestamp- and series-atomic 60/10/10/5/15 lifecycle |
| Done | P0 | Direct-series model family | Rating logistic baseline, ten week-block LightGBM members, held-out blend/calibration/bound |
| Done | P0 | Fail-closed serving health | Direct ratings, parity, symmetry contract, forbidden fields, bundle actionability |
| Done | P0 | Winner-only paper policy | Model favorite, 52.5%, 5% conservative edge, 48–24h, disagreement and attribution quarantines |
| Done | P0 | Safe owner acceptance | Fresh two-book re-quote plus separate 120-second Confirm; flat one-unit evidence |
| Done | P0 | Market contract safety | Resolution source and token orientation required; failures isolated per token; no trading surface |
| Done | P0 | Gateway-only delivery | Durable send intent, history recovery, process lock; no webhook duplication path |
| Done | P1 | Manual owner settlement | Win/Loss/Push/Void with source reference; no automatic reconciliation |
| Done | P2 | Read-only timing observer | Hourly evidence/report snapshots; provisional 48–24h policy |
| Done | P2 | Experimental next-map dataset/model | Frozen prematch plus manually sourced map state; shadow-only safety |
| Now | P0 | One explicit Winner V2 Optuna study | Search only rolling-origin folds inside development; owner reviews fixed parameters |
| Next | P0 | Clean V2 rebuild and manual first promotion | Build series, train complete bundle, pass sealed rating-baseline/calibration/cohort gates |
| Next | P1 | Accumulate prematch paper evidence | At least 200 manually settled signals with positive lower-95% ROI and CLV plus healthy calibration |
| Later | P2 | Reactive evidence sufficiency | Separate 200-series-clustered gate; no martingale or overlapping prematch exposure |
| Later | P2 | Provider-backed settlement | Explicit opt-in only after identity/rule/conflict/correction proof |
| Later | P3 | Portfolio sizing | Correlation-aware exposure before any real-money sizing claim |
| Later | P3 | Counter-Strike / real-sport adapters | Only after LoL has credible sealed and settled evidence |

The repository is ready for a V2 research retune and rebuild, not proven
profitable. Real-money consideration stays blocked until the evidence gates are
met; individual predictions can always lose.
