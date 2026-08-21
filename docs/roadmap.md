# Roadmap and status board

| Status | Priority | Capability | Next gate |
| --- | --- | --- | --- |
| Done | P0 | Legacy champion containment | Champion quarantined; undecided legacy proposals corrected/expired |
| Done | P0 | Winner V2 preprocessing correctness | Signed deltas retained, stale opponent deltas overwritten, production correlation/CV pruning removed |
| Done | P0 | Direct historical series reconstruction | Sequential BO1/3/5 rules, frozen Map-1 features, rejection manifest |
| Done | P0 | Feature lineage and temporal partitions | Source/availability/swap/eligibility lineage; timestamp- and series-atomic 60/10/10/5/15 lifecycle |
| Done | P0 | Direct-series model family | Rating logistic baseline, ten week-block LightGBM members, held-out blend/calibration/bound |
| Done | P0 | Fail-closed serving health | Direct ratings, parity, symmetry contract, forbidden fields, bundle actionability |
| Done | P0 | Immutable serving artifact ownership | History refresh writes league-rating inputs under processed data; only candidate promotion writes serving bundles |
| Done | P0 | Winner-only paper policy | Model favorite, 52.5%, 5% conservative edge, 48–24h, disagreement and attribution quarantines |
| Done | P0 | Safe owner acceptance | Fresh two-book re-quote plus separate 120-second Confirm; flat one-unit evidence |
| Done | P0 | Market contract safety | Resolution source and token orientation required; failures isolated per token; no trading surface |
| Done | P0 | Exact owner-selected market review | Canonical Polymarket event URLs use exact slug lookup, typed series contracts, executable CLOB quotes, one report pair, and explicit `--publish` |
| Done | P0 | Gateway-only delivery | Durable send intent, history recovery, process lock; no webhook duplication path |
| Done | P1 | Manual owner settlement | Win/Loss/Push/Void with source reference; no automatic reconciliation |
| Done | P2 | Read-only timing observer | Hourly evidence/report snapshots; provisional 48–24h policy |
| Done | P2 | Experimental next-map dataset/model | Frozen prematch plus manually sourced map state; shadow-only and excluded from the normal bundle |
| Review | P0 | Latest series-winner Optuna study | Run `20260821T112810_829000Z` improved aggregate and actionable log loss versus its rating baseline; hard performance gates passed, while calibration-intercept, holdout-reuse, and small non-actionable cohort findings remain explicit paper-mode warnings |
| Done | P0 | Calibration-safe tuning selection | Unsafe slope/intercept candidates cannot win calibration selection merely through lower log loss |
| Done | P0 | Experimental next-map tuning review | Independent study required; bootstrap and cohorts cluster by series; it is not required by normal training or promotion |
| Blocked | P0 | First next-map Optuna study | Run `20260814T005027_980518Z` improved aggregate log loss by 2.72% but improvement was not statistically proven; ECE, calibration intercept, and multiple cohorts failed; parameters were not promoted |
| Done | P0 | Holdout exposure disclosure | Reviews record date window/label hash and flag overlapping evidence; reused evidence is never described as fresh confirmation |
| Now | P0 | Clean V2 rebuild and manual first promotion | Review/promote the superior direct-series parameters, run the five-target fixed-parameter full-feature build, then promote only if structural, aggregate, and actionable hard gates pass |
| Next | P1 | Accumulate prematch paper evidence | At least 200 manually settled signals with positive lower-95% ROI and CLV plus healthy calibration |
| Later | P2 | Reactive evidence sufficiency | Separate 200-series-clustered gate; no martingale or overlapping prematch exposure |
| Later | P2 | Provider-backed settlement | Explicit opt-in only after identity/rule/conflict/correction proof |
| Later | P3 | Portfolio sizing | Correlation-aware exposure before any real-money sizing claim |
| Later | P3 | Counter-Strike / real-sport adapters | Only after LoL has credible sealed and settled evidence |

The repository is not proven profitable. The legacy champion remains
non-actionable and proposal publication stays disabled until a complete healthy
V2 bundle exists. The next-map study remains experimental and cannot delay the
direct-series paper workflow.
Real-money consideration remains blocked until every evidence gate is met;
individual predictions can always lose.
