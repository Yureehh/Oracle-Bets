# Roadmap and status board

This is the repository's small Kanban. `Done` means code and tests exist;
`Now` is the next acceptance gate; `Next` is valuable after that gate; `Later`
is intentionally deferred. Issues/PRs can replace this board if the project
gains multiple active maintainers.

| Status | Priority | Capability | Evidence / next gate |
| --- | --- | --- | --- |
| Done | P0 | Canonical winner matchup schema | Reverse-order probabilities are exact complements; side/first-pick leakage tests |
| Done | P0 | Temporal train/calibrate/test separation | Whole-game chronological splits and held-out calibration metrics |
| Done | P0 | Local 2024–2026 Oracle's Elixir ingestion | Local Drive source, history reconciliation, quality manifests |
| Done | P0 | Bounded logging and durable daily reports | Rotating logs; atomic JSON/Markdown output |
| Done | P0 | One canonical evidence database and model registry | Schema/health/backup tests; immutable checksummed candidates |
| Done | P0 | Conservative read-only market infrastructure | Strict identity/time/BO match and public order-book unit tests; no execution surface |
| Now | P0 | Complete one clean fixed-parameter rebuild | Refresh 2024–2026 history, retrain all four targets without Optuna, review sealed evidence, and promote one healthy bootstrap bundle |
| Done | P0 | Typed daily market assessment and action gates | Exact Game N/series/totals matching, two-book executable pricing, target gates and exposure caps |
| Done | P0 | Sealed-row champion/challenger lifecycle | Routine auto-promotion only after paired calibration/cohort/prop/artifact gates; Optuna stays manual |
| Done | P1 | Paper evidence and owner controls | Forecast/quote/decision/position/manual owner settlement chain, persistent Discord controls |
| Now | P1 | Accumulate real paper evidence | Capture actual decisions, prices, rejections, settlements and available pre-start closes without hindsight edits |
| Later | P2 | Opt-in provider-backed settlement | Verified outcome identity, captured resolution rules, conflicts/corrections, tests, and explicit owner opt-in before any automation |
| Done | P2 | Crash-safe Discord publication recovery | Durable send intent, deterministic marker, bounded channel-history recovery, and one Gateway process lock |
| Next | P1 | Profit evidence sufficiency | Positive CLV/ROI lower bounds, controlled drawdown, calibration and enough samples by target/cohort |
| In progress | P2 | Feature-family ablation and drift | Reports show missingness, league/prediction shift, and attribution stability; temporal family ablations remain evidence work |
| Next | P2 | Decompose oversized operational modules | Split daily orchestration, evidence persistence, model lifecycle, CLI handlers, market adapters, and GBDT calibration behind existing public imports; preserve pickle compatibility and rerun characterization suites after each move |
| Done | P2 | Conservative historical lineup fallback | Only an absent provider lineup can use one exact role-mapped five repeated across the latest three completed consecutive series |
| Later | P3 | Counter-Strike / real-sport adapters | Reuse narrow core contracts only after LoL has credible calibration and profit evidence |

The system is research-ready, not proven profitable. No model or architecture
can guarantee wealth; promotion and staking must be driven by sufficient
out-of-time and settled-market evidence.
