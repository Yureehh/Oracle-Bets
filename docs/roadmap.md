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
| Now | P0 | Complete one clean research rebuild | Retune all four targets, review untouched temporal test/calibration/attribution, promote bundle, retrain |
| Now | P0 | Connect strict market assessment to the daily presentation | Replace legacy text-score display matching; preserve manual verification and no-bet default |
| Now | P0 | Establish action gates per target | A weak prop model must be labelled research-only and must not generate a stake proposal |
| Now | P1 | Accumulate real paper evidence | Capture decision/open/closing prices, rejections and settlements without hindsight edits |
| Next | P1 | Walk-forward champion/challenger evaluation | Non-inferior log loss/Brier/ECE plus league/patch stability before promotion |
| Next | P1 | Profit evidence with uncertainty | CLV, ROI confidence intervals, drawdown, sample counts by league/market/edge bucket |
| Next | P2 | Feature-family ablation and drift | Temporal permutation/SHAP stability; remove signals that fail out-of-time evidence |
| Next | P2 | Roster/lineup quality | Improve provider coverage without fabricating unavailable pre-match facts |
| Later | P3 | Counter-Strike / real-sport adapters | Reuse narrow core contracts only after LoL has credible calibration and profit evidence |

The system is research-ready, not proven profitable. No model or architecture
can guarantee wealth; promotion and staking must be driven by sufficient
out-of-time and settled-market evidence.
