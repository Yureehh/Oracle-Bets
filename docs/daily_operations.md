# Daily operations

## Ownership and cadence

| Cadence | Owner | Command/artifact | Why |
| --- | --- | --- | --- |
| Daily, 00:15 | `launchd` | `oracle-bets daily lol` | Fixtures, incremental history, conditional retraining, health gates, predictions, read-only market comparison, reports, Discord |
| Every training event | training code | `reports/lol/training/<run-id>/` | Metrics, calibration, attribution, manifest, model cards; no separate report job |
| Weekly | none required | Review the latest compact daily/training reports only | Do not retrain or retune just because a week elapsed |
| Monthly, day 1 | `launchd` | `oracle-bets audit monthly` | Settled evidence, calibration/profit sample health, drawdown and owner review |
| On demand | owner | retune, promotion, rollback, settlement, notebooks, scratch rebuild | These actions need evidence or an explicit research decision |

The production operator is normal Python plus `launchd`, not an LLM. This makes
the result reproducible and costs no OpenAI tokens. If an OpenAI summarizer is
added later, give it only the latest compact JSON report, call it only when
there are visible predictions, cap its output, and let it produce prose
proposals only. It must never run the pipeline, promote a model, place a bet,
sign anything, access a wallet, or move funds. The owner remains the only bet
decision-maker.

## Daily workflow

The midnight job performs:

1. fetch the next 36 hours of PandaScore fixtures;
2. refresh expected lineups when credentials permit;
3. exclude blank/TBD teams and Equal eSports Cup;
4. refresh 2024–2026 historical data;
5. validate source and supervised artifacts;
6. evaluate whether new-map thresholds justify routine retraining;
7. retrain without Optuna when triggered;
8. validate model/calibrator and registry health;
9. resolve teams and produce symmetric winner, series, and prop predictions;
10. compare conservatively with read-only Polymarket candidates;
11. write JSON and Markdown reports atomically;
12. only then send the optional Discord webhook;
13. append canonical evidence for non-dry runs.

Every run, including `--dry-run`, writes
`reports/lol/daily/<UTC timestamp>.json` and `.md`. Unsupported teams and
excluded fixtures appear compactly in the report rather than as noisy repeated
terminal errors. Webhook URLs are never serialized, and Discord mentions are
disabled.

If schedule refresh fails, a stored schedule may be used with an explicit
warning. A failed required mutation suppresses predictions. Report persistence
failure prevents webhook delivery.

### PandaScore lineup enrichment

Schedule discovery and match-detail enrichment are separate. The workflow uses
PandaScore's generic `/matches/{id}` detail endpoint and never requests a
missing provider ID. A 401/403 disables further detail requests for that run,
keeps any lineups embedded in the schedule response, and records a single
compact enrichment warning. It does not turn a valid schedule into a pipeline
failure. Predictions without a complete expected lineup remain shadow-only.

Use the two examples under `ops/launchd/` for the supported recurring jobs.
Application logging is bounded; launchd stdout/stderr point to `/dev/null`.
