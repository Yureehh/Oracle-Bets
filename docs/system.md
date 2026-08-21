# System design and operating decisions

Oracle Bets is a local-first League of Legends betting-research system. It
produces independent probabilities, compares them with public executable market
quotes, and records paper decisions. Correct calibration and honest evidence
matter more than prediction volume. The repository contains no betting,
signing, wallet, private-key, order-submission, or fund-movement path.

## Architecture

The code is split into three packages:

- `oracle_bets_core`: CLI, configuration, paths, evidence, model registry,
  health, performance, backups, and read-only market clients;
- `lol_bets`: ingestion, identities, features, ratings, series reconstruction,
  training, calibration, inference, lifecycle, and workflow orchestration;
- `oracle_bets_discord`: bounded rendering, crash-safe Gateway delivery, and
  owner-only paper decision and manual settlement controls.

```text
Oracle's Elixir 2024–2026
  -> validated local cache
  -> history reconciliation and game quarantine
  -> chronological team/player features and ratings
  -> deterministic complete-series reconstruction
  -> independent research study or fixed-parameter refit
  -> calibrated direct-series candidate
  -> immutable registry and champion pointer

PandaScore schedule + owner-selected Polymarket event URLs
  -> conservative identity and roster resolution
  -> independent series-winner probability
  -> exact typed contract and token matching
  -> two public executable order-book observations
  -> deterministic paper gate
  -> JSON/Markdown report and optional Discord card
  -> owner re-quote, confirmation, and later manual settlement
  -> append-only performance evidence
```

Generated state has explicit ownership:

- `data/lol/`: replaceable source, interim, processed, and series data;
- `models/lol/`: replaceable training workspace;
- `data/state/oracle_bets.db`: canonical append-only operational evidence;
- `data/state/model-registry/lol/`: immutable reviewed model bundles;
- `reports/lol/`: deterministic review artifacts;
- `logs/lol/oracle-bets.log`: bounded rotating runtime log (10 MiB plus five
  backups by default).

The registry is not a duplicate model folder. The model workspace may be
rebuilt; the registry preserves the exact checksummed bundle that was reviewed
and served.

## Configuration and league scopes

`config/product/product.json` owns timezone, windows, league scopes, read-only
market policy, and retraining triggers. Historical training and ratings use
`research_all_supported`. Owner-facing prediction uses `tier1_plus_erls`, with
LCP and CBLOL omitted.

`config/lol/data_ingestion/team_aliases.json` separates provider aliases from
reviewed historical identity merges. Unknown identity or a similar name never
justifies fabricated history. `import_columns.json` is the source allowlist;
new CSV fields are candidates rather than automatic features.

Search priors live in
`config/lol/hyperparameters/default_models_parameters.json`. Reviewed
production parameters live under `hyperparameters/tuned/`. Routine training
requires the reviewed files and never starts Optuna. A research study stays
isolated until the owner reviews it and explicitly promotes its parameters.

Secrets belong only in the ignored mode-`0600` `.env`:

- `PANDASCORE_API_KEY`;
- `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID`, `DISCORD_OWNER_USER_ID`;
- `DISCORD_DELIVERY_MODE=gateway|off`;
- optional `OPENAI_API_KEY` and `OPENAI_MODEL`.

AWS and webhook configuration are obsolete. Polymarket public reads require no
credential or VPN.

## Data ingestion, identity, and features

Source refresh obtains the reviewed 2024–2026 public Oracle's Elixir files and
atomically replaces the cache only after ZIP, CSV, filename, size, stability,
schema, and freshness checks pass. The source is read-only; the pipeline never
modifies Google Drive.

Incremental ingestion replaces overlapping source rows by stable identity. Full
reconciliation rebuilds all retained history. Every accepted map must have two
team rows, ten player rows, Blue/Red composition, five roles per side, one
winner, valid identities, and no conflicting duplicate. Invalid maps enter an
explicit quarantine rather than being silently repaired.

Feature generation is chronological: each pre-match value is emitted before
the current result updates its state. Useful families include direct Elo,
Glicko, Plackett-Luce and TrueSkill strength/uncertainty, recent and long-term
form, inactivity, roster continuity, player-role form, patch/season context,
league strength, and current-opponent deltas.

Winner serving always rebuilds deltas from the two current team snapshots.
Persisted historical opponent deltas cannot survive inference assembly. Numeric
matchup features negate when teams swap; invariant context remains unchanged.
Side, first pick, draft, current-series results, post-start statistics, odds,
prices, and all other market data are forbidden winner inputs.

`reports/lol/ingestion/data_quality.json` records source coverage, accepted and
quarantined maps, schema changes, and the disposition of observed columns. A new
field is admitted only after availability, leakage, missingness, stability,
calibration, and temporal-ablation review.

## Winner V2 model lifecycle

The only actionable target is direct prematch series winner. Legacy map winner,
next-map winner, game length, total kills, and total towers remain diagnostic or
shadow research.

Historical maps are reconstructed into deterministic BO1/BO3/BO5 series. Each
accepted series produces one canonical row ordered by stable team ID using only
facts available before Map 1. Ambiguous, incomplete, contradictory, or
non-sequential groups are rejected with reasons.

Winner V2 predeclares three components:

1. a regularized rating-only logistic baseline;
2. a ten-member week-block LightGBM ensemble using eligible prematch features;
3. a convex logit blend selected on the calibration-selection partition.

Timestamp- and series-atomic partitions reserve development, calibration fit,
blend/calibration selection, uncertainty estimation, and an untouched final
test. Optuna may read only development rolling-origin folds. The final test can
accept or reject the predeclared choice; it cannot choose another component.

Inference canonicalizes the teams, predicts once, and complements for reversed
caller order. Exact `P(A beats B) + P(B beats A) == 1` is a serving invariant,
not an averaged repair. The conservative probability is an ensemble lower
quantile adjusted by held-out calibration bias; it is not a guarantee.

Promotion keeps structural safety separate from empirical model choice. Clean
provenance, a checksummed bundle, train/serve parity, zero forbidden or market
fields, exact symmetry, and no temporal or intra-series leakage are hard gates.
Aggregate log loss, Brier/ECE, bootstrap evidence, and the actionable betting
cohort are also hard performance gates. Calibration-intercept diagnostics and
small non-actionable league/region regressions are visible warnings during
paper testing: they no longer veto an otherwise superior model. Holdout reuse
is disclosed as a warning, never presented as fresh confirmation.

Among valid candidates, the system promotes the best supported aggregate and
actionable probability model; recency breaks ties but cannot excuse a material
regression. Shadow prop quality cannot block a better series-winner model.
Reviewed map/prop parameters may be reused across a changed input-column schema
only because those targets are non-actionable; the training manifest and logs
flag that compatibility exception. Direct series and next-map targets retain an
exact feature-schema parameter contract.

Routine retraining and retuning are different:

- retraining refits weights and calibrators with already reviewed parameters,
  registers a candidate, and may auto-promote only a non-inferior routine bundle;
- retuning searches parameters with Optuna, carries extra overfitting risk, and
  never changes reviewed parameters or the champion automatically.

Series and next-map studies must run independently. The normal bundle contains
direct series winner, diagnostic map winner, and three shadow scalar props.
Next-map is an explicit experimental target and is not required for normal
training or promotion. The first Winner V2 champion and every Optuna-derived
candidate require explicit owner promotion.

Training reports include metrics, reliability, cohorts, feature lineage,
availability/missingness, league coverage, gain/split importance, held-out
permutation importance, SHAP attribution, drift evidence, model cards, data/code
fingerprints, and calibration artifacts. Importance is diagnostic, not causal.

## Market review and paper policy

The model is permanently independent from Polymarket. Market information may
identify a contract and measure executable odds, liquidity, edge, timing, and
later closing-line value; it never modifies the probability.

The preferred owner workflow is exact-link review. `lol market-review` accepts
canonical `https://polymarket.com/...` LoL event URLs, loads each exact Gamma
event by slug, derives its teams/start/BO from the series moneyline, optionally
enriches roster identity from the saved PandaScore schedule, and refuses fuzzy
fallbacks. Only typed series-winner contracts with compatible competition,
teams, start, BO, status, outcome orientation, and public resolution source may
proceed.

Each token is quoted independently. The client captures two books with one
shared wait, uses the larger reported minimum share size, walks complete ask
depth, and keeps the worse executable average odds. Missing, malformed, stale,
crossed, empty, or insufficient books block only that outcome. Gamma display
prices are never substituted for executable quotes.

Only a prematch series-winner selection may become paper-actionable. It must:

- be the independent model's point favorite at 57.5% or higher;
- produce at least 5% expected edge using its conservative probability and the
  worse executable quote;
- start 24–48 hours later;
- have healthy champion, identity, roster, attribution, liquidity, timestamp,
  contract, resolution, and token evidence.

A model/market gap of at least 20 percentage points, or a full-model favorite
that the rating baseline prices at 45% or below, is quarantined rather than
treated as a dream upset. Polymarket favorite status is irrelevant. Paper stake
is a flat one unit; quarter-Kelly is counterfactual evidence only. There is no
arbitrary minimum market-odds gate: a short price still has to clear the same
model-favorite, uncertainty, timing, anomaly, and 5% conservative-edge rules.

The original 52.5% point gate was raised after the sealed V2 holdout showed
sub-50% favorite accuracy in both the 52.5–55% and 55–57.5% actionable bands.
The conservative bound may still cross 50%, so a close independent-model edge
is not rejected merely because uncertainty spans an even matchup.

Every exact-link review returns the independent direct-series price, a separate
Map 1 research price, and game-length/kills/towers point means. Props need an
explicit line before an over/under probability can be calculated. Only the
series result can become actionable.

`--publish` queues one compact Discord review with its Markdown report attached.
If every series gate passes, the system records a separate interactive proposal.
Accept begins a two-phase process: fetch two fresh books, rerun every gate, then
show a Confirm/Cancel control valid for 120 seconds. The confirmed quote becomes
the immutable paper entry. The bot uses durable send intent, deterministic
markers, bounded history recovery, owner checks, and a single-instance lock.

Positions close only through owner-confirmed `win`, `loss`, `push`, or `void`
with a non-empty source reference. Identical repeats are idempotent; conflicts
fail visibly. There is no automatic reconciliation or settlement. LLM output
may summarize an existing deterministic result but cannot alter probabilities,
gates, stake, promotion, acceptance, or settlement.

## Evidence, reports, and review cadence

The evidence database is append-only across runs, fixtures, identities, model
versions, predictions, market observations, proposals, positions, settlements,
corrections, and audit events. No-bet and rejected decisions are valuable
evidence.

Probability evidence comes first: temporal log loss, Brier, ECE, reliability,
slope/intercept, and meaningful cohorts. Profit evidence adds executable entry
odds, CLV, turnover, ROI with uncertainty, drawdown, hit rate, and performance by
league/edge band. The strategy is not proven by a small sample or a theoretical
edge.

Production Python writes deterministic JSON/Markdown and training artifacts.
Notebooks are read-only views for ingestion quality, calibration, attribution,
and paper performance. The pipeline never executes notebooks or depends on
their state, and committed outputs stay cleared.

The owner refreshes history and reviews schedules/links when preparing bets.
Model research is on demand, not daily. Review paper performance weekly and
monthly. The Gateway bot may stay online through the one retained launchd
example; daily workflows and studies remain explicit owner actions.

## Verification and hosting

Every code change must pass the repository tests, Ruff, Ty, and strict MkDocs.
Schema/model changes additionally require data validation, series rebuild,
winner health, exact symmetry, and leakage checks. Do not run a real Optuna
study as a smoke test: it is expensive and permanently exposes its final
holdout.

The Gateway bot needs an always-on Mac, VPS, or VM. A one-way scheduled report
can run on a free/near-free scheduler, but interactive owner controls need the
Gateway process or a public interactions endpoint. Oracle Cloud Always Free is
an option with capacity, verification, reclamation, maintenance, regional, and
outbound-network caveats.

See [Command runbook](commands.md) for exact operations and [Roadmap](roadmap.md)
for the current evidence gates.

## Repository audit and simplification boundary

The August 2026 audit found a sound safety architecture and no market-derived
model feature or trading surface. The main risk is maintainability, not missing
layers: model training, daily orchestration, model operations, market matching,
and the Gateway bot contain several large multi-purpose functions. The test
suite covers the repository at roughly 67% overall, with materially weaker
coverage in Discord interaction recovery, live match inference, and several
rating implementations.

Safe simplifications completed in the current product cutover remove duplicated
directory documentation, unused generic prediction contracts, an unintegrated
outcome recorder, and the superseded map-derived series calculator. Matchup
assembly is shared by series and Map 1 inference, unused serving state was
removed, and the experimental next-map target was removed from normal training
and registry requirements. The owner-facing documentation is intentionally
limited to this design, the command runbook, and the roadmap.

The remaining cleanup is deliberately staged:

1. raise focused coverage for Discord recovery, match inference, and ratings;
2. split the largest orchestration functions along existing data/model/evidence
   boundaries without changing public commands or artifacts;
3. deduplicate the rating-tuning research utilities only after benchmark tests
   establish equivalent results;
4. delete code only when static search, coverage, and artifact compatibility
   prove it has no caller.

This is not a claim that the code is bug-free or that the strategy is
profitable. It is the minimum-risk route to a maintainable paper-testing system.
