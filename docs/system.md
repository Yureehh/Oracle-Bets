# System design and operating decisions

Oracle Bets is a local-first League of Legends betting-research system. Its
probabilities are independent of Polymarket. Market data identifies contracts,
quotes executable prices, and measures later CLV; it never enters a model.
There is no betting, signing, wallet, private-key, order, or fund-movement code.

## End-to-end flow

```text
Oracle's Elixir 2024–2026
  -> source validation and history reconciliation
  -> chronological ratings/features
  -> complete historical series
  -> fixed-parameter training or explicit Optuna research
  -> calibrated, symmetric model bundle
  -> immutable candidate review and champion promotion

PandaScore schedule
  -> owner chooses 1–2 Polymarket/Thunderpick links for one fixture
  -> exact event/team/start/BO/roster resolution
  -> independent series, map, and prop forecasts
  -> compact contract inventory
  -> one current public CLOB book per supported Polymarket outcome
  -> manual Thunderpick lines
  -> exact-semantic provider comparison, warnings, and best odds
  -> one JSON/Markdown report pair
  -> owner-only Discord review and confirmed paper/real ledger entry
  -> manual settlement and separated performance evidence
```

Daily ingestion, health, optional fixed-parameter retraining, schedule, and
reporting remain available. Daily runs do not discover Polymarket events or
create market reviews or bets. Exact owner-supplied links are the only market
review source.

## Package and state ownership

- `oracle_bets_core`: CLI, product configuration, evidence, registry, public
  market reads, unified bet ledger, health, backup, and performance.
- `lol_bets`: ingestion, identity, ratings, features, series construction,
  training, calibration, inference, exact-link interpretation, reports.
- `oracle_bets_discord`: Gateway lifecycle, `/oracle` owner console, manual
  review/bet/settlement controls, and in-memory charts.

Generated state is intentionally separate:

- `data/lol/`: replaceable source/interim/processed datasets;
- `models/lol/`: replaceable training workspace;
- `data/state/oracle_bets.db`: canonical append-only operational evidence;
- `data/state/model-registry/lol/`: immutable reviewed serving bundles;
- `reports/lol/`: generated review/training/health reports;
- `logs/lol/pipeline.log`: ingestion, feature, rating, and training operations;
- `logs/lol/schedule.log`: PandaScore schedule/lineup operations;
- `logs/lol/discord.log`: Gateway interaction and rendering failures.

Each persistent log rotates at 5 MiB with three backups by default, so normal
operation cannot grow without bound. `ORACLE_BETS_LOG_MAX_BYTES` and
`ORACLE_BETS_LOG_BACKUPS` override those limits. General interactive messages
remain console-only. Logs are disposable diagnostics, not evidence.

The registry is not a duplicate of `models/lol`: the workspace can be rebuilt,
while the registry preserves the exact checksummed bundle that was reviewed.
`models/lol/.staging` is intentionally used for crash-safe candidate assembly
and should be empty when training is not running. `site/` is only generated
MkDocs output. `.hypothesis/` is only a property-test cache. `ops/launchd/`
contains the optional macOS service template for the always-on Discord bot.

`data/state/` must survive normal cleanup. It owns the append-only SQLite paper
ledger, consistent backups/exports, Discord publication state, and immutable
model registry. By contrast, `data/lol/` is a large replaceable data build tree;
it is not expected to be empty after ingestion.

## Data and model contract

Training uses `research_all_supported`. Owner-facing systematic review uses
`tier1_plus_erls` minus LCP and CBLOL. Those two leagues remain training-only.

Source refresh validates filenames, content, schema, size, stability, and
freshness before replacing the local cache. Accepted maps require two team
rows, ten player rows, five roles per side, one winner, and stable identity.
Invalid maps enter an explicit quarantine.

Features are emitted before the current result updates state. Eligible families
include direct Elo/Glicko/Plackett-Luce/TrueSkill strength and uncertainty,
recent and long form, inactivity, roster continuity, player-role form, patch,
season, and league context. Winner inference rebuilds current-opponent deltas;
historical opponent deltas cannot survive serving assembly.

Winner inputs forbid side, first pick, draft, current-series results, post-start
statistics, odds, prices, and every market-derived field. Team swap negates
numeric matchup deltas and leaves invariant context unchanged. Inference
canonicalizes once and complements the result, so order is exactly symmetric.

## Forecasts and probability sources

- Series winner: direct calibrated Winner V2 model (`series_direct_v2`).
- Game N winner: prematch map model (`map_prematch_v1`). The same pre-draft
  probability is used for each map; this is experimental, not in-play advice.
- Series totals: legal BO3/BO5 map paths (`series_totals_map_path_v1`).
- Series/map handicap: final map-differential distribution from the same legal
  paths (`series_handicap_map_path_v1`).
- Length, kills, towers: calibrated regression artifacts, display/record only.
- Baron, dragon, inhibitor, first blood, multikill, odd/even, or unknown:
  inventoried without book fan-out and marked `no_model_target`.

Legal paths stop as soon as either team reaches two BO3 wins or three BO5 wins.
Exact score, total maps, and final map differential sum to one. Swapping teams
complements winner/handicap probabilities; totals and scalar props are invariant.
Derived probabilities are labeled and their disagreement with the direct series
model is recorded.

## Roster evidence

Resolution order is:

1. complete PandaScore role-mapped lineup;
2. most frequent player per role in the last ten historical maps strictly before
   fixture start, with ties broken by latest appearance;
3. existing missing-value inference with `unknown` roster evidence.

Historical evidence stores map IDs/dates, appearance shares, alternates,
substitutions, confidence (`high|medium|low|unknown`), and a hash. Roster
confidence is visible evidence and a warning, not a proposal hard gate.

## Exact-link review and quotes

`lol market-review` and Discord **Review Markets** accept one or two comma-,
whitespace-, or newline-separated Polymarket/Thunderpick URLs, deduplicate in
order, require one fixture, and isolate failures per provider. There is no fuzzy
Polymarket event fallback and no Thunderpick HTTP client.

Every returned contract is recorded. Supported meaning is parsed from
`sportsMarketType`, question/group title, game number, line, outcomes, teams,
start/BO, token orientation, and resolution rules. Ambiguity stays visible and
cannot become systematic evidence.

Review fetches one public CLOB book per compatible Polymarket outcome, with
bounded concurrency and request timeouts. It walks asks at the provider's
minimum order size and converts average share cost to decimal odds. Missing or
old timestamps are warnings; malformed, empty, crossed, or insufficient books
remain outcome-local failures. Unsupported specials do not trigger book reads.

Thunderpick is deliberately manual. Its link identifies the provider/fixture;
the owner enters the visible target, selection, line, and decimal odds. Provider
rows are compared only when target, period, selection, line, and settlement
meaning match exactly. Prices never enter inference.

## Transparent comparison and manual control

There are no automatic proposals or eligibility thresholds. Each comparison
shows model probability, fair odds, provider odds, implied probability, point
EV, and the best equivalent provider. Positive EV is evidence, not permission.
Warnings and negative EV remain visible, and the owner may still record the
decision with a rationale.

The active append-only schema has `bets` for immutable paper/real entry terms
and `bet_events` for lifecycle/settlement events. Legacy proposal, approval,
paper-position, and settlement tables remain readable/exportable but receive no
new writes. Settlement is manual `win|loss|push|void` with a source reference;
identical repeats are idempotent and conflicts fail visibly.

Paper and real evidence never combine. Monetary totals are grouped by currency,
and markets without model probabilities contribute PnL but not calibration.

## Model lifecycle

Routine retraining refits weights/calibrators with reviewed fixed parameters. It
does not run Optuna. Retuning is an explicit research search, never changes the
champion automatically, and requires owner review before parameter/model
promotion. Structural safety—clean provenance, checksums, train/serve parity,
symmetry, forbidden fields, and leakage—never relaxes for recency.

Maps/totals/handicaps stay experimental during paper testing. The validation
command checks sealed Map-1/later/actionable/probability-band evidence,
forbidden features, symmetry/path algebra, reconstructed BO history, and
held-out totals/handicap scores derived only from each series' Map-1 price. It
does not claim market-profit evidence.

## Discord and future draft model

`/oracle` is an owner-only hub. Its first row is Review Markets, Record Bet,
Open Bets, and Closed Bets; its second row is Schedule, Performance, and Health.
Review work defers immediately and edits that same private interaction when
complete—there is no publisher queue or polling delay. Bet recording has an
explicit confirmation screen, real mode states that it records an already
manually placed bet, and settlements remain owner-only. Charts are rendered in
memory. Ingestion, training, retuning, and promotion remain CLI-only.

Future draft integration requires timestamped champions/bans/sides/patch/player
identities and a source reference. It will be a separate symmetric draft-aware
map model and a separate `draft_assisted` evidence lane.
