# LoL Feature Engineering 2026 Audit

This note records the feature/rating decisions for the 2026 ruleset pass. The
goal is to keep only stable pre-match signal, remove patch artifacts, and make
future experiments easy to compare.

## Ruleset Alignment

- Atakhan, Blood Roses, and Feats of Strength are removed from current gameplay,
  so Atakhan columns should not be ingested, engineered, trained, flattened, or
  surfaced in Discord.
- Oracle's Elixir exposes 15-minute and 25-minute checkpoint columns in the
  current local schema. It does not expose 14-minute checkpoint columns, so 15 is
  the closest reliable post-plates mid-game marker.
- Void Grubs now have one spawn cycle. Keep `void_grubs` as an early-objective
  count, but treat it as a capped 0-3 objective family in analysis and avoid any
  two-spawn assumptions.
- The 25-minute checkpoint is useful but missing for short games. It should stay
  in the full and compact configs for now, then survive only if walk-forward
  calibration improves.

## Changes Made In This Pass

- Removed `atakhans` from ingestion columns.
- Removed all Atakhan EMA features from training and flattened configs.
- Removed 10-minute and 20-minute checkpoint features from ingestion, training,
  and flattened configs.
- Kept checkpoint features centered on 15 and 25 minutes.
- Added a regression test that blocks Atakhan, `_std`, `at10`, and `at20`
  feature families from config files.

## Feature Families

### Keep

- Ratings: Elo, Glicko2, TrueSkill, PL, league Elo, and side/season likelihoods.
- Team macro form: game length, team KPM, EGPM, towers, dragons,
  barons, heralds, grubs, inhibitors, elders.
- Mid-game lane/economy checkpoints: gold, XP, CS, and their differentials at
  15 minutes.
- Late-game checkpoint signal: the same families at 25 minutes, but only if
  missingness and calibration remain acceptable.
- Series context: game number, BO1/BO3/BO5, deciding game.
- Rest context: first season game and post-break indicator.
- Head-to-head: keep with prior-game count gating; never trust sparse H2H alone.

### Drop Or Gate

- `_std` features: do not add them as model inputs. They are high-dimensional
  noise unless a future experiment proves otherwise.
- Atakhan features: drop completely for current/future models.
- 10-minute checkpoints: likely too early and highly correlated with stronger
  15-minute deltas.
- 20-minute checkpoints: redundant with 15 and 25 while adding another
  missingness boundary.
- Outcome-adjacent summary economy aggregates are removed from configs until
  ablations prove they add calibrated signal.
- Player pentakills/doublekills are very sparse. Keep only if feature importance
  and ablations justify them.
- Player damage-to-towers is role/meta sensitive and can be lineup noisy.
  Ablate against compact player features.

### Add Next

- Roster continuity: percentage of expected starters retained from prior match
  and prior split. This is one of the most LoL-specific missing signals.
- Days since last game for players, not only teams.
- Patch age: number of days since a patch first appeared in the dataset, because
  early-patch volatility is real.
- League-stage context: regular season, playoffs, international, promotion, and
  academy/development split, derived from league taxonomy and playoffs flag.
- Recent form windows by entity: last 3 and last 5 games for team economy and
  objective deltas, compared against the existing EMA.
- Side-adjusted team strength: team form split by Blue/Red with minimum games.
- Opponent-adjusted recent form: recent performance relative to opponent rating,
  not just raw EMA.
- Champion and draft features only if reliable pre-match draft data becomes
  available. Do not infer unavailable draft signal from post-game stats.

## Rating Audit

### Elo

The implementation has useful lifecycle hooks: seasonal decay, league-transfer
adjustments, new-entity priors, and position-switch resets. The biggest issue is
not architecture, it is tuning discipline.

Recommended fixes:

- Tune team Elo and player Elo separately with walk-forward folds.
- Tune `k_factor`, `elo_divisor`, seasonal decay, league-transfer factor,
  initial league adjustment, and position reset.
- Add a separate inactivity decay based on days since last official game.
- Treat role swaps differently from emergency substitutes. A player moving from
  bot to mid should reset more than top to jungle, but less than a brand-new
  player.
- Make player aggregation role-aware but tunable. Current fixed weights
  overweight mid/bot slightly; test equal weights, current weights, and learned
  role weights.

### Glicko2

Glicko2 is valuable if uncertainty is meaningful. Current defaults are generic.

Recommended fixes:

- Tune initial `mu`, `phi`, `sigma`, and `tau`.
- Increase uncertainty after long inactivity or a season break.
- Persist and expose uncertainty as a confidence warning in Discord, not only as
  a model feature.
- Compare mean aggregation against role-weighted aggregation for player teams.

### TrueSkill And Plackett-Luce

These can be strong lineup priors, but they likely overlap. Keep both in the
full model, then require ablation proof before putting both in compact.

Recommended fixes:

- Tune `mu`, `sigma`, `beta`, and `tau` out-of-time.
- Keep draw probability at zero for single-game winner markets.
- Test PL as a player-only ranker and TrueSkill as the primary team/lineup
  uncertainty model.

## Experiment Plan

Run all experiments with identical walk-forward splits and report log loss,
Brier, AUC, calibration error, league cohorts, patch cohorts, and ROI-style edge
simulation.

1. `full_2026_clean`: current full config after Atakhan and 10/20 removal.
2. `compact_2026_clean`: compact config with 15/25 deltas.
3. `rating_only`: all ratings and league/side/season likelihoods only.
4. `team_only`: team features plus ratings, no player pivot.
5. `player_only`: player form plus ratings, no team macro except metadata.
6. `no_25`: remove 25-minute checkpoint features.
7. `no_sparse_player`: remove pentakills, doublekills, tower damage, and
   damage-mitigated features.
8. `no_h2h`: remove head-to-head.
9. `elo_only_vs_all_ratings`: test whether Glicko2, PL, and TrueSkill add signal
   beyond Elo.

The production candidate should be the simplest feature set that improves
calibrated out-of-time probability quality. Profitability comes from probability
calibration and market disagreement, not from a giant feature table.
