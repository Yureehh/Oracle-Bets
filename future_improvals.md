# FUTURE IMPROVEMENTS

## Feature engineering
- Draft-phase signals (if picks/bans available).
- Stage/series context: BO1/BO3/BO5, game number in series, playoff vs regular.
- Interaction features: elo_delta * winrate_l5, trueskill_mu_delta * patch_winrate, league_elo_delta * first_baron%.
- Opponent-weighted EWMs so farming weak teams doesn’t inflate form.
- Patch/meta drift: days_since_patch, games_since_patch, patch_winrate_delta.
- Uncertainty features: glicko_rd_delta, trueskill_sigma_delta, plus lane max/sum for volatility.
- Lineup continuity: %same_lineup_last_N or lineup_changes_last_5.
- Side baseline priors: blue_winrate_league_patch (EWM).

## Feature governance
- Use only pre-match features (no post-game stats).
- Standardize delta naming: kda_delta, gd15_delta, first_baron_delta, etc.
- Keep raw rating win_likelihoods and their deltas (models like both).

## Modeling & training (GBDT/XGBoost)
- Train on one row per game (blue only) when using deltas; infer red via sign flip.
- Monotonic constraints for rating deltas and *_win_likelihood_delta (+1), uncertainty deltas often (-1).
- Temporal CV (rolling origin) + early stopping; logloss metric; tree_method="hist"; depth 4–7; L1/L2; subsample/colsample.
- Calibration (isotonic/Platt) on out-of-fold predictions; apply to test/live.

## Evaluation & monitoring
- Profit-aware evaluation: expected value vs implied odds (margin removed), betting ROC/profit curve.
- Drift monitoring: PSI across patches/leagues for key features; retrain triggers.

## Data & process
- Re-do notebooks with new data/features.
- LCK/LPL group split: skip betting on cross-group matchups (exclude or flag in dataset).
