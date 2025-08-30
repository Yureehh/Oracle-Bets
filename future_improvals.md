# FEATURE ENGINEERING

Draft-phase signals (if you have picks/bans)

League & tournament context (Stage/series features: BO1 vs BO5, game number in series)

Interaction features that are cheap but powerful (elo_delta * winrate_l5, trueskill_mu_delta * patch_winrate, league_elo_delta * first_baron%)

Target shaping & modeling tips (Calibration (isotonic or Platt) on a time-held fold — huge for betting decisions. Temporal CV (rolling origin) in Optuna for more stable params.)


# FOR XGBoost:
Train on one row per game (blue only)
Since you use deltas, keeping only the Blue row avoids duplicating labels and enforces p_red = 1 - p_blue at inference (you can compute the red prob by flipping signs).

Monotonic constraints (big win)
In XGBoost set monotone constraints for features that must move the probability in one direction, e.g.

elo_delta, glicko_mu_delta, trueskill_mu_delta, league_elo_delta, *_win_likelihood_delta → +1

uncertainty_deltas (sigma/RD) often −1 if you believe more uncertainty should reduce win odds.
This improves stability, interpretability, and calibration.

Temporal CV + early stopping
Use rolling-origin splits for tuning; logloss as the metric; tree_method="hist", modest depth (e.g., 4–7), min_child_weight, subsample, colsample_bytree, and L1/L2 regularization.

Calibration on out-of-fold predictions
After tuning, fit isotonic (or Platt) on OOF predictions from the temporal CV. Apply that calibrator to test/live. This materially improves betting decisions.

Feature governance

Use only pre-match features (no post-game stats).

Standardize column naming for deltas: kda_delta, gd15_delta, first_baron_delta, etc.

Keep raw rating win_likelihoods AND their deltas—models like both.

# Extra suggestions most teams miss

Opponent-weighted EWMs: weight your rolling stats by opponent strength (opp_league_elo_before or opp_trueskill_mu_before) so farming weak teams doesn’t inflate form.

Patch/meta drift controls: add days_since_patch, games_since_patch, patch_winrate_delta, and consider patch-specific calibration if distribution shifts are big.

Uncertainty as a feature: include glicko_rd_delta, trueskill_sigma_delta, maybe the sum and max uncertainties across lanes (for players) — volatility predicts upsets.

Lineup continuity (you said you avoid non-starters, good): still add %same_lineup_last_N or simple lineup_changes_last_5 as a stability proxy.

Side baseline priors: include blue_winrate_league_patch (EWM). The model will learn it, but an explicit prior helps across small samples.

Drift monitoring: track population stability (PSI) across patches/leagues for key features and rating deltas so you know when to retune.

Profit-aware evaluation: besides logloss & calibration curves, track backtested expected value vs. implied odds (with margin removed) and Betting ROC (profit curve).

# TODOs
- Add code documentation with MKDocs
- Re-do notebooks with new data and new features
