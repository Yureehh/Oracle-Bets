# Predictions and markets

The actionable model is Winner V2: one independent, calibrated probability for
the **prematch series winner**. It is trained directly on complete historical
series, not inferred from a map model. Canonical team ordering plus complement
logic guarantees that swapping the caller's team order changes `p` to exactly
`1-p`.

Winner V2 combines three predeclared components:

1. a regularized logistic baseline using direct Elo, Glicko, Plackett-Luce, and
   TrueSkill likelihoods, with its own pre-final calibration so sealed
   comparisons are fair;
2. a ten-member week-block LightGBM ensemble using prematch form, rating,
   roster, player, patch, inactivity, and league-strength features;
3. a convex logit blend selected only on the calibration-selection partition.

The final 15% temporal holdout accepts or rejects that choice; it never chooses
a different component. Calibration and uncertainty have separate partitions.
The conservative probability is the calibrated member 10th percentile adjusted
by held-out one-sided calibration bias. It is a risk bound, not a guarantee.

Side, draft, first pick, current-series outcomes, post-start statistics,
Polymarket, bookmaker odds, prices, and other market data are forbidden model
inputs. The market can disagree with the model; it never changes the model
probability.

## Paper action policy

Only prematch series-winner contracts can be actionable. The selected team must
be the independent model's point favorite at 52.5% or higher. Its conservative
probability may cross 50%, but it must still produce at least 5% expected edge
at the worse of two executable quotes. The fixture must begin 24–48 hours later,
and model, roster, identity, liquidity, timestamps, token orientation, and
resolution rules must all be healthy.
For LoL, the resolution source must match Polymarket's published LoL sports
metadata (`liquipedia.net/leagueoflegends`); a generic or unrelated URL is
rejected even when the contract title looks plausible.

The proposal is quarantined when model and market differ by 20 percentage
points or more, when the full model selects a team that the rating baseline puts
at 45% or lower, or when unavailable/unstable features dominate attribution.
Polymarket favorite status is irrelevant.

Accepted evidence uses a flat one-unit paper position. Quarter-Kelly is stored
only as a counterfactual. Clicking Accept does not open a position: it captures
two new read-only books and reruns every gate; the owner must separately Confirm
within 120 seconds. The confirmed quote becomes the immutable entry price.

Props, totals, and prematch individual-map forecasts remain shadow research and
do not appear as actionable cards. The experimental next-map model uses frozen
prematch features plus sequential, conflict-checked owner-recorded series score
and map number. Its inference interface is explicitly `shadow_only`; it remains
shadow-only when a prematch position exists and never chases losses.

The hourly market observer records first-seen and later public price/book
snapshots around 14d, 7d, 72h, 48h, 36h, 24h, 12h, 6h, 1h, and close. The
48–24-hour entry window remains provisional until sealed CLV evidence supports a
change. No code places bets, signs payloads, accesses wallets, or moves funds.
