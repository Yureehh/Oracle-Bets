from glicko2 import Rating
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    apply_inactivity_decay as apply_elo_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    apply_inactivity_decay as apply_glicko_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    handle_new_entity,
)
from oracle_bets_core.pd import pd

EXPECTED_DECAYED_STRENGTH = 1550.0
EXPECTED_DECAYED_PHI = 225.0
EXPECTED_SIGMA = 0.06
TUNED_PHI = 500.0
TUNED_SIGMA = 0.08


def test_rating_inactivity_uses_grace_plus_half_life():
    last_active = pd.Timestamp("2026-01-01")
    as_of = pd.Timestamp("2026-08-14")  # 45-day grace + 180-day half-life
    elo = {"elo": 1600.0, "last_active": last_active}
    glicko = {
        "rating": Rating(mu=1600.0, phi=100.0, sigma=EXPECTED_SIGMA),
        "last_active": last_active,
    }

    apply_elo_inactivity_decay(elo, as_of, baseline_elo=1500.0)
    apply_glicko_inactivity_decay(glicko, as_of, baseline_mu=1500.0, baseline_phi=350.0)

    assert elo["elo"] == EXPECTED_DECAYED_STRENGTH
    assert glicko["rating"].mu == EXPECTED_DECAYED_STRENGTH
    assert glicko["rating"].phi == EXPECTED_DECAYED_PHI
    assert glicko["rating"].sigma == EXPECTED_SIGMA


def test_new_glicko_entity_uses_tuned_uncertainty_parameters():
    ratings = {}

    handle_new_entity(
        "team-1",
        ratings,
        {"LCK": 1500.0},
        "LCK",
        2026,
        1500.0,
        TUNED_PHI,
        TUNED_SIGMA,
        1.0,
    )

    rating = ratings["team-1"]["rating"]
    assert rating.phi == TUNED_PHI
    assert rating.sigma == TUNED_SIGMA
