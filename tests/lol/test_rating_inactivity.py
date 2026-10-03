from glicko2 import Rating
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    apply_inactivity_decay as apply_elo_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    run_elo_computation,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    apply_inactivity_decay as apply_glicko_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    handle_new_entity,
    run_glicko2_computation,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    run_pl_computation,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    run_trueskill_computation,
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


def _same_date_team_games() -> pd.DataFrame:
    rows = []
    for gameid, hour, winner in (("g1", 10, "a"), ("g2", 18, "b")):
        for side, teamid in (("Blue", "a"), ("Red", "b")):
            rows.append(
                {
                    "season": 2026,
                    "date": pd.Timestamp(f"2026-01-01 {hour}:00"),
                    "gameid": gameid,
                    "teamid": teamid,
                    "league": "LCK",
                    "side": side,
                    "result": int(teamid == winner),
                }
            )
    return pd.DataFrame(rows)


def test_all_rating_families_freeze_inputs_across_same_date_games():
    frame = _same_date_team_games()
    outputs = (
        (
            run_elo_computation(
                frame,
                "team",
                1500.0,
                32.0,
                0.9,
                400.0,
                0.5,
                0.5,
                0.2,
                {},
                show_progress=False,
            ),
            "elo_before",
            "elo_win_likelihood",
        ),
        (
            run_glicko2_computation(
                frame,
                "team",
                1500.0,
                350.0,
                0.06,
                0.9,
                0.5,
                0.5,
                0.2,
                {},
                show_progress=False,
            ),
            "glicko2_mu_before",
            "glicko2_win_likelihood",
        ),
        (
            run_pl_computation(
                frame,
                "team",
                25.0,
                8.333,
                0.9,
                0.5,
                0.5,
                0.2,
                {},
                show_progress=False,
            ),
            "pl_mu_before",
            "pl_win_likelihood",
        ),
        (
            run_trueskill_computation(
                frame,
                "team",
                25.0,
                8.333,
                4.167,
                0.9,
                0.5,
                0.5,
                0.2,
                {},
                show_progress=False,
            ),
            "trueskill_mu_before",
            "trueskill_win_likelihood",
        ),
    )

    for output, rating_column, probability_column in outputs:
        team_a = output.loc[output["teamid"] == "a"]
        assert team_a[rating_column].nunique() == 1
        assert team_a[probability_column].nunique() == 1
