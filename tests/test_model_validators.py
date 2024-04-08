import pandas as pd

# Import the necessary classes
from src.model_validators import (
    MixedValidator,
    PlayerEgpmDominanceValidator,
    PlayerEloValidator,
    PlayerEnsembleValidator,
    PlayerPlackettLuceValidator,
    PlayerSideEmaValidator,
    TeamEgpmDominanceValidator,
    TeamEloValidator,
    TeamEnsembleValidator,
    TeamPlackettLuceValidator,
    TeamSideEmaValidator,
    TrueSkillValidator,
)
from utils.paths import PROCESSED_DIR

# Load test data
team_data = pd.read_csv(PROCESSED_DIR / "team_data.csv")
player_data = pd.read_csv(PROCESSED_DIR / "player_data.csv")


class TestModelValidators:
    def test_team_elo_validator(self):
        validator = TeamEloValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_player_elo_validator(self):
        validator = PlayerEloValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_trueskill_validator(self):
        validator = TrueSkillValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_team_plackett_luce_validator(self):
        validator = TeamPlackettLuceValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_player_plackett_luce_validator(self):
        validator = PlayerPlackettLuceValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_team_egpm_dominance_validator(self):
        validator = TeamEgpmDominanceValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_player_egpm_dominance_validator(self):
        validator = PlayerEgpmDominanceValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_team_side_ema_validator(self):
        validator = TeamSideEmaValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_player_side_ema_validator(self):
        validator = PlayerSideEmaValidator(team_data, player_data)
        accuracy, logloss, brier = validator.validate()
        assert isinstance(accuracy, float)
        assert isinstance(logloss, float)
        assert isinstance(brier, float)

    def test_team_ensemble_validator(self):
        validator = TeamEnsembleValidator(team_data, player_data)
        ensemble_metrics, majority_voting_metrics = validator.validate()
        assert isinstance(ensemble_metrics, list)
        assert isinstance(majority_voting_metrics, list)
        assert len(ensemble_metrics) == 3
        assert len(majority_voting_metrics) == 3

    def test_player_ensemble_validator(self):
        validator = PlayerEnsembleValidator(team_data, player_data)
        ensemble_metrics, majority_voting_metrics = validator.validate()
        assert isinstance(ensemble_metrics, list)
        assert isinstance(majority_voting_metrics, list)
        assert len(ensemble_metrics) == 3
        assert len(majority_voting_metrics) == 3

    def test_mixed_validator(self):
        validator = MixedValidator(team_data, player_data)
        ensemble_metrics, majority_voting_metrics = validator.validate()
        assert isinstance(ensemble_metrics, list)
        assert isinstance(majority_voting_metrics, list)
        assert len(ensemble_metrics) == 3
        assert len(majority_voting_metrics) == 3
        weights = validator.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) == 9
