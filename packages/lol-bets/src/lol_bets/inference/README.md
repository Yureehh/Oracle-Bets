# inference

LoL prediction-time helpers.

- `team.py`: loads latest team/player state from flattened artifacts.
- `match_predictor.py`: assembles inference features and calls trained models.
- `team_resolver.py`: conservative provider-to-training identity resolution.
- `roster.py`: expected-lineup action gate.
- `series.py`: best-of probabilities derived from the map engine.

Example:

```python
from lol_bets.inference.match_predictor import MatchPredictor
from lol_bets.inference.team import Team
```
