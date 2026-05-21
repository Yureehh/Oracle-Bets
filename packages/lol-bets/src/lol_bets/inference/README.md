# inference

LoL prediction-time helpers.

- `team.py`: loads latest team/player state from flattened artifacts.
- `match_predictor.py`: assembles inference features and calls trained models.

Example:

```python
from lol_bets.inference.match_predictor import MatchPredictor
from lol_bets.inference.team import Team
```
