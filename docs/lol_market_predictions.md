# LoL Market Predictions

Oracle Bets v1 supports four LoL markets:

- **Winner**: one-map win probability, plus BO1/BO2/BO3/BO5 series math.
- **Game length**: expected map duration in minutes.
- **Total kills**: expected combined champion kills.
- **Total towers**: expected combined towers destroyed.

Volatile props such as first blood, penta kills, quadra kills, both teams Baron,
both teams dragon, inhibitors, and odd/even kills are intentionally excluded for
now. They are either too sparse, too path-dependent, or too sensitive to live
game state for this first pre-match model suite.

## Training

Train every supported model:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets all
```

Train only the winner model:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets outcome
```

Train only prop models:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets props
```

Train selected prop models:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets total_kills,total_towers
```

Generate a feature-selection recommendation report while training:

```bash
uv run oracle-bets lol train --model-type lightgbm --targets all --feature-selection report
```

Artifacts are stored under `models/lol/<ModelName>_LightGBM/`. Each trained model
stores the fitted model, final feature list, feature pipeline, categorical
features, metrics, model card, and validation predictions. The winner model also
stores a validation-fitted probability calibrator when the validation split is
large enough. Each prop model stores a residual summary used to price lines.

## How To Use The Bot

Show the LoL command help:

```text
!lol
```

Winner prediction:

```text
!lol predict "Team WE" "LNG Esports"
```

Winner prediction with known side and first pick:

```text
!lol predict "Team WE" "LNG Esports" --side Blue --first-pick "Team WE"
```

BO3 or BO5 series prediction:

```text
!lol predict "Team WE" "LNG Esports" --bo3
!lol predict "Team WE" "LNG Esports" --bo5
```

Raw prop projections:

```text
!lol props "Team WE" "LNG Esports"
```

Prop line pricing:

```text
!lol props "Team WE" "LNG Esports" --kills-line 26.5
!lol props "Team WE" "LNG Esports" --towers-line 12.5
!lol props "Team WE" "LNG Esports" --length-line 31.5
```

Prop line pricing with market odds:

```text
!lol props "Team WE" "LNG Esports" --kills-line 26.5 --kills-over-odds 1.85 --kills-under-odds 1.95
!lol props "Team WE" "LNG Esports" --towers-line 12.5 --towers-over-odds 1.90
```

Polymarket search:

```text
!lol edge "Team WE" "LNG Esports"
!lol edge "Team WE" "LNG Esports" Team WE LNG kills
```

The older direct commands still work:

```text
!bo1 "Team WE" "LNG Esports"
!bo3 "Team WE" "LNG Esports"
!bo5 "Team WE" "LNG Esports"
!props "Team WE" "LNG Esports" --kills-line 26.5
```

## How To Read The Output

**Model probability** is the model estimate for the event. For winner markets,
the bot uses calibrated probability when a calibrator artifact exists.

**Fair odds** are no-vig decimal odds from the model probability. If the model
says a team has a 55% chance, fair odds are roughly `1 / 0.55 = 1.82`.

**Expected total** is the central estimate for a prop. For example, expected
total kills of `27.4` means the model's mean projection is 27.4 combined kills.

**Over/Under probability** uses validation residual error. If expected kills are
27.4 and the line is 26.5, the bot does not blindly say Over is certain; it uses
historical model miss/error to estimate the chance that the true value clears the
line.

**Edge** compares model probability to market odds:

```text
edge = model_probability * market_decimal_odds - 1
```

Positive edge means the price is above the model's fair price. It is not a
guarantee and should be filtered by liquidity, roster confidence, model
confidence, and line movement.

**Half-Kelly** is a conservative bankroll sizing signal. It is decision support
only; the bot never places orders and never touches wallet keys.

**Confidence** is a practical warning label:

- **High**: calibrated/residual artifacts available, known teams, complete rosters,
  and no major warnings.
- **Medium**: prediction is usable but one warning exists.
- **Low**: multiple warnings exist, such as missing prop residuals, incomplete
  rosters, stale artifacts, or unknown selection context.

Map 3/4/5 props are treated as conditional on that map being played. The bot does
not dilute those props with a non-played 50-50 adjustment.
