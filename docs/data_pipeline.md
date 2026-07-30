# Data pipeline

## Ingestion and history

The normal pipeline requests the current season plus the previous two seasons.
`lol reconcile-history` replaces overlapping source rows after verifying stable
game/player/team identity; `lol ingest` performs an incremental merge.

Each requested year must have a nonempty local CSV. A cloud placeholder, file
changing during read, missing year, empty frame, or rows without player/team
identity fails clearly. Raw games are checked for two team rows, ten player
rows, Blue/Red composition, five roles per side, one winner, valid entities,
and exact-duplicate conflicts. Invalid games are quarantined, not silently
repaired.

## Identity

Provider names resolve through exact canonical names, reviewed external aliases,
normalized text, and conservative suggestions. Historical merges require
verified IDs. `AG.AL` maps to `Anyone's Legend`; unsupported current teams such
as Cupid Esports and MIBR.LOS remain unsupported until genuine usable history
exists.

## Features and ratings

Feature generation is chronological. Pre-match rolling values are computed
before the current result updates state. Retained families include rating
strength/uncertainty, recent performance, inactivity, roster stability,
player-role form, patch/season context, and opponent-relative statistics.

For winner training, numeric team features become canonical Team A minus Team B
deltas. Context is invariant under caller order. Map side and first pick are
excluded because they are unknown for the scheduled pre-match fixture.

## Source-column audit

`reports/lol/ingestion/data_quality.json` records:

- accepted and quarantined games;
- source seasons and schema;
- missing/new columns;
- disposition of every configured and observed source column.

New source fields are candidates, not automatic features. A field is admitted
only after availability, leakage, missingness, stability, calibration, and
temporal ablation review.
