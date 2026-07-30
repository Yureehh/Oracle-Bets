# LoL model workspace

Routine full training atomically replaces the four LightGBM directories here:
winner, game length, total kills, and total towers. Temporary `.staging/` and
`.backup/` directories are cleaned after publication.

This workspace is mutable. Long-lived candidate/champion bundles belong in the
checksum-verified registry at `data/state/model-registry/lol/`.
