"""Corpus preparation, plus the data it produces and the dicts trained on it.

One stage per tile layer, both writing under `data_pcg_ready` and sharing their
plumbing - the csv dialect, the vocabulary check, the naming contract - through
`corpus`. `probability_estimation_dicts` holds the DPEs estimated from the result.

- `recombine_fg_entities` folds the original levels' foreground tiles together
  with their entities into single-symbol-per-tile rooms, collapsing solid
  variants to `"1"` on the way;
- `collect_bg_rooms` validates and renames the background tiles, which need
  neither folding nor collapsing: they arrive one symbol per cell already, and
  their variants are the signal rather than decoration over one.

Neither corpus is committed - both are derived from the game. Rebuild them with
`python -m seleCte.pcg.preprocessing.recombine_fg_entities` and
`python -m seleCte.pcg.preprocessing.collect_bg_rooms` once the game data is in
place.

Nothing is re-exported here, matching `seleCte.pcg`: the module is a script as
much as a library, and re-exporting it would make `python -m` import it twice.
"""
