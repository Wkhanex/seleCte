# Changelog

Notable changes to seleCte, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The repository's history starts at the first public release, so this file is the record
of everything before it.

## [0.10.0] - 2026-10-12

First public release.

### Fixed

- Level generation no longer hangs. Exit coordinates were converted between absolute and
  room-local positions in three places that disagreed, so placing the spawn or the
  crystal heart could fail on every attempt, and the retry loop never gave up. The
  conversion now lives in one place.
- Horizontally connected rooms carved their halves of a shared doorway at different
  heights.
- Every retry loop is bounded, so a run always terminates, and reports when what it
  wrote is suspect.

### Changed

- Only rooms on the route from the starting room to the ending room are held to the
  playability check. Rooms on side branches are filled once and not retried, so levels
  are no longer thrown away over rooms the player never has to cross.
- The README is rewritten around a quickstart, with the technical detail moved to
  `docs/how-it-works.md`.
- The license names the author, Louis Robinet, for 2022 to 2026.

### Added

- `Celeskeleton.get_traversal_path` and `get_room_graph`, which read a level's route and
  room adjacency off its exits, and `Room.is_connected_to`.
- `CITATION.cff`, so GitHub's *Cite this repository* button gives the paper.

### Removed

- All extracted Celeste level data. Extract it from your own copy of the game instead;
  the README explains how.
- `ipykernel`, unused since the notebooks were removed, along with its dependencies,
  including `tornado`.

## [0.9.0] - 2026-08-04

### Added

- Background tile generation, through the existing pipeline with a `--layer {fg,bg}`
  flag: corpus building in `preprocessing/collect_bg_rooms.py`, training and sampling.
  Train background models one chapter at a time, since each chapter uses one tileset.
- A CI job that encodes a level end to end in Julia, `tests/julia_smoke.jl`.

### Changed

- Maple is pinned to [a fork](https://github.com/robilouis/Maple) that can encode the
  crystal heart. Upstream cannot, so its levels could not be finished.
- The encoder writes a full `Side`, so the `HeartIsEnd` setting reaches the map and
  collecting the heart ends the chapter.
- `preprocessing/corpus.py` holds the csv format, vocabulary check and naming rules
  shared by both layers.
- `color_map` takes a per-layer palette, so background rooms can be displayed.
- `generate_empty_room_data` takes a border symbol, and background rooms start from
  void.

### Fixed

- `data_to_level.jl` placed every room one room-height too low, so rooms of different
  heights overlapped.
- Feather and bumper were keyed by readable names instead of Celeste's own,
  `infiniteStar` and `bigSpinner`, so neither ever reached the training corpus.
- Background rooms could draw foreground symbols for patterns the model had not seen.
- `estimate_dpe` fails clearly on a missing corpus, or a dataset matching no room.

### Removed

- The `itertools.product` pre-seed in `dpe_generation`, which made masks wider than
  `d7` impractical.

Known limitations: background output is sparser than the game's, and backgrounds cannot
yet be written into a map.

## [0.8.0] - 2026-07-31

### Added

- `data_loader/data_loader.jl`, a command-line tool that extracts levels from a maps
  folder into a separate output folder.
- `room_encoder/ALPHABET.md`, documenting how the tile alphabet maps onto Celeste's
  tiles and entities.
- A hand-written 40x23 round-trip fixture under `room_encoder/fixtures/`.

### Changed

- Maple is pinned by commit in a root `Project.toml`, instead of being re-resolved
  against upstream on every run.
- `data_to_level.jl` creates its output folder, names missing rooms instead of failing
  obscurely, and prints usage on `-h`.

### Fixed

- `data_to_level.jl` never imported Maple, so it could not run.
- `data_loader` misidentified files such as `a.b.bin`.

### Removed

- `CSV` and `DataFrames` from the Julia dependencies.
- The `data_loader` and `data_to_level` notebooks.

## [0.7.0] - 2026-07-31

### Added

- `preprocessing/recombine_fg_entities.py`, replacing the notebook that built the
  training corpus.

### Changed

- The entity-to-symbol map moved into `tiles_info.json`.
- All ten probability dicts were regenerated. `pe_d5_full.json` now covers all 824
  rooms rather than a partial corpus.
- The recombined corpus is no longer committed. Rebuild it from your own game data.

### Fixed

- Point entities on a room's first row or column were dropped, losing 903 entities
  across 112 rooms.
- A circular import made every generation entry point fail on import.
- `dpe_generation` raised `KeyError` on its first iteration, so dicts could not be
  regenerated.
- `filter_entities` ignored its entity-map argument, and the invalid-tile warning
  reported the wrong count.

### Removed

- The unreferenced `rooms_40x23/` and `40x23_fg/` folders, and the notebook's duplicated
  helpers.

## [0.6.0] - 2026-07-28

### Changed

- Refactor for publication: fixed broken imports and paths, moved to uv, and added
  tests, linting and type checking.
- `Celeskeleton.show_skeleton` rewritten with matplotlib, dropping `bokeh`.

## [0.5.1] - 2026-01-02

- Global variables moved to config files.
- Readability and maintainability improvements in `celeskeleton`.

## [0.5.0] - 2024-12-10

The version described in the FDG 2025 paper.

- Repository cleaned up for publication, removing tests and local experiments.

## [0.4.0] - 2024-08-05

- Evaluation functions and helpers in `utils.py`.

## [0.3.0] - 2024-04-06

- A\*-based playability check, integrated into generation.
- `celeste_pcg_utils.py` and `celeste_playability_utils.py` merged into `utils.py`.
- A command-line pipeline from parameters to a binary map file.

## [0.2.0] - 2024-02-02

- The model creates borders and exits for each room.
- Major code reorganization.
- The pipeline links the skeleton to the room encoder.

## [0.1.0] - 2023-12-22

- Notebook combining the skeleton with room generation.
- Map encoding in Julia, without exit handling yet.

## [0.0.8] - 2023-10-10

- First draft of the multi-dimensional Markov chain model.
- Visual interface built on the pandas Styler.
- Foreground tiles and entities recombined into one structure the model can read.

## [0.0.7] - 2023-09-22

- Room size distribution: csv summaries of rooms per size, room sizes and entities,
  with plots.
- Data exploration notebook.

## [0.0.6] - 2023-09-19

- Skeleton code reorganized, with functions converting between skeletons and JSON.
- Minor fixes to `Room` and `Celeskeleton`.
- First steps from notebooks to Julia scripts.

## [0.0.5] - 2023-03-07

- Data exploration functions.
- First draft of the playability exploration.

## [0.0.4] - 2023-02-11

- Skeleton v0 finished.

## [0.0.3] - 2023-02-07

- `Celeskeleton` class.

## [0.0.2] - 2023-02-04

- `Room` class.

## [0.0.1] - 2023-02-03

- Project structure.
- Bokeh visual interface for the level skeleton.
- `data_loader` to build the rooms database.
