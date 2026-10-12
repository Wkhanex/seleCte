# seleCte

Procedural generation of playable [Celeste](https://www.celestegame.com/) levels, from
the game's own levels back into a `.bin` the game can load.

A multi-dimensional Markov chain is trained on the tile patterns of Celeste's 824
vanilla rooms, then sampled to fill the rooms of a procedurally drawn level skeleton.
Both ends of the pipeline talk to Celeste's binary map format through
[Maple](https://github.com/CelestialCartographers/Maple).

This started as a Master's thesis and is an active research project. It produces
levels that Celeste loads, that start on a player spawn and end on a crystal heart —
it does not yet produce levels that are reliably *good*. Read [Status](#status)
first.

## Paper

This repository is the implementation behind:

> Louis Robinet, Marcello A. Gómez-Maureira, and Mike Preuss. 2025. **Towards a Celeste
> AI Framework: Agent-free Automated 2D Level Generation for Multidirectional
> Platformers.** In *FDG '25: Proceedings of the 20th International Conference on the
> Foundations of Digital Games*. https://doi.org/10.1145/3723498.3723796

```bibtex
@inproceedings{robinet2025celeste,
  author    = {Robinet, Louis and G\'{o}mez-Maureira, Marcello A. and Preuss, Mike},
  title     = {Towards a Celeste AI Framework: Agent-free Automated 2D Level
               Generation for Multidirectional Platformers},
  booktitle = {FDG '25: Proceedings of the 20th International Conference on the
               Foundations of Digital Games},
  year      = {2025},
  doi       = {10.1145/3723498.3723796}
}
```

`CITATION.cff` carries the same metadata, so GitHub's *Cite this repository* button
produces it directly.

The code has moved on considerably since the paper was written. The state it
describes is tagged [`v0.5.0`](https://github.com/robilouis/seleCte/tree/v0.5.0);
everything from `0.5.1` onwards in [CHANGELOG.md](CHANGELOG.md) postdates it,
including the whole refactor, the Julia CLIs and the background layer.

## Status

Honest scope, as of `0.9.0`.

**Works**

- Skeleton generation: room rectangles, sizes, positions and the exits linking them,
  from a corridor (`p = 0`) to a labyrinth (`p` near 1).
- Foreground room sampling from a trained probability dict.
- Level assembly: rooms filled, exits carved, respawn points placed, player spawn and
  crystal heart positioned, with retries at three nested levels — every retry loop
  bounded, so a run always terminates and says so when what it wrote is suspect.
- Encoding to a `.bin` that Celeste loads and that *ends* when you collect the heart.
- Background room sampling, as a standalone `height x width` array of symbols.

**Approximate — read this before trusting the output**

Playability is 8-connected A\* over open cells. It models **no gravity, no jump reach,
no dash, no stamina and no wall-grab**, so "playable" currently means *a path through
empty space exists*, which is nearly always true and says very little about Celeste. It
certifies rooms that cannot be played, and rejects rooms that are fine. Replacing it
with a flood-based directed reachability model is the main line of active work.

**Not done**

- The background layer stops at an array. There is no skeleton integration
  (`-et skel` rejects `--layer bg`), `Room` has no background layer, and the encoder
  never writes `lvl_<id>_bg.csv` back out — so generated backgrounds cannot reach the
  game yet.
- The v1 encoding is a static 2D grid, so **nothing that moves can be represented**:
  `zipMover`, `moveBlock`, `fallingBlock`, `swapBlock` and their kin are dropped.
- 15 of roughly 150 vanilla entity names are mapped. The largest gameplay-relevant
  omissions are `lightning` and springs.
- Generated rooms under-fill relative to the corpus on both layers.

**Slow, and honest about it**

The playability check is the bottleneck, and it scales badly with room area: a 7-room
level of 22x14 rooms takes a few seconds, the same level at the default 40x23 takes
minutes. Most of that is A\* rescanning its open and closed lists rather than the
search itself. The flood rework replaces it.

Known defects and planned work live in the
[issue tracker](https://github.com/robilouis/seleCte/issues).

## Requirements

| | |
|---|---|
| Python 3.14 | managed with [uv](https://docs.astral.sh/uv/) |
| Julia 1.11+ | `[sources]` in `Project.toml` is a 1.11 feature |
| A copy of Celeste | **optional** — only needed to retrain the model |

The trained probability dicts are committed, so you can generate and encode levels
without owning the game. You need Celeste only to rebuild the corpus from its maps.

## Install

```bash
uv sync
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The Julia step resolves `Maple`, which is pinned by commit in `Project.toml`. **The pin
is [our fork](https://github.com/robilouis/Maple), not upstream, and that is
load-bearing rather than a preference:** upstream cannot encode the crystal heart, and a
generated level ends on one, so building against upstream silently ships levels that
cannot be finished. The fork also flips `HeartIsEnd`, which is what makes collecting the
heart end the chapter. To move the pin, change the single `[sources]` entry — a commit
on the fork, or a local checkout while working on it — and re-instantiate. Commit the
resulting `Manifest.toml`, so the pin covers the whole dependency tree rather than just
`Maple`.

## Quickstart

Generate a complete level and encode it, with no game data required:

```bash
# 1. draw a 7-room level, fill it, place the spawn and the crystal heart
uv run python -m seleCte.pcg.level_generation --nb-rooms 7 --proba 0.3 \
    --room-size 22 14 --level-name my_level --tries-limit 4

# 2. encode it into a .bin Celeste can load
julia --project=. src/seleCte/room_encoder/data_to_level.jl my_level my_level
```

Step 1 writes `my_level/`: one `room_<n>.csv` per room plus a `data.json` holding each
room's origin, size and exits. Step 2 turns that into
`generated_bin_files/my_level.bin`. To play it, drop the `.bin` into a mod folder for
[Everest](https://everestapi.github.io/) and load it from the chapter list.

Expect a few seconds at this room size. The default 40x23 works but takes far longer —
the playability check is the bottleneck and it scales badly with room area, so start
small while you are looking around.

Useful flags: `--proba` is the labyrinth probability (`0` is a straight corridor),
`--room-size` takes width then height, and `--reset-skeleton` redraws the whole level
rather than keeping a room that never came out playable.

### Sampling rooms without building a level

`rooms_generation` is the experiment harness. It samples from a dict with no
playability check and no retries, which is what makes it the tool for comparing dicts
rather than for producing levels — nothing filters what it writes, so nothing biases
what you measure.

```bash
# loose, unconnected rooms
uv run python -m seleCte.pcg.rooms_generation --experiment-type rooms \
    --nb-rooms 5 --dict-proba-estimation pe_d5_full --output-dir my_rooms

# or a connected skeleton, filled and carved but never checked - about a second
uv run python -m seleCte.pcg.rooms_generation --experiment-type skel \
    --nb-rooms 5 --proba 0.3 --output-dir my_rooms_skel
```

## Retraining the model

This repository **ships no Celeste level data**. It is not ours to redistribute — you
extract it from your own copy of the game. What *is* committed is
`probability_estimation_dicts/`: the trained model, which is tile-frequency statistics
rather than level content.

Dump the vanilla chapters into the tree the rest of the pipeline expects:

```bash
julia --project=. src/seleCte/data_loader/data_loader.jl \
    "<Celeste>/Content/Maps" src/seleCte/data/levels/original
```

Each map becomes one subdirectory holding `lvl_<id>_{fg,bg,obj}.csv` and
`lvl_<id>_entities.json` per room. Anything that is not a `.bin` is skipped, so the
texture caches sitting next to the maps are harmless. Mod dumps such as Strawberry Jam
work the same way — point `<maps-dir>` at them.

Then build the corpus and estimate a new dict:

```bash
uv run python -m seleCte.pcg.preprocessing.recombine_fg_entities   # -> all_rooms/
uv run python -m seleCte.pcg.dpe_generation --dataset 0123456789L
```

`--dry-run` on the preprocessing step reports what would change without writing;
`--levels-dir` points it at a mod dump instead of the vanilla tree.

## Pipeline

```
Celeste .bin                                      you supply, from your own copy
  │  data_loader/data_loader.jl                   Julia + Maple, loadMap
  ▼
data/levels/original/<Chapter>/lvl_<id>_{fg,bg,obj}.csv + _entities.json
  │                                    │
  │  preprocessing/                    │  preprocessing/
  │    recombine_fg_entities.py        │    collect_bg_rooms.py
  ▼                                    ▼
data_pcg_ready/all_rooms/              data_pcg_ready/all_rooms_bg/
        824 rooms each, generated, not committed
  │  pcg/dpe_generation.py [--layer fg|bg]        3x3 MdMC window, mask 000011012
  ▼
probability_estimation_dicts/{,bg_}*.json         the trained model, COMMITTED
  │                                               ← start here without the game
  │  pcg/level_generation.py + celeskeleton + playability
  ▼                                               (foreground only past this point)
<level>/room_N.csv + data.json
  │  room_encoder/data_to_level.jl                Julia + Maple, encodeSide
  ▼
playable .bin
```

## Modules

### `celeskeleton`

The structural stage, run before any tile is sampled. A `Celeskeleton` is a mapping of
room name to `Room`, each an axis-aligned rectangle on one integer grid.
`generate_celeskeleton` grows a level as a **tree** — each new room attaches to one
existing room — so there is exactly one route from the starting room to the ending room.
`get_traversal_path` returns it, and only those rooms are held to the playability check;
everything else is an optional branch. `show_skeleton` plots the layout with matplotlib.

### `pcg.preprocessing`

`recombine_fg_entities.py` folds each room's foreground tiles and entities into the
single-symbol-per-tile grid the model trains on. Every solid tile variant collapses to
`1` — subtypes are decoration, and would have the Markov model learn tileset transitions
instead of level shape — and each recognised entity stamps its symbol over the tiles it
covers.

`collect_bg_rooms.py` is the background counterpart and is mostly the *absence* of the
foreground one: the game's `_bg.csv` files are already one symbol per cell, so it only
validates and renames. Nothing is collapsed, because on a decorative layer the tileset
variants *are* the signal rather than noise over it.

### `pcg`

`dpe_generation.py` slides a window over the corpus and counts which symbol follows each
neighbourhood pattern, normalized per pattern. The window is the **MdMC mask**, a
flattened string where `1` marks a neighbour and `2` the cell being predicted. The
default `000011012` is `d5`:

```
0 0 0
0 1 1     ← hence `pe_d5_full.json`
0 1 2
```

`rooms_generation.py` samples loose rooms with no playability guarantees — the
experiment harness. `level_generation.py` wraps the same sampling in the skeleton, the
playability check, the special-point placement and the retry loops.

### Background layer

The same model runs over the background layer, selected with `--layer bg`:

```bash
uv run python -m seleCte.pcg.preprocessing.collect_bg_rooms
uv run python -m seleCte.pcg.dpe_generation --layer bg --dataset 6
uv run python -m seleCte.pcg.rooms_generation --layer bg -et rooms -nr 5 \
    -dpe bg_6_000011012 -od my_bg_rooms
```

**Train one chapter at a time.** A background tileset belongs to the chapter, not to the
room: 674 of the 824 vanilla rooms hold exactly void plus one tileset, and which tileset
is fixed per chapter (Reflection is `a`, Golden Ridge `6`). Trained over all chapters at
once, the dominant all-void pattern resolves to a mix of seven tilesets that never
co-occur in the game, and generation sprinkles them.

### Playability

`Room.is_playable_room` runs A\* between a room's exits, via `utils.astar`. See
[Status](#status) for what this does and does not model — the short version is that it
is an indicator, not a verdict. Most of the code is in `utils/utils.py`, alongside the
difficulty and interestingness metrics, which score a room along the path rather than
over the whole room.

### The Julia ends

`data_loader/data_loader.jl` decodes `.bin` maps into the per-room csv/json dump, and
`room_encoder/data_to_level.jl` encodes a generated level back into a `.bin`. Both are
run from the repo root with `--project=.`, take positional arguments, and print usage
on `-h`.

`room_encoder/fixtures/round_trip_room/` is a hand-written 40x23 room you can encode
straight away to check a setup works:

```bash
julia --project=. src/seleCte/room_encoder/data_to_level.jl \
    src/seleCte/room_encoder/fixtures/round_trip_room round_trip_room
```

## The alphabet

One character per tile, throughout:

```
0 void   1 solid   ^ < v > spikes   S spinner   D dreamBlock   C crumbleBlock
_ jumpThru   O greenBooster   Q redBooster   W water   B bigSpinner   R refill
F infiniteStar   L lightbeam
```

Keys are Celeste's own internal names, so two do not read as what they are:
`bigSpinner` is the bumper and `infiniteStar` is the feather. `P` (player) and `H`
(crystal heart) are placed after generation and are deliberately outside the sampled
vocabulary. The vocabulary itself lives in `src/seleCte/utils/data/tiles_info.json`.

[`src/seleCte/room_encoder/ALPHABET.md`](src/seleCte/room_encoder/ALPHABET.md)
documents how this alphabet maps onto Celeste's own tiles and entities, which symbols
upstream Maple cannot decode, and where the round trip is not exact. **Read it before
touching either Julia script or the symbol vocabulary.**

## Development

```bash
git add -A                      # pre-commit only sees files known to git
uvx pre-commit run --all-files
uv run mypy --no-pretty
uv run pytest -q
julia --project=. tests/julia_smoke.jl
```

The four checks below the `git add` all run in CI. Staging first is not incidental:
pre-commit only sees files known to git, so running the hooks before `git add`
skips new files entirely.

Conventions: ruff at line length 88 with Google-style docstrings, mypy `strict` as the
gate, [ty](https://github.com/astral-sh/ty) as the fast local loop, `pathlib` over
`os.path`, and a module-level `logger` rather than `print`.

## Roadmap

Tracked work is in the
[issue tracker](https://github.com/robilouis/seleCte/issues). The research-level
directions beyond it:

- **Replace the playability model.** Reachability under gravity is *directed* — falling
  is free, climbing is not — so per-room "is it playable" is not even well-formed. The
  design is an n x n exit-reachability matrix from n floods, over a movement model with
  jump, dash, stamina and wall-grab, validated against the 824 vanilla rooms as a
  known-playable positive class.
- **An ending generator**, reading ending rooms and choosing between a finish zone and a
  crystal heart, instead of always emitting a heart.
- **Directional bias in the skeleton**, so a level can be told to trend in one of eight
  directions with a tunable strength, rather than growing at random.
- **Interesting collectibles**: strawberries placed so that reaching them is a
  deliberate detour.

## License and attribution

The code in this repository is MIT licensed — see [LICENSE](LICENSE).

Celeste is © [Maddy Makes Games](https://www.maddymakesgames.com/) / Extremely OK
Games. This project is an unaffiliated, non-commercial research project. **No game
assets or level data are redistributed here** — everything derived from the game is
extracted by you, from your own copy, with the data loader above.

Both Julia scripts are built on
[Maple](https://github.com/CelestialCartographers/Maple) by the Celestial
Cartographers, also MIT licensed, pinned to [a fork](https://github.com/robilouis/Maple)
that adds crystal-heart encoding.
