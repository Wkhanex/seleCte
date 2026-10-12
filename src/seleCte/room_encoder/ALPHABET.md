# The recombined alphabet, and how Maple reads it

The grids this project generates are written in one alphabet - the one
`pcg/preprocessing/recombine_fg_entities.py` produces and `utils/data/tiles_info.json`
defines - and Celeste stores rooms in another, where tiles and entities are separate
things. This file is the correspondence between the two. It was reconstructed from
`data_to_level.ipynb` (now deleted), from Maple's `src/tiles.jl` and `src/entity_map.jl`,
and checked against the vanilla dump in `data/levels/original`.

## The trick the alphabet is built around

`FgTiles` and `entityMap` are meant to be handed **the same grid**:

```julia
data = readdlm("room_1.csv", ' ', Char)
Room(fgTiles = FgTiles(data), entities = entityMap(data), ...)
```

That works because the two functions partition the character set between them.
`FgTiles` keeps only characters that name a real tileset and blanks everything else to
air (Maple's own comment on it: *"Removes illegal characters from the tileset. Makes it
possible to put simple entity map and fg in same string"*), and `entityMap` picks up
exactly the characters `FgTiles` threw away. So a symbol has to be an entity glyph
**or** a tileset glyph, never both, and the recombined alphabet is chosen to respect
that. This is also why `1` is the solid: `1` is Celeste's Dirt tileset, so it survives
`FgTiles` as an actual tile.

Nothing privileges `1` in particular - Maple's own `entityMap` example uses `3` (Snow)
for the solid and a literal space for void, and reads identically. `1` is this project's
choice, and `0` for void matters only because `FgTiles` blanks *to* `0`.

The partition holds for vanilla tilesets, listed as `valid_fg_tiles` in both
`tiles_info.json` and Maple's `src/tiles.jl`, and it is exactly what mods break. A
Strawberry Jam room dump in the deleted notebook was written in `h`, `)`, `(`, `#` and
`S`: `h` is Grass and survives, `)`, `(` and `#` are custom tilesets that `FgTiles`
blanks to air, and `S` is both a modded tileset glyph *and* `entityMap`'s spinner - so
running that grid through the pair turns 178 wall tiles into spinners. Reading modded
maps means extending `tiles.jl`, not just `entities.jl`.

## Symbol by symbol

`entityMap`'s table is `default_entity_map` in Maple's `src/entity_map.jl`.

| Symbol | `tiles_info.json` | Maple entity   | Size search | Pixel offset |
| ------ | ----------------- | -------------- | ----------- | ------------ |
| `0`    | void              | -              | -           | air          |
| `1`    | solid             | -              | -           | Dirt tileset |
| `^`    | `spikesUp`        | `SpikesUp`     | width       | `(0, 8)`     |
| `<`    | `spikesLeft`      | `SpikesLeft`   | height      | `(8, 0)`     |
| `v`    | `spikesDown`      | `SpikesDown`   | width       | `(0, 0)`     |
| `>`    | `spikesRight`     | `SpikesRight`  | height      | `(0, 0)`     |
| `D`    | `dreamBlock`      | `DreamBlock`   | both        | `(0, 0)`     |
| `C`    | `crumbleBlock`    | `CrumbleBlock` | width       | `(0, 0)`     |
| `_`    | `jumpThru`        | `JumpThru`     | width       | `(0, 0)`     |
| `O`    | `greenBooster`    | `GreenBooster` | -           | `(4, 4)`     |
| `Q`    | `redBooster`      | `RedBooster`   | -           | `(4, 4)`     |
| `W`    | `water`           | `Water`        | both        | `(0, 0)`     |
| `S`    | `spinner`         | `Spinner`      | -           | `(4, 4)`     |
| `B`    | `bigSpinner`      | `Bumper`       | -           | `(4, 4)`     |
| `R`    | `refill`          | `Refill`       | -           | `(4, 4)`     |
| `F`    | `infiniteStar`    | `Feather`      | -           | `(4, 4)`     |
| `P`    | *(not in the map)* | `Player`      | -           | `(8, 8)`     |
| `H`    | *(not in the map)* | **no entry**  | -           | -            |
| `L`    | `lightbeam`       | **no entry**   | -           | -            |

Fourteen of the fifteen entity symbols in `tiles_info.json` are Maple's glyphs exactly.
That is not a coincidence to be re-derived later; it is the reason the encoder is three
lines long.

"Size search" is `entityMap` walking right and/or down from the first cell while the
character repeats, and passing the run length to the entity as a width/height. Entities
with no size search occupy one cell and are emitted with no dimensions at all. The
placement rule, for a 1-based row/column, is:

```
x = column * 8 + offset_x - 8
y = row    * 8 + offset_y - 8
```

### The two gaps

`H` and `L` have no *upstream* entry. They look alike and are not remotely equally
urgent: `L` is cosmetic, `H` is how a generated level ends. `H` is closed - the pinned
fork carries it - and `L` is not, and is not going to be.

- **`H` (crystal heart)** is written by `celeskeleton` after generation, along with `P`.
  `P` round-trips against any Maple; `H` needs the fork. The deleted notebook's saved
  output showed a `CrystalHeart("blackGem", ...)` coming back out of `entityMap`, at the
  exact pixel coordinate the `H` in the fixture below predicts - which meant those cells
  had been run against a **patched Maple**. That patch is
  <https://github.com/robilouis/Maple>, which adds `'H' => CrystalHeart` and also lists
  `CrystalHeart` in `entity_offsets` and `search_for_size` (both at the values it would
  fall through to anyway). It is what `Project.toml` now pins, and encoding the fixture
  against it reproduces the eleven-entity table below exactly, `blackGem` at x=256, y=32
  included. Against upstream the heart is silently dropped and the level ships
  uncompletable, which is why the pin is the fork rather than a preference.

  **Why it was the priority.** Vanilla ending zones are hand-authored one-offs - a
  Badeline launcher into a finish zone, a campfire, a cliff - and fewer than ten of them
  exist. Nothing learns "what an ending is" from that. Every B-side, C-side and the 100+
  Strawberry Jam levels instead end on a crystal heart the player dashes through, which
  is why `celeskeleton` places exactly one, in the end room, deterministically. So the
  heart is never something to *learn*: it is placed, not sampled, and it is deliberately
  absent from `ALL_TILES_AND_ENTITIES` for that reason. No corpus rebuild, no DPE
  regeneration, no symbol budget, no widening of the MdMC state space - unlike the
  springs below, which need all four. It should ship first.

  **It took two things, though, not one.** The glyph only gets the heart *placed*.
  Making the level *end* on it is `HeartIsEnd`, a chapter setting that lives in the
  map's `meta` element - and a bare `Map` has no metadata at all, so for as long as the
  encoder called `encodeMap` the flag could not reach the file no matter what the fork
  said. `data_to_level.jl` encodes a `Side` for exactly this reason, merging Maple's
  `default_meta`/`default_mode`/`default_audiostate` in; see `chapter_metadata` there.
  Note those consts are templates rather than defaults the library applies - nothing in
  Maple reads them - so they only ever take effect because the encoder merges them. Miss
  either half and the level is silently uncompletable: no heart, or a heart that is just
  a collectible. Both halves have to move together, so when bumping the Maple pin, check
  a generated `.bin` still comes back with a `meta` element and a `blackGem` in it.

  Longer term the ending is still open. A dedicated "ending generator" - a submodel that
  reads ending rooms and chooses between a finish zone and a crystal heart - would buy
  back some of the variety the hand-authored endings have. The heart is the simpler and
  far more visual answer for now, not the final one.
- **`L` (lightbeam)** is in `tiles_info.json`, so the model can learn it and generation
  can emit it, but upstream Maple cannot encode it. A generated room containing `L`
  loses those cells silently. Lightbeams are decoration, so this is a known and accepted
  loss rather than something to fix - do not spend fork budget on it.

### TODO: springs, and the fork that has to carry them

Springs are absent from *both* ends of the pipeline, which is why they never show up
anywhere: `tiles_info.json` has no entry for them, so recombination drops them, and
`default_entity_map` has no glyph for them, so the encoder could not write them back
even if it did. They are not rare - the vanilla dumps hold 465 (`spring` 217,
`wallSpringLeft` 166, `wallSpringRight` 82), making them the largest gameplay-relevant
omission after `lightning`.

Only the glyph line is missing on Maple's side: upstream already carries
`entity_offsets` and `search_for_size` entries for all three (`Spring => (8, 8)`,
`SpringLeft => (0, 8)`, `SpringRight => (8, 8)`, all point entities), so the fork patch
is three lines that sit alongside the `'H' => CrystalHeart` one above.

Deferred to a session dedicated to widening the entity vocabulary. The symbol choice is
the real decision, not the plumbing: springs are directional like the spikes, so three
symbols is the consistent reading, but that spends three characters and widens the MdMC
state space. Whatever is picked has to stay disjoint from `valid_fg_tiles`, per the
partition at the top of this file. Adding them also invalidates the corpus and the
committed `probability_estimation_dicts`, which have to be rebuilt together.

## The fixture

`fixtures/round_trip_room/` is a hand-written 40x23 room - the modal Celeste room size,
and the generator's default - in this alphabet, with `data.json` alongside it so it can
be fed straight to the encoder:

```bash
julia --project=. src/seleCte/room_encoder/data_to_level.jl \
    src/seleCte/room_encoder/fixtures/round_trip_room round_trip_room
```

It exercises three of the four size-search behaviours (`^` a width run, `_` a width run,
the point entities `R`/`Q`/`F`) plus `P`, and `H` for the gap above. Against a Maple with
the `H` patch it produces these eleven entities, in this order - this is the assertion,
recorded from the notebook's saved output and re-derived from the placement rule above:

| Entity                  | x   | y   | size        | from        |
| ----------------------- | --- | --- | ----------- | ----------- |
| `Refill`                | 164 | 28  | -           | row 4, col 21  |
| `CrystalHeart` (blackGem) | 256 | 32  | -         | row 5, col 33  |
| `Refill`                | 148 | 52  | -           | row 7, col 19  |
| `Refill`                | 116 | 76  | -           | row 10, col 15 |
| `Booster` (`red=true`)  | 212 | 84  | -           | row 11, col 27 |
| `Refill`                | 204 | 116 | -           | row 15, col 26 |
| `Feather` (infiniteStar) | 116 | 140 | -          | row 18, col 15 |
| `Refill`                | 196 | 140 | -           | row 18, col 25 |
| `Player`                | 48  | 152 | -           | row 19, col 6  |
| `JumpThru`              | 32  | 160 | `width=24`  | row 21, col 5  |
| `SpikesUp`              | 112 | 176 | `width=192` | row 22, col 15 |

`entityMap` scans row-major, so the order is reading order, and `Map`/`encodeMap` keep
it. Rooms sized in pixels come out as `size = (320, 184)`, since `size(::Tiles)` is
`(width, height) * 8`.

## Round-trip caveats

Encoding is exact for tiles. Decoding a `.bin` back through
`recombine_fg_entities.py` is **not** exact for every entity, because that side of the
pipeline does not implement Maple's `entity_offsets` table:

- **Point entities** (`S`, `R`, `F`, `O`, `Q`, `B`) are written at the centre of their
  cell, `column * 8 - 4`. Decoding floors that back to the right cell and then applies
  `POINT_ENTITY_SHIFT`, which exists for Celeste's own centred 16x16 boxes, so a
  Maple-authored point entity comes back one tile up and to the left.
- **`^` and `<`** are the two spikes with a non-zero offset - `(0, 8)` and `(8, 0)`.
  Decoding floors `y`/`x` with no offset applied, so it lands on the solid the spikes
  stand against rather than on the cell the spikes occupy. The vanilla dump agrees:
  1512 of 1987 in-bounds `spikesUp` land on a solid tile, with air in the cell above in
  1957 of them; 384 of 532 `spikesLeft` land on a solid, with air to the left in 530.
  `v`, `>`, `_`, `D`, `C` and `W` have a zero offset and are unaffected.

Neither of these matters for the current corpus, which is built from the game's own
`.bin` files rather than from re-encoded output - but a generated level that is decoded
and fed back in as training data will drift by a tile each pass. `entity_offsets` is
the correction table if that ever needs fixing.
