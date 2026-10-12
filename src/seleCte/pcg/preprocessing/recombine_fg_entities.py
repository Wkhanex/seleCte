"""Builds the corpus of rooms the MdMC model trains on, from the raw game data.

Run as a script, this is the first offline stage of the PCG pipeline, sitting
directly upstream of `dpe_generation`. For every room the data loader extracted
it reads two files - the foreground tiles (`lvl_<id>_fg.csv`) and the entities
(`lvl_<id>_entities.json`) - and folds them into the single representation
everything downstream understands: a `(height, width)` grid holding one tile
symbol per cell.

Two things happen on the way:

- Every solid foreground variant collapses to `"1"`, leaving `"0"` for void. The
  subtypes encode decoration, not structure, and feeding them to a Markov model
  would have it learn tileset transitions instead of level shape. Re-decorating a
  generated room is a separate concern.
- Each entity the vocabulary knows about stamps its symbol over the tiles it
  covers, so a hazard is just another symbol in the same grid rather than a
  parallel data structure the model would have to be taught to read.

Entity coordinates arrive in pixels at `TILE_SIZE` pixels to the tile, with `x`
along the columns and `y` along the rows. `width`/`height` are optional and
independent - vertical spikes carry only `height`, horizontal spikes only
`width`, and `spinner`/`refill`/`booster` carry neither, which makes them point
entities. Entities are stamped in file order, so a later one overwrites an
earlier one where they overlap.

This is a v1 encoding and deliberately lossy: a static grid cannot express
anything that moves, so `zipMover`, `moveBlock`, `trackSpinner` and their kin are
dropped along with every other name absent from `ENTITY_SYMBOLS`. That is the
right trade for the MdMC model; a richer model would need a different structure.

Output names are `{chapter_prefix}_{fg_filename}`, e.g. `1_lvl_1_fg.csv` - see
`corpus.output_filename`. `dpe_generation.extract_dataset_from_filepath` reads the
dataset id off the first character of that name, so the prefix is a contract
between the two stages, not cosmetic.

The csv dialect, the vocabulary check and that naming contract are shared with the
background stage and live in `corpus`; what is left here is the part that is
genuinely foreground-only, which is all of the entity handling.
"""

import argparse
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pandas.core.frame import DataFrame

from seleCte.paths import ALL_ROOMS_DIR, LEVELS_DIR
from seleCte.pcg.preprocessing.corpus import (
    SOLID_TILE,
    invalid_symbol_counts,
    output_filename,
    read_room_csv,
    write_room_csv,
)
from seleCte.utils import ENTITY_SYMBOLS, VALID_FG_TILES

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

# Celeste's grid: every entity coordinate and size in the json is in pixels.
TILE_SIZE: int = 8

FG_SUFFIX: str = "_fg.csv"
ENTITIES_SUFFIX: str = "_entities.json"

# An entity with no width or height is a point entity, whose box is centred on
# `(x, y)` rather than starting there, so it covers the tile before the one the
# coordinate lands in. Every `spinner` in the game data has `originX == originY ==
# 8`, which is exactly this one-tile shift.
# TODO: derive the shift per entity as `originX // TILE_SIZE` instead of assuming
# it. `lightbeam` has an `originX` between 8 and 80, so the constant
# under-corrects for its 19 instances - too few to justify moving the corpus now.
POINT_ENTITY_SHIFT: int = 1

# Immutable so it can be a default argument without tripping B006, and so a caller
# passing its own map cannot mutate the shared vocabulary.
DEFAULT_ENTITY_MAP: Mapping[str, str] = MappingProxyType(ENTITY_SYMBOLS)

# Boosters carry their colour in a `red` flag rather than in `__name`, so they are
# the one entity whose symbol depends on more than its name.
BOOSTER_NAME: str = "booster"
RED_BOOSTER_NAME: str = "redBooster"
GREEN_BOOSTER_NAME: str = "greenBooster"


class Args:
    """Parsed command-line arguments, one attribute per `add_argument` call.

    Deliberately not an `argparse.Namespace` subclass, for the reason spelled out
    in `seleCte.pcg.dpe_generation.Args`: `Namespace.__getattr__` returns `Any`,
    so a typo would type-check and only fail mid-run.
    """

    levels_dir: str
    output_dir: str
    dry_run: bool


@dataclass(frozen=True, slots=True)
class MappedEntity:
    """One entity recombination knows how to stamp, in pixel coordinates.

    `width`/`height` are `None` when the entity's json carries no such key, which
    is how a point entity is spelled. Modelling that as `None` rather than as the
    `NaN` a ragged frame would produce is what keeps `_tile_span` arithmetic in
    integers.
    """

    name: str
    symbol: str
    x: float
    y: float
    width: float | None
    height: float | None


@dataclass(frozen=True, slots=True)
class RoomReport:
    """What happened to one room, so `main` can summarize without re-reading."""

    fg_path: Path
    n_entities_kept: int
    n_entities_outside: int
    invalid_symbols: dict[str, int]


def collapse_fg_tiles(df_fg: DataFrame) -> DataFrame:
    """Rewrites every solid foreground variant to `"1"`, leaving the rest alone.

    Unknown symbols are passed through rather than dropped: `invalid_symbol_counts`
    has to be able to see them.

    Args:
        df_fg: a room's raw foreground tiles, one symbol per cell.

    Returns:
        A new frame holding only `"0"`, `"1"` and whatever was not recognized.
    """
    return df_fg.map(func=lambda tile: SOLID_TILE if tile in VALID_FG_TILES else tile)


def _resolve_entity_name(entity: dict[str, Any]) -> str:
    """Returns an entity's name, splitting `booster` by colour.

    Raises:
        ValueError: the entity is a booster with no `red` flag to split on.
    """
    name: str = str(entity["__name"])
    if name != BOOSTER_NAME:
        return name

    if "red" not in entity:
        raise ValueError(
            f"Booster id={entity.get('__name')} has no `red` flag to pick a colour "
            f"from: {entity!r}."
        )

    return RED_BOOSTER_NAME if entity["red"] else GREEN_BOOSTER_NAME


def _optional_size(entity: dict[str, Any], key: str) -> float | None:
    """Returns an entity's `width`/`height` in pixels, or `None` if it has none."""
    value: Any = entity.get(key)

    return None if value is None else float(value)


def _build_mapped_entity(
    entity: dict[str, Any], name: str, symbol: str
) -> MappedEntity:
    """Builds the carrier for one entity, validating its coordinates.

    Raises:
        ValueError: the entity has no `x`/`y`, or they are not numeric.
    """
    try:
        x: float = float(entity["x"])
        y: float = float(entity["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Entity {name!r} id={entity.get('id')} has no usable x/y "
            f"coordinates: {entity!r}."
        ) from exc

    return MappedEntity(
        name=name,
        symbol=symbol,
        x=x,
        y=y,
        width=_optional_size(entity=entity, key="width"),
        height=_optional_size(entity=entity, key="height"),
    )


def filter_entities(
    entities: list[dict[str, Any]],
    entity_map: Mapping[str, str] = DEFAULT_ENTITY_MAP,
) -> list[MappedEntity]:
    """Keeps the entities `entity_map` has a symbol for, in file order.

    Boosters are resolved to their coloured name first, so a map can address
    `redBooster` and `greenBooster` separately even though the game data spells
    both `booster`.

    Args:
        entities: one room's entities, as read from its json.
        entity_map: entity names to the symbol they stamp. Defaults to the
            project vocabulary in `tiles_info.json`.

    Returns:
        The kept entities, in the order they appeared.

    Raises:
        ValueError: a kept entity has no usable coordinates, or a booster carries
            no `red` flag.
    """
    mapped: list[MappedEntity] = []
    for entity in entities:
        name: str = _resolve_entity_name(entity=entity)
        symbol: str | None = entity_map.get(name)
        if symbol is not None:
            mapped.append(_build_mapped_entity(entity=entity, name=name, symbol=symbol))

    return mapped


def _tile_span(
    origin_px: float, size_px: float | None, limit: int
) -> tuple[int, int] | None:
    """Converts one axis of an entity's pixel box to a half-open tile span.

    The span is clamped to `[0, limit)`, which matters twice over. Positionally
    indexing with a negative bound would count back from the end of the room
    instead of falling off it, and an entity may legitimately sit partly - or
    wholly - outside the room rectangle, which Celeste's editor allows.

    The clamp is asymmetric on purpose. At the low edge, a point entity shifted to
    `-1` has a box straddling the room border, so tile `0` really is covered and
    clamping recovers it. At the high edge, a start past `limit` means the whole
    box is beyond the room and there is nothing to recover.

    Args:
        origin_px: the box's origin along this axis, in pixels.
        size_px: its extent in pixels, or `None` for a point entity.
        limit: the room's size along this axis, in tiles.

    Returns:
        The `(start, stop)` tile span, half-open, or `None` if it covers nothing.
    """
    start: int = math.floor(origin_px / TILE_SIZE)
    stop: int
    if size_px is None:
        start = max(start - POINT_ENTITY_SHIFT, 0)
        stop = start + 1
    else:
        stop = start + math.floor(size_px / TILE_SIZE)

    low: int = max(start, 0)
    high: int = min(stop, limit)

    return None if high <= low else (low, high)


def stamp_entity(df_room: DataFrame, entity: MappedEntity) -> bool:
    """Writes an entity's symbol over every tile it covers, in place.

    Args:
        df_room: the room being built up, modified in place.
        entity: the entity to stamp.

    Returns:
        Whether anything was written. `False` means the entity lies outside the
        room and was skipped.
    """
    height, width = df_room.shape
    columns: tuple[int, int] | None = _tile_span(
        origin_px=entity.x, size_px=entity.width, limit=width
    )
    rows: tuple[int, int] | None = _tile_span(
        origin_px=entity.y, size_px=entity.height, limit=height
    )
    if columns is None or rows is None:
        return False

    df_room.iloc[rows[0] : rows[1], columns[0] : columns[1]] = entity.symbol

    return True


def integrate_entities_to_fg(
    df_fg: DataFrame, entities: list[MappedEntity]
) -> tuple[DataFrame, int]:
    """Stamps every entity onto a copy of a room's foreground tiles.

    Args:
        df_fg: the collapsed foreground tiles. Never modified.
        entities: the entities to stamp, in the order they should be applied.

    Returns:
        The recombined room, and how many entities fell outside it.
    """
    df_recombined: DataFrame = df_fg.copy()
    n_outside: int = 0
    for entity in entities:
        if not stamp_entity(df_room=df_recombined, entity=entity):
            n_outside += 1

    return df_recombined, n_outside


def discover_room_pairs(levels_dir: Path) -> list[tuple[Path, Path]]:
    """Finds every room in a levels tree, as a foreground/entities pair.

    A foreground csv with no entities json beside it is skipped with a warning
    rather than processed: a room written without its entities would be stripped
    of every hazard, which teaches the model that safe rooms are common - worse
    than the room simply being absent.

    Args:
        levels_dir: the directory holding one subdirectory per chapter.

    Returns:
        The `(foreground, entities)` pairs, sorted by foreground path.

    Raises:
        ValueError: the directory does not exist or holds no foreground csv.
    """
    if not levels_dir.is_dir():
        raise ValueError(
            f"Levels directory {levels_dir} does not exist. Extract the game data "
            f"there first - see the data loader in `seleCte/data_loader`."
        )

    pairs: list[tuple[Path, Path]] = []
    fg_paths: list[Path] = sorted(levels_dir.glob(pattern=f"*/*{FG_SUFFIX}"))
    for fg_path in fg_paths:
        entities_path: Path = fg_path.with_name(
            name=fg_path.name.removesuffix(FG_SUFFIX) + ENTITIES_SUFFIX
        )
        if not entities_path.is_file():
            logger.warning(
                msg=f"Skipping {fg_path}: no entities file at {entities_path}."
            )
            continue
        pairs.append((fg_path, entities_path))

    if not fg_paths:
        raise ValueError(
            f"No `*{FG_SUFFIX}` found under {levels_dir}. Expected one "
            f"subdirectory per chapter, each holding the rooms of that chapter."
        )

    return pairs


def _load_entities(path: Path) -> list[dict[str, Any]]:
    """Reads one room's entities json.

    Raises:
        ValueError: the file does not hold a list of entities.
    """
    with path.open(encoding="utf-8") as file:
        payload: Any = json.load(fp=file)

    if not isinstance(payload, list):
        raise ValueError(
            f"Entities file {path} holds {type(payload).__name__}, expected a list."
        )

    return payload


def recombine_room(
    fg_path: Path,
    entities_path: Path,
    entity_map: Mapping[str, str] = DEFAULT_ENTITY_MAP,
) -> tuple[DataFrame | None, RoomReport]:
    """Folds one room's foreground tiles and entities into a single grid.

    Args:
        fg_path: the room's foreground csv.
        entities_path: its entities json.
        entity_map: entity names to the symbol they stamp.

    Returns:
        The recombined room and its report, or `(None, report)` when the room
        holds tiles outside the vocabulary and has to be skipped.
    """
    # Order matters: the validity check has to run on the collapsed tiles but
    # before any entity is stamped, or every entity symbol would be reported as an
    # unrecognized tile.
    df_fg: DataFrame = collapse_fg_tiles(df_fg=read_room_csv(path=fg_path))
    invalid: dict[str, int] = invalid_symbol_counts(df_room=df_fg)
    entities: list[MappedEntity] = filter_entities(
        entities=_load_entities(path=entities_path), entity_map=entity_map
    )
    if invalid:
        return None, RoomReport(
            fg_path=fg_path,
            n_entities_kept=len(entities),
            n_entities_outside=0,
            invalid_symbols=invalid,
        )

    df_room, n_outside = integrate_entities_to_fg(df_fg=df_fg, entities=entities)

    return df_room, RoomReport(
        fg_path=fg_path,
        n_entities_kept=len(entities),
        n_entities_outside=n_outside,
        invalid_symbols={},
    )


def main(args: Args) -> None:
    """Recombines every room under the levels directory into the corpus.

    Rooms holding symbols outside the vocabulary are skipped with a warning rather
    than aborting the batch - one unreadable room should not cost the other 823 -
    but a malformed entities file or an entity without coordinates is a hard
    error, since that is corrupt input rather than input this stage cannot model.

    Args:
        args: parsed arguments, see `Args`.

    Raises:
        ValueError: two chapters produce the same output file name.
    """
    levels_dir: Path = Path(args.levels_dir)
    output_dir: Path = Path(args.output_dir)

    pairs: list[tuple[Path, Path]] = discover_room_pairs(levels_dir=levels_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_written: int = 0
    n_changed: int = 0
    n_entities: int = 0
    n_outside: int = 0
    skipped_invalid: list[RoomReport] = []
    seen_names: dict[str, Path] = {}

    for fg_path, entities_path in pairs:
        name: str = output_filename(room_path=fg_path)
        if name in seen_names:
            raise ValueError(
                f"{fg_path} and {seen_names[name]} both map to {name!r}. Chapter "
                f"directories need distinct ids before the dash."
            )
        seen_names[name] = fg_path

        df_room, report = recombine_room(fg_path=fg_path, entities_path=entities_path)
        if df_room is None:
            total_invalid: int = sum(report.invalid_symbols.values())
            logger.warning(
                msg=f"Skipping {fg_path}: {total_invalid} tiles outside the "
                f"vocabulary {report.invalid_symbols}. Add them to "
                f"`valid_fg_tiles` in `tiles_info.json`, or fix the map."
            )
            skipped_invalid.append(report)
            continue

        n_entities += report.n_entities_kept
        n_outside += report.n_entities_outside
        n_written += 1

        target: Path = output_dir / name
        if args.dry_run:
            # The point of a dry run is to answer "would this change the corpus?",
            # so compare against what is already on disk rather than just counting.
            if not target.is_file() or target.read_text(
                encoding="utf-8"
            ) != df_room.to_csv(header=False, index=False, sep=";"):
                n_changed += 1
            continue

        write_room_csv(df_room=df_room, path=target)
        logger.debug(msg=f"Recombined {fg_path} into {target}.")

    logger.info(
        msg=f"Recombined {n_written}/{len(pairs)} rooms "
        f"{'(dry run, nothing written)' if args.dry_run else f'into {output_dir}'}: "
        f"{len(skipped_invalid)} skipped for unknown tiles, {n_entities} entities "
        f"mapped, {n_outside} of them outside room bounds."
    )
    if args.dry_run:
        logger.info(msg=f"{n_changed} file(s) would change in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels-dir",
        "-ld",
        required=False,
        type=str,
        help="Directory holding one subdirectory of rooms per chapter",
        default=str(LEVELS_DIR),
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        required=False,
        type=str,
        help="Where the recombined rooms are written",
        default=str(ALL_ROOMS_DIR),
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Recombine everything but write nothing, reporting what would change",
    )
    args: Args = parser.parse_args(namespace=Args())

    main(args)
