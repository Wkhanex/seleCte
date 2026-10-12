"""Builds the background corpus the MdMC model trains on, from the raw game data.

Run as a script, this is the background counterpart of `recombine_fg_entities`,
and it is mostly defined by what it does *not* have to do. A background layer
carries no entities, so there is no pixel geometry to convert, no point-entity
shift to correct and nothing to stamp; and its symbols are the signal rather than
decoration over one, so nothing is collapsed either. What is left is discovery,
validation and the rename that makes a file addressable by dataset id.

Which is to say: the game's `lvl_<id>_bg.csv` files are *already* the
representation the rest of the pipeline reads - a `(height, width)` grid holding
one symbol per cell. This stage copies them into `ALL_ROOMS_BG_DIR` under the
`{chapter_prefix}_{bg_filename}` name `dpe_generation` selects on, having checked
every symbol against `VALID_BG_TILES` on the way through.

**Keeping the alphabet is the point.** The foreground stage collapses every solid
variant to `"1"` because there the subtypes encode decoration and a Markov model
fed them would learn tileset transitions instead of level shape. On the background
layer there is no level shape to learn and the tileset transitions *are* the
thing, so all fifteen symbols survive.

**Train per chapter, not over the whole corpus.** A background tileset is a
property of the chapter, not of the room: 674 of the 824 rooms use exactly two
symbols, void plus one tileset, and that tileset is fixed per chapter
(`6-Reflection` is always `a`, `4-GoldenRidge` always `6`). Estimated across all
chapters at once, the dominant all-void pattern resolves to a mix of seven
tilesets that never co-occur in the game, and generation sprinkles them. Estimated
per chapter it resolves to the one correct tileset. So this stage keeps the
chapter id in every file name, and `dpe_generation --layer bg` is meant to be run
with a single-chapter `--dataset`.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pandas.core.frame import DataFrame

from seleCte.paths import ALL_ROOMS_BG_DIR, LEVELS_DIR
from seleCte.pcg.preprocessing.corpus import (
    invalid_symbol_counts,
    output_filename,
    read_room_csv,
    write_room_csv,
)
from seleCte.utils import VALID_BG_TILES

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

BG_SUFFIX: str = "_bg.csv"


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
class RoomReport:
    """What happened to one room, so `main` can summarize without re-reading."""

    bg_path: Path
    invalid_symbols: dict[str, int]


def discover_bg_rooms(levels_dir: Path) -> list[Path]:
    """Finds every background csv in a levels tree.

    Unlike the foreground stage there is nothing to pair a room with: the layer is
    self-contained, so a room is one file and a room with no entities json is not
    a problem.

    Args:
        levels_dir: the directory holding one subdirectory per chapter.

    Returns:
        The background csvs, sorted by path.

    Raises:
        ValueError: the directory does not exist or holds no background csv.
    """
    if not levels_dir.is_dir():
        raise ValueError(
            f"Levels directory {levels_dir} does not exist. Extract the game data "
            f"there first - see the data loader in `seleCte/data_loader`."
        )

    bg_paths: list[Path] = sorted(levels_dir.glob(pattern=f"*/*{BG_SUFFIX}"))
    if not bg_paths:
        raise ValueError(
            f"No `*{BG_SUFFIX}` found under {levels_dir}. Expected one "
            f"subdirectory per chapter, each holding the rooms of that chapter."
        )

    return bg_paths


def collect_room(bg_path: Path) -> tuple[DataFrame | None, RoomReport]:
    """Reads one background room and checks it against the vocabulary.

    Args:
        bg_path: the room's background csv.

    Returns:
        The room and its report, or `(None, report)` when the room holds symbols
        outside `VALID_BG_TILES` and has to be skipped.
    """
    df_bg: DataFrame = read_room_csv(path=bg_path)
    invalid: dict[str, int] = invalid_symbol_counts(df_room=df_bg, valid=VALID_BG_TILES)
    if invalid:
        return None, RoomReport(bg_path=bg_path, invalid_symbols=invalid)

    return df_bg, RoomReport(bg_path=bg_path, invalid_symbols={})


def main(args: Args) -> None:
    """Collects every background room under the levels directory into the corpus.

    Rooms holding symbols outside the vocabulary are skipped with a warning rather
    than aborting the batch, matching the foreground stage. On the vanilla data
    that branch never fires - all 824 rooms use exactly the fifteen symbols in
    `VALID_BG_TILES` - so it exists for modded maps, whose tilesets are the reason
    the check is worth keeping at all.

    Args:
        args: parsed arguments, see `Args`.

    Raises:
        ValueError: two chapters produce the same output file name.
    """
    levels_dir: Path = Path(args.levels_dir)
    output_dir: Path = Path(args.output_dir)

    bg_paths: list[Path] = discover_bg_rooms(levels_dir=levels_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_written: int = 0
    n_changed: int = 0
    skipped_invalid: list[RoomReport] = []
    seen_names: dict[str, Path] = {}

    for bg_path in bg_paths:
        name: str = output_filename(room_path=bg_path)
        if name in seen_names:
            raise ValueError(
                f"{bg_path} and {seen_names[name]} both map to {name!r}. Chapter "
                f"directories need distinct ids before the dash."
            )
        seen_names[name] = bg_path

        df_room, report = collect_room(bg_path=bg_path)
        if df_room is None:
            total_invalid: int = sum(report.invalid_symbols.values())
            logger.warning(
                msg=f"Skipping {bg_path}: {total_invalid} tiles outside the "
                f"vocabulary {report.invalid_symbols}. Add them to "
                f"`valid_bg_tiles` in `tiles_info.json`, or fix the map."
            )
            skipped_invalid.append(report)
            continue

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
        logger.debug(msg=f"Collected {bg_path} into {target}.")

    logger.info(
        msg=f"Collected {n_written}/{len(bg_paths)} background rooms "
        f"{'(dry run, nothing written)' if args.dry_run else f'into {output_dir}'}: "
        f"{len(skipped_invalid)} skipped for unknown tiles."
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
        help="Where the collected background rooms are written",
        default=str(ALL_ROOMS_BG_DIR),
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Collect everything but write nothing, reporting what would change",
    )
    args: Args = parser.parse_args(namespace=Args())

    main(args)
