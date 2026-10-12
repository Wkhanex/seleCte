"""Shared plumbing for turning extracted level data into a training corpus.

Both preprocessing stages - `recombine_fg_entities` for the foreground layer and
`collect_bg_rooms` for the background one - read the same csv dialect out of the
same directory layout, validate what they read against a vocabulary, and write
the result under the same naming contract. That common part lives here so the two
stages cannot drift apart on any of it.

The naming contract is the load-bearing piece. `dpe_generation` selects a training
corpus by dataset id and reads that id off the *first character* of each file
name, so `output_filename` prefixing the chapter id is what makes `--dataset 6`
mean "Reflection" rather than "whatever happened to sort first".
"""

from collections.abc import Collection
from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame

VOID_TILE: str = "0"
SOLID_TILE: str = "1"

# The recombined foreground alphabet, post-collapse and pre-stamping: every solid
# variant is already `"1"` by then, so anything else is a symbol the vocabulary does
# not know about.
COLLAPSED_FG_TILES: tuple[str, str] = (VOID_TILE, SOLID_TILE)


def read_room_csv(path: Path) -> DataFrame:
    """Reads a room csv as raw symbols, one per cell.

    `dtype=str` and `header=None` are both load-bearing: the grids hold single
    characters, several of which are digits that pandas would otherwise read as
    numbers, and the first row is data rather than a header.

    Args:
        path: the csv to read.

    Returns:
        The room, one symbol per cell.
    """
    return pd.read_csv(filepath_or_buffer=path, header=None, sep=";", dtype=str)


def write_room_csv(df_room: DataFrame, path: Path) -> None:
    """Writes a room csv in the shape the rest of the pipeline reads.

    Args:
        df_room: the room to write, one symbol per cell.
        path: where to write it.
    """
    df_room.to_csv(path_or_buf=path, header=False, index=False, sep=";")


def invalid_symbol_counts(
    df_room: DataFrame, valid: Collection[str] = COLLAPSED_FG_TILES
) -> dict[str, int]:
    """Counts the cells holding a symbol outside the vocabulary, by symbol.

    Args:
        df_room: a room's tiles. On the foreground side this has to be the
            collapsed grid, before any entity has been stamped onto it, or every
            entity symbol reads as an unrecognized tile.
        valid: the symbols the layer allows.

    Returns:
        `{symbol: occurrences}` for every unrecognized symbol; empty when the room
        is clean.
    """
    counts: dict[str, int] = {}
    for symbol, count in df_room.stack().value_counts().items():
        if symbol not in valid:
            counts[str(symbol)] = int(count)

    return counts


def output_filename(room_path: Path) -> str:
    """Names the corpus file a room's csv becomes.

    The chapter directory contributes its id - `1-ForsakenCity` gives `1`,
    `LostLevels` gives `LostLevels` - and the room file keeps its own name. Only
    the *directory* is split, because room names contain dashes too
    (`lvl_-1_fg.csv` has to survive intact).

    Args:
        room_path: a room's csv, inside its chapter directory.

    Returns:
        The corpus file name, e.g. `1_lvl_1_fg.csv`.

    Raises:
        ValueError: the chapter directory contributes no id.
    """
    chapter_id: str = room_path.parent.name.split(sep="-", maxsplit=1)[0]
    if not chapter_id:
        raise ValueError(
            f"Chapter directory {room_path.parent.name!r} yields no dataset id; "
            f"directories are expected to be named like `1-ForsakenCity`."
        )

    return f"{chapter_id}_{room_path.name}"
