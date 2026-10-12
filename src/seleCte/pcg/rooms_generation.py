"""Samples rooms straight from a DPE, with no playability guarantees.

Run as a script, this is the experiment harness for the generator: it writes
whatever the sampler produces, so the output is meant to be inspected and
scored rather than played. `level_generation` is the pipeline that turns the
same sampling step into a finished level, wrapping it in the A* playability
check, the special-point placement and the retry loops this module omits.

Two experiment types, selected by `--experiment-type`:

- `rooms` samples `--nb-rooms` independent rooms and writes each as its own
  semicolon-separated csv. No skeleton, so no exits and no connectivity - use
  this to look at raw tile output for a given dict;
- `skel` draws a `Celeskeleton`, fills every room in it, then carves the exits
  the skeleton assigned and adds respawn points. The rooms connect, but
  nothing checks that any of them can actually be crossed.

Unlike `level_generation`, the dict is a CLI argument (`--dict-proba-estimation`),
which is what makes this the module to reach for when comparing DPEs trained
on different corpora or neighbourhood masks.

`--layer` selects which tile layer is being sampled, and has to agree with the
dict: a background dict paired with the foreground vocabulary would fill unseen
patterns with spikes. Background sampling is `rooms`-only - a skeleton carves
exits and places respawn points, neither of which means anything on a decorative
layer - so for now `--layer bg` produces exactly one thing, a `height x width`
array of background symbols per room.
"""

import argparse
import logging
from typing import Any

import numpy as np

from seleCte.celeskeleton import Celeskeleton, generate_celeskeleton
from seleCte.paths import PE_DICTS_DIR
from seleCte.utils import (
    ALL_TILES_AND_ENTITIES,
    VALID_BG_TILES,
    generate_room_batch,
    generate_room_data,
    load_dpe,
)

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

# The vocabulary an unseen pattern falls back on, and the symbol the empty room is
# seeded with, per layer. The foreground border is the wall a room is enclosed by;
# a background layer has no walls, so it seeds with void.
LAYER_SAMPLING: dict[str, tuple[list[str], str]] = {
    "fg": (ALL_TILES_AND_ENTITIES, "1"),
    "bg": (VALID_BG_TILES, "0"),
}


class Args:
    """Parsed command-line arguments, one attribute per `add_argument` call.

    Same rationale as `dpe_generation.Args`: a plain class rather than an
    `argparse.Namespace` subclass, so a misspelled attribute is a type error
    instead of an `Any` that only blows up mid-run.
    """

    nb_rooms: int
    experiment_type: str
    room_size: tuple[int, int]
    bt_depth: int
    proba: float
    output_dir: str
    dict_proba_estimation: str
    layer: str
    verbose: bool


def main(args: Args) -> None:
    """Samples rooms into `args.output_dir`, standalone or inside a skeleton.

    In `rooms` mode the rooms are independent: each is sampled at the
    requested size and written as its own csv, with nothing relating one to
    the next. In `skel` mode a skeleton is drawn first and each of its rooms
    is filled in place, so the exits line up between neighbours and the whole
    level is saved as one `Celeskeleton`.

    Neither mode validates what it samples. There is no playability check and
    no retry, so a room that is unreachable or unsolvable is written out like
    any other - that is the point of the harness, since discarding bad rooms
    would bias any measurement taken over the results.

    Args:
        args: parsed arguments, see `Args`.

    Raises:
        ValueError: a background layer was asked for in `skel` mode, which has
            no meaning: exits and respawn points are foreground concepts.
    """
    btd: int = args.bt_depth
    nb_rooms: int = args.nb_rooms
    width, height = args.room_size
    all_symbols, border = LAYER_SAMPLING[args.layer]

    d_proba_estimation: dict[str, dict[str, int | float]]
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]]
    d_proba_estimation, mdmc_matrix = load_dpe(
        path=PE_DICTS_DIR / f"{args.dict_proba_estimation}.json"
    )

    if args.experiment_type == "rooms":
        generate_room_batch(
            n_rooms=nb_rooms,
            output_dir=args.output_dir,
            d_proba_estimation=d_proba_estimation,
            width=width,
            height=height,
            backtracking_depth=btd,
            verbose=args.verbose,
            mdmc_matrix=mdmc_matrix,
            all_symbols=all_symbols,
            border=border,
        )

    elif args.layer != "fg":
        raise ValueError(
            f"--experiment-type skel is foreground-only, got --layer {args.layer!r}. "
            "A skeleton carves exits and places respawn points, which a decorative "
            "layer has none of; how the two layers are stitched into one level is "
            "not decided yet. Use --experiment-type rooms."
        )

    else:
        level_skeleton: Celeskeleton = generate_celeskeleton(
            nb_rooms, p=args.proba, room_size=(width, height)
        )
        for n_room in range(1, nb_rooms + 1):
            level_skeleton.lvl[f"room_{n_room}"].data = generate_room_data(
                d_proba_estimation=d_proba_estimation,
                room_size=(width, height),
                backtracking_depth=btd,
                mdmc_matrix=mdmc_matrix,
            )
            level_skeleton.lvl[f"room_{n_room}"].create_exits_in_matrix()
            level_skeleton.lvl[f"room_{n_room}"].add_respawn_points()
        level_skeleton.save(output_dir=args.output_dir)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb-rooms", "-nr", required=True, type=int, help="Number of rooms to generate"
    )

    parser.add_argument(
        "--experiment-type",
        "-et",
        required=True,
        choices=["rooms", "skel"],
        type=str,
        help="Rooms only or skeleton integration",
    )

    parser.add_argument(
        "--room-size",
        "-rs",
        nargs="+",
        required=False,
        type=int,
        help="Room size as two separate numbers, width then height",
        default=[40, 23],
    )

    parser.add_argument(
        "--bt-depth",
        "-btd",
        required=False,
        type=int,
        help="Depth of backtracking used in the room generation function",
        default=0,
    )

    parser.add_argument(
        "--proba",
        "-p",
        required=False,
        type=float,
        help="Labyrinth proba: 0 for a pathway level, 1 for a completely random order",
        default=0.2,
    )

    parser.add_argument(
        "--output-dir",
        "-od",
        required=False,
        type=str,
        help="Output directory where generated files will be saved",
        default="default_generated_test_rooms",
    )

    parser.add_argument(
        "--dict-proba-estimation",
        "-dpe",
        required=False,
        type=str,
        help="Name of the PE dictionary to use",
        default="pe_d5_full",
    )

    parser.add_argument(
        "--layer",
        "-l",
        required=False,
        choices=sorted(LAYER_SAMPLING),
        type=str,
        help="Tile layer being sampled. Has to match the layer --dict-proba-"
        "estimation was trained on; bg is --experiment-type rooms only",
        default="fg",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args: Args = parser.parse_args(namespace=Args())

    main(args)
