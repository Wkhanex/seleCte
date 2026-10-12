"""Generates a complete level: skeleton geometry filled with sampled tiles.

Run as a script, this is the online generation step of the PCG pipeline and
the counterpart to `dpe_generation`: where that module trains a probability
dict from a corpus of existing levels, this one samples a new level back out
of it. A run draws a `Celeskeleton` for the coarse room layout, fills each
room with tiles drawn from the DPE, and writes the result to `{level_name}/`.
The dict itself is not a CLI argument - `pe_d5_full.json` is hardcoded.

A sampled room is not guaranteed to be usable, so generation is retried at
three nested levels:

- special points: a start or end room needs somewhere to put the player spawn
  or the crystal heart, so its tiles are resampled until placement succeeds;
- room: a filled room must be traversable, checked by A*, and is resampled up
  to `--tries-limit` times;
- skeleton: a room that never becomes playable usually means its exits are
  badly placed, which no amount of resampling fixes - so with
  `--reset-skeleton` the whole level is discarded and redrawn.

Without `--reset-skeleton` that last level of retry is skipped and the
unplayable room is kept, which lets a run terminate at the cost of a level
that may not be completable.

Every one of those loops is bounded - see `MAX_ROOM_RESAMPLES` and
`MAX_SKELETON_REDRAWS`. Exhausting a bound keeps the last attempt and logs a
warning rather than raising, which is the same trade the paragraph above
describes: a run always terminates, and says so when what it wrote is suspect.

Only the rooms on the skeleton's traversal path are held to the playability
check. `generate_celeskeleton` grows a level as a tree - each new room attached
to one existing room - so there is exactly one route from the starting room to
the ending room, and every room off it is a branch the player never has to
cross to finish. Gating those would discard levels over rooms that cannot make
a level uncompletable. They are still filled, once, so the level is whole; they
are just not resampled until they happen to be traversable.
"""

import argparse
import logging
from typing import Any

import numpy as np

from seleCte.celeskeleton import generate_celeskeleton
from seleCte.celeskeleton.celeskeleton import Celeskeleton, Room
from seleCte.paths import PE_DICTS_DIR
from seleCte.utils import generate_temp_room, load_dpe

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_ROOM_SIZE: list[int] = [40, 23]

# How many times a room is resampled while trying to place its special point, and
# how many whole skeletons are drawn, before the run gives up and writes what it
# has. Both loops were unbounded, which is how a room that could never satisfy a
# check turned into a hang rather than a reported failure. The numbers are
# deliberately generous: they are a backstop against spinning forever, not a
# tuning knob, and a run that reaches either has hit something worth a warning.
MAX_ROOM_RESAMPLES: int = 10
MAX_SKELETON_REDRAWS: int = 10


class Args:
    """Parsed command-line arguments, one attribute per `add_argument` call.

    Same rationale as `dpe_generation.Args`: a plain class rather than an
    `argparse.Namespace` subclass, so a misspelled attribute is a type error
    instead of an `Any` that only blows up mid-run.
    """

    nb_rooms: int
    proba: float
    room_size: tuple[int, int]
    bt_depth: int
    level_name: str
    tries_limit: int
    reset_skeleton: bool


def fill_room(
    room: Room,
    d_proba_estimation: dict[str, dict[str, int | float]],
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
    args: Args,
) -> tuple[Room, bool]:
    """Samples one room's tiles, resampling until its special point fits.

    The room's role comes from the skeleton rather than from its position in
    the level: `set_start_end` is what marks the starting and ending rooms, and
    `add_special_points` is a no-op on every other room, returning `True`
    immediately. So only a start or end room can loop here.

    Resampling stops at `MAX_ROOM_RESAMPLES`. `add_special_points` already caps
    its own attempts and reports failure, and this loop used to discard that by
    retrying forever - which turned a room that could never satisfy the check
    into a hang rather than into a warning.

    Args:
        room: the skeleton room to fill, read for its exits, origin and status.
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        mdmc_matrix: the mask `d_proba_estimation` was trained with.
        args: parsed arguments, see `Args`.

    Returns:
        A filled room detached from the skeleton - the caller writes its data
        back - and whether its special point was placed. `False` means the room
        is missing its player spawn or its crystal heart, so a level keeping it
        cannot be started or cannot be finished.
    """
    temp_room: Room = room
    for attempt in range(1, MAX_ROOM_RESAMPLES + 1):
        temp_room = generate_temp_room(
            d_exits=room.get_exits(),
            d_proba_estimation=d_proba_estimation,
            backtracking_depth=args.bt_depth,
            room_size=args.room_size,
            origin=tuple(room.get_origin()),
            status=room.get_status(),
            mdmc_matrix=mdmc_matrix,
        )
        if temp_room.add_special_points(nb_tries_limit=args.tries_limit):
            return temp_room, True
        logger.info(msg=f"Special point placement failed, resample {attempt}.")

    logger.warning(
        msg=f"No special point placed in {MAX_ROOM_RESAMPLES} resamples - keeping "
        "the last attempt. `add_special_points` needs one random point to reach "
        "every exit, which is a demanding check run through a slow search."
    )

    return temp_room, False


def fill_traversal_path(
    lvl_skeleton: Celeskeleton,
    traversal_path: list[str],
    d_proba_estimation: dict[str, dict[str, int | float]],
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
    args: Args,
) -> bool:
    """Fills the rooms a run must cross, resampling each until it is playable.

    Each attempt resamples the room's tiles from the DPE, restores the exits and
    origin the skeleton assigned it, places the spawn point or crystal heart if
    the room is the first or last, and accepts the result only if A* finds a
    path across it. The most recent attempt is written back to the skeleton
    either way, so a room that exhausted `args.tries_limit` still holds tiles
    rather than being left empty.

    A room that never becomes playable usually means its exits are badly placed,
    which resampling cannot fix. With `args.reset_skeleton` the search stops
    there, since the caller is about to discard the skeleton and every room
    filled after it would be thrown away too. Without it the remaining rooms are
    still filled, which is what lets a run terminate at the cost of a level that
    may not be completable.

    Args:
        lvl_skeleton: the skeleton being filled, modified in place.
        traversal_path: the rooms to fill, in the order they are crossed.
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        mdmc_matrix: the mask `d_proba_estimation` was trained with.
        args: parsed arguments, see `Args`.

    Returns:
        Whether every room on the path came out playable.
    """
    all_playable = True

    for room_name in traversal_path:
        room: Room = lvl_skeleton.get_room_by_name(room_name=room_name)
        is_playable = False
        nb_tries = 0

        while not is_playable and nb_tries < args.tries_limit:
            nb_tries += 1
            logger.info(msg=f"{room_name}, try nb {nb_tries}")

            temp_room, special_placed = fill_room(
                room=room,
                d_proba_estimation=d_proba_estimation,
                mdmc_matrix=mdmc_matrix,
                args=args,
            )
            # A start or end room with no special point is unusable even if A*
            # can cross it, so it counts as a failed attempt like any other.
            is_playable = special_placed and temp_room.is_playable_room()
            room.set_data(temp_room.data)

        if not is_playable:
            all_playable = False
            logger.warning(msg=f"{room_name} generation failed in {nb_tries} tries.")
            if args.reset_skeleton:
                return False
        else:
            logger.info(msg=f"Generated {room_name} in {nb_tries} iterations.")

    return all_playable


def main(args: Args) -> None:
    """Draws and fills skeletons until the traversal path is playable, saves.

    The outer loop redraws the whole skeleton while rooms on the traversal path
    keep failing and `args.reset_skeleton` is set; without it a single pass is
    made and whatever came out is written, unplayable rooms included.

    Rooms off the traversal path are filled last and only once. They are
    branches the player never has to cross, so an unplayable one cannot make the
    level uncompletable, and holding them to the same check would throw away
    levels for free.

    Args:
        args: parsed arguments, see `Args`.
    """
    d_proba_estimation: dict[str, dict[str, int | float]]
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]]
    d_proba_estimation, mdmc_matrix = load_dpe(PE_DICTS_DIR / "pe_d5_full.json")

    level_name: str = args.level_name

    for redraw in range(1, MAX_SKELETON_REDRAWS + 1):
        logger.info(msg="Generating new level skeleton...")
        lvl_skeleton: Celeskeleton = generate_celeskeleton(
            args.nb_rooms, p=args.proba, room_size=args.room_size
        )
        traversal_path: list[str] = lvl_skeleton.get_traversal_path()
        logger.info(
            msg=f"Traversal path, {len(traversal_path)} of {args.nb_rooms} rooms: "
            f"{' -> '.join(traversal_path)}"
        )

        if fill_traversal_path(
            lvl_skeleton=lvl_skeleton,
            traversal_path=traversal_path,
            d_proba_estimation=d_proba_estimation,
            mdmc_matrix=mdmc_matrix,
            args=args,
        ):
            break
        if not args.reset_skeleton:
            logger.warning(msg="Pursuing generation with an unplayable room.\n")
            break
        if redraw == MAX_SKELETON_REDRAWS:
            logger.warning(
                msg=f"{MAX_SKELETON_REDRAWS} skeletons in a row had a room that "
                "never came out playable - keeping the last one. The level is "
                "very likely not completable.\n"
            )
            break
        logger.warning(msg="Re-generating a skeleton.\n")

    for room_name in lvl_skeleton.list_all_rooms():
        if room_name in traversal_path:
            continue
        logger.info(msg=f"Filling {room_name}, off the traversal path.")
        off_path_room: Room = lvl_skeleton.get_room_by_name(room_name=room_name)
        filled_room, _ = fill_room(
            room=off_path_room,
            d_proba_estimation=d_proba_estimation,
            mdmc_matrix=mdmc_matrix,
            args=args,
        )
        off_path_room.set_data(filled_room.data)

    lvl_skeleton.format_filled_celeskeleton()
    lvl_skeleton.save(output_dir=f"{level_name}")

    logger.info(
        msg=f"Celeskeleton object has been formatted and saved in folder {level_name}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb-rooms", "-nr", required=True, type=int, help="Number of rooms to generate"
    )

    parser.add_argument(
        "--proba",
        "-p",
        required=True,
        type=float,
        help="Labyrinth proba: 0 for a pathway level, 1 for a completely random order",
    )

    parser.add_argument(
        "--room-size",
        "-rs",
        nargs="+",
        required=False,
        type=int,
        default=DEFAULT_ROOM_SIZE,
        help="Room size as two separate numbers, width then height",
    )

    parser.add_argument(
        "--bt-depth",
        "-btd",
        required=False,
        type=int,
        default=0,
        help="Depth of backtracking used in the room generation function",
    )

    parser.add_argument(
        "--level-name",
        "-ln",
        required=False,
        type=str,
        default="default_generated_level",
        help="Name of the folder where generated files will be stored",
    )

    parser.add_argument(
        "--tries-limit",
        "-tl",
        required=False,
        type=int,
        default=10,
        help="Limit of tries per room gen",
    )

    parser.add_argument(
        "--reset-skeleton",
        "-r",
        action="store_true",
        help="Resetting skeleton if room gen fails",
    )

    args: Args = parser.parse_args(namespace=Args())

    main(args)
