"""Which rooms `level_generation` gates on playability, and which it does not."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from seleCte.celeskeleton.celeskeleton import Celeskeleton, Room
from seleCte.pcg.level_generation import (
    MAX_ROOM_RESAMPLES,
    MAX_SKELETON_REDRAWS,
    Args,
    main,
)

ROOM_SIZE = (10, 10)

# Every retry loop here is bounded, so the number of stubbed calls a run can make
# is too. Raising past this ceiling turns a regression that would hang the suite
# forever into a test that fails in milliseconds and names the reason.
CALL_CEILING = 1000

# Room origins double as room identity in the assertions below: the harness has
# no name to work with, but `fill_room` passes the skeleton room's origin down.
ROOM_1, ROOM_2, ROOM_3, ROOM_4 = (0, 0), (10, 0), (20, 0), (0, 10)


def branching_skeleton() -> Celeskeleton:
    """Four rooms whose traversal path is room_1 -> room_4.

    room_2 and room_3 hang off the chain, so they are branches the player never
    has to cross::

        room_4
        room_1  room_2  room_3
    """
    skeleton = Celeskeleton()
    room_1 = Room(width=10, height=10, origin_x=0, origin_y=0)
    room_2 = Room(width=10, height=10, origin_x=10, origin_y=0)
    room_3 = Room(width=10, height=10, origin_x=20, origin_y=0)
    room_4 = Room(width=10, height=10, origin_x=0, origin_y=10)

    room_1.add_exit(side="right", c1=2, c2=5)
    room_2.add_exit(side="left", c1=2, c2=5)
    room_2.add_exit(side="right", c1=3, c2=6)
    room_3.add_exit(side="left", c1=3, c2=6)
    room_1.add_exit(side="up", c1=3, c2=6)
    room_4.add_exit(side="down", c1=3, c2=6)

    for name, room in (
        ("room_1", room_1),
        ("room_2", room_2),
        ("room_3", room_3),
        ("room_4", room_4),
    ):
        skeleton.add_room(room, name)
    skeleton.set_start_end()

    return skeleton


class Harness:
    """Records what generation did, and decides what A* would have said."""

    def __init__(self) -> None:
        self.skeletons_drawn: int = 0
        self.checked: list[tuple[int, int]] = []
        # Called with the number of checks made so far, this one included.
        self.verdict: Callable[[int], bool] = lambda _: True
        self.special_point_calls: int = 0
        # Same shape, for `add_special_points`: placing the player spawn or the
        # crystal heart, which only start and end rooms need.
        self.special_point_verdict: Callable[[int], bool] = lambda _: True


@pytest.fixture
def harness(monkeypatch):
    """Replaces sampling, spawn placement and A* with instant, counted stand-ins."""
    recorder = Harness()

    def fake_generate_celeskeleton(*_args: Any, **_kwargs: Any) -> Celeskeleton:
        recorder.skeletons_drawn += 1
        return branching_skeleton()

    def fake_generate_temp_room(**kwargs: Any) -> Room:
        width, height = kwargs["room_size"]
        room = Room(width=width, height=height, status=kwargs["status"])
        # A symbol no border or exit carving writes, so a filled room stays
        # distinguishable from an untouched one after `format_filled_celeskeleton`.
        room.set_data(np.full((height, width), "S", dtype="<U1"))
        room.set_origin(x=kwargs["origin"][0], y=kwargs["origin"][1])
        return room

    def fake_is_playable_room(self: Room, *_args: Any, **_kwargs: Any) -> bool:
        recorder.checked.append(self.get_origin())
        return recorder.verdict(len(recorder.checked))

    monkeypatch.setattr(
        "seleCte.pcg.level_generation.generate_celeskeleton", fake_generate_celeskeleton
    )
    monkeypatch.setattr(
        "seleCte.pcg.level_generation.generate_temp_room", fake_generate_temp_room
    )

    def fake_add_special_points(_self: Room, *_args: Any, **_kwargs: Any) -> bool:
        recorder.special_point_calls += 1
        if recorder.special_point_calls > CALL_CEILING:
            raise AssertionError(
                f"add_special_points called more than {CALL_CEILING} times - "
                "a retry loop is unbounded again."
            )
        return recorder.special_point_verdict(recorder.special_point_calls)

    monkeypatch.setattr(Room, "is_playable_room", fake_is_playable_room)
    monkeypatch.setattr(Room, "add_special_points", fake_add_special_points)

    return recorder


def build_args(tmp_path, **overrides) -> Args:
    args = Args()
    args.nb_rooms = 4
    args.proba = 0.5
    args.room_size = ROOM_SIZE
    args.bt_depth = 0
    args.level_name = str(tmp_path / "level")
    args.tries_limit = 3
    args.reset_skeleton = False
    for name, value in overrides.items():
        setattr(args, name, value)

    return args


def test_only_the_traversal_path_rooms_are_checked(tmp_path, harness):
    """room_2 and room_3 are branches off the path, so nothing gates them."""
    main(build_args(tmp_path))

    assert harness.checked == [ROOM_1, ROOM_4]


@pytest.mark.usefixtures("harness")
def test_every_room_is_filled_including_the_branches(tmp_path):
    """Not being gated is not the same as being left empty - the level is whole."""
    args = build_args(tmp_path)

    main(args)

    for name in ("room_1", "room_2", "room_3", "room_4"):
        written = np.loadtxt(f"{args.level_name}/{name}.csv", dtype=str)
        assert set(np.unique(written[1:-1, 1:-1])) == {"S"}, name


def test_a_playable_path_never_redraws_the_skeleton(tmp_path, harness):
    main(build_args(tmp_path, reset_skeleton=True))

    assert harness.skeletons_drawn == 1


def test_a_failing_path_room_redraws_the_skeleton(tmp_path, harness):
    # The first skeleton's rooms all fail; from the fourth check on they pass.
    harness.verdict = lambda n: n > 3

    main(build_args(tmp_path, reset_skeleton=True))

    assert harness.skeletons_drawn > 1
    assert harness.checked[-2:] == [ROOM_1, ROOM_4]


def test_reset_skeleton_stops_filling_the_path_it_is_about_to_discard(
    tmp_path, harness
):
    # room_1 exhausts its budget, so room_4 is not attempted on that skeleton -
    # every room filled after the failure would be thrown away with it.
    harness.verdict = lambda n: n > 3

    main(build_args(tmp_path, tries_limit=3, reset_skeleton=True))

    assert harness.checked[:3] == [ROOM_1, ROOM_1, ROOM_1]


def test_without_reset_a_failed_room_does_not_stop_the_rest(tmp_path, harness):
    """Every path room still gets its full budget, and the level is written anyway."""
    harness.verdict = lambda _: False
    args = build_args(tmp_path, tries_limit=2, reset_skeleton=False)

    main(args)

    assert harness.skeletons_drawn == 1
    assert harness.checked == [ROOM_1, ROOM_1, ROOM_4, ROOM_4]
    assert (tmp_path / "level" / "data.json").exists()


def test_fill_room_gives_up_instead_of_resampling_forever(tmp_path, harness):
    """A special point that can never be placed must not spin the run.

    This is the hang: `add_special_points` caps itself and reports failure, and
    `fill_room` used to answer by resampling the room again, without limit.
    """
    harness.special_point_verdict = lambda _: False

    main(build_args(tmp_path, tries_limit=1, reset_skeleton=False))

    # One skeleton, one budgeted try for each of the two path rooms, plus the two
    # branch rooms filled once at the end - each resampling up to the cap.
    assert harness.special_point_calls == 4 * MAX_ROOM_RESAMPLES
    assert (tmp_path / "level" / "data.json").exists()


def test_skeleton_redraws_are_bounded(tmp_path, harness):
    """`--reset-skeleton` redraws on failure, and used to do so forever."""
    harness.special_point_verdict = lambda _: False

    main(build_args(tmp_path, tries_limit=1, reset_skeleton=True))

    assert harness.skeletons_drawn == MAX_SKELETON_REDRAWS
    assert (tmp_path / "level" / "data.json").exists()


def test_a_room_missing_its_special_point_is_not_accepted(tmp_path, harness):
    """No spawn means the level cannot be started, whatever A* thinks of it.

    A* is stubbed to approve everything here, so an accepted room would have to
    come from the special point being ignored.
    """
    harness.special_point_verdict = lambda _: False
    harness.verdict = lambda _: True

    main(build_args(tmp_path, tries_limit=1, reset_skeleton=True))

    # Short-circuited before A* ran: a room with no special point is already lost.
    assert harness.checked == []
    assert harness.skeletons_drawn == MAX_SKELETON_REDRAWS
