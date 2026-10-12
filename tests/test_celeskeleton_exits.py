"""Exit coordinates: absolute in the skeleton, local in a room's tile array.

Every test here fails on the code as it stood before `_local_row`/`_local_col`
existed. The conversion used to be open-coded three times, with three different
formulas - `get_starting_ending_points` added the origin where it should have
subtracted it, `create_exits_in_matrix` subtracted without flipping the axis,
and `add_respawn_points` did a third thing. Nothing agreed except by accident at
`origin == (0, 0)`, which is why only `room_1` ever worked.
"""

import numpy as np
import pytest

from seleCte.celeskeleton.celeskeleton import Room

WIDTH, HEIGHT = 40, 23

# Rooms at the origin are the case the old code got right by accident, so every
# test sweeps origins that are not (0, 0) as well.
ORIGINS = [(0, 0), (0, 23), (100, 50), (-60, -46)]
SIDES = ["up", "down", "left", "right"]


def walled_room(origin: tuple[int, int]) -> Room:
    """A room enclosed by solid tiles, with void inside - no exits carved yet."""
    room = Room(width=WIDTH, height=HEIGHT, origin_x=origin[0], origin_y=origin[1])
    data = np.full((HEIGHT, WIDTH), "0", dtype="<U1")
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = "1"
    room.set_data(data)
    return room


def exit_span(room: Room, side: str) -> tuple[int, int]:
    """A five-tile exit span near the middle of `side`, in absolute coordinates."""
    axis = 0 if side in ("up", "down") else 1
    start = room.get_origin()[axis] + 10
    return start, start + 4


def open_cells(room: Room, side: str) -> list[int]:
    """Indices along the relevant border that are void, i.e. part of an opening."""
    if side == "left":
        return [r for r in range(HEIGHT) if room.data[r, 0] == "0"]
    if side == "right":
        return [r for r in range(HEIGHT) if room.data[r, -1] == "0"]
    if side == "up":
        return [c for c in range(WIDTH) if room.data[0, c] == "0"]
    return [c for c in range(WIDTH) if room.data[-1, c] == "0"]


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("side", SIDES)
def test_exit_point_lands_on_the_opening(side, origin):
    """The regression test for the hang.

    `is_valid_special` searches from a point to each of these, so a point that
    misses the hole makes the search fail for every candidate, forever.
    """
    room = walled_room(origin)
    c1, c2 = exit_span(room, side)
    room.add_exit(side, c1=c1, c2=c2)
    room.create_exits_in_matrix()

    (point,) = room.get_starting_ending_points()
    along = point[0] if side in ("left", "right") else point[1]

    assert along in open_cells(room, side)


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("side", SIDES)
def test_exit_points_are_inside_the_room(side, origin):
    """An out-of-bounds point does not raise - A* just never finds it."""
    room = walled_room(origin)
    c1, c2 = exit_span(room, side)
    room.add_exit(side, c1=c1, c2=c2)

    (row, column) = room.get_starting_ending_points()[0]

    assert 0 <= row < HEIGHT
    assert 0 <= column < WIDTH


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("side", SIDES)
def test_the_opening_is_placed_the_same_wherever_the_room_sits(side, origin):
    """A room's own geometry cannot depend on where the skeleton put it."""
    at_origin = walled_room((0, 0))
    moved = walled_room(origin)
    for room in (at_origin, moved):
        c1, c2 = exit_span(room, side)
        room.add_exit(side, c1=c1, c2=c2)
        room.create_exits_in_matrix()

    assert open_cells(moved, side) == open_cells(at_origin, side)


def absolute_height_of_row(room: Room, row: int) -> int:
    """The skeleton y coordinate a row of `data` sits at.

    Row 0 is the room's top edge, at `origin_y + height`, and rows count down
    from there - the inverse of `Room._local_row`.
    """
    return room.get_origin()[1] + room.get_size()[1] - row


def test_one_opening_is_at_the_same_height_in_both_rooms_sharing_it():
    """Two rooms juxtaposed left-to-right share one opening, at one height.

    `connect` writes the same absolute span into both rooms, so if the two
    disagree about which row that is, the player walks into a wall. The rooms
    below differ in origin *and* height, which is what the old formula could not
    survive: it made the row a function of both.
    """
    left = Room(width=WIDTH, height=23, origin_x=0, origin_y=50)
    right = Room(width=WIDTH, height=18, origin_x=WIDTH, origin_y=55)
    left.add_exit("right", c1=60, c2=64)
    right.add_exit("left", c1=60, c2=64)

    (left_point,) = left.get_starting_ending_points()
    (right_point,) = right.get_starting_ending_points()

    assert absolute_height_of_row(left, left_point[0]) == absolute_height_of_row(
        right, right_point[0]
    )


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("side", SIDES)
def test_respawn_point_is_placed_against_its_own_opening(side, origin):
    """The spawn is how the player enters, so it belongs beside the hole."""
    room = walled_room(origin)
    c1, c2 = exit_span(room, side)
    room.add_exit(side, c1=c1, c2=c2)
    room.create_exits_in_matrix()
    room.add_respawn_points()

    spawns = list(zip(*np.where(room.data == "P"), strict=True))
    assert len(spawns) == 1
    row, column = spawns[0]
    along = row if side in ("left", "right") else column

    assert along in open_cells(room, side)


def test_respawn_clearing_never_writes_through_the_opposite_wall():
    """A spawn on row 0 must not clear tiles off the bottom of the room."""
    room = walled_room((0, 0))
    # The topmost legal left exit, which `_local_row` maps to row 0.
    room.add_exit("left", c1=HEIGHT - 5, c2=HEIGHT - 1)
    room.create_exits_in_matrix()
    room.add_respawn_points()

    assert (room.data[-1, 1:-1] == "1").all()
