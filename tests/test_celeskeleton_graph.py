from itertools import pairwise

import numpy as np
import pytest

from seleCte.celeskeleton import generate_celeskeleton
from seleCte.celeskeleton.celeskeleton import Celeskeleton, Room


def build_branching_skeleton() -> Celeskeleton:
    """A chain of three rooms, plus a fourth branching off the first.

    Insertion order makes room_1 the start and room_4 the end, so the traversal
    is room_1 -> room_4 and the two rooms of the chain sit off it::

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


def test_is_connected_to_recognises_a_shared_opening():
    skeleton = build_branching_skeleton()

    assert skeleton.get_room_by_name("room_1").is_connected_to(
        room=skeleton.get_room_by_name("room_2")
    )


def test_is_connected_to_is_symmetric():
    skeleton = build_branching_skeleton()
    room_1 = skeleton.get_room_by_name("room_1")
    room_2 = skeleton.get_room_by_name("room_2")

    assert room_1.is_connected_to(room=room_2) == room_2.is_connected_to(room=room_1)


def test_touching_rooms_without_a_shared_exit_are_not_connected():
    # Juxtaposed, and each with an exit on the facing side, but at coordinates
    # that do not line up - which is what makes this different from adjacency.
    left = Room(width=10, height=10, origin_x=0, origin_y=0)
    right = Room(width=10, height=10, origin_x=10, origin_y=0)
    left.add_exit(side="right", c1=2, c2=5)
    right.add_exit(side="left", c1=6, c2=8)

    assert not left.is_connected_to(room=right)


def test_rooms_with_matching_exits_but_apart_are_not_connected():
    # Same exit segment on the facing sides, but the borders do not coincide.
    left = Room(width=10, height=10, origin_x=0, origin_y=0)
    far = Room(width=10, height=10, origin_x=50, origin_y=0)
    left.add_exit(side="right", c1=2, c2=5)
    far.add_exit(side="left", c1=2, c2=5)

    assert not left.is_connected_to(room=far)


def test_faces_rejects_an_unknown_side():
    room = Room(width=10, height=10)

    with pytest.raises(ValueError, match="Select a side"):
        room.faces(Room(width=10, height=10), side="sideways")


def test_get_room_graph_lists_every_room():
    graph = build_branching_skeleton().get_room_graph()

    assert set(graph) == {"room_1", "room_2", "room_3", "room_4"}
    assert graph["room_1"] == {"room_2", "room_4"}
    assert graph["room_2"] == {"room_1", "room_3"}
    assert graph["room_3"] == {"room_2"}
    assert graph["room_4"] == {"room_1"}


def test_get_room_graph_keeps_unconnected_rooms():
    skeleton = Celeskeleton()
    skeleton.add_room(Room(width=10, height=10, origin_x=0, origin_y=0), "room_1")
    skeleton.add_room(Room(width=10, height=10, origin_x=50, origin_y=50), "room_2")

    assert skeleton.get_room_graph() == {"room_1": set(), "room_2": set()}


def test_get_traversal_path_skips_rooms_off_the_route():
    path = build_branching_skeleton().get_traversal_path()

    assert path == ["room_1", "room_4"]


def test_get_traversal_path_raises_on_a_disconnected_skeleton():
    skeleton = Celeskeleton()
    skeleton.add_room(Room(width=10, height=10, origin_x=0, origin_y=0), "room_1")
    skeleton.add_room(Room(width=10, height=10, origin_x=50, origin_y=50), "room_2")
    skeleton.set_start_end()

    with pytest.raises(ValueError, match="not traversable"):
        skeleton.get_traversal_path()


@pytest.mark.parametrize("proba", [0.0, 0.3, 0.6])
def test_generated_skeletons_have_a_connected_traversal_path(proba):
    # The property that matters: whatever `p` does to room numbering, the path
    # is a real chain of connected rooms running from the start to the end.
    for seed in range(5):
        np.random.seed(seed)
        skeleton = generate_celeskeleton(8, p=proba, room_size=(40, 23))

        path = skeleton.get_traversal_path()
        graph = skeleton.get_room_graph()

        assert path[0] == skeleton.get_starting_room()
        assert path[-1] == skeleton.get_ending_room()
        assert len(set(path)) == len(path)
        for current, following in pairwise(path):
            assert following in graph[current]


def test_generated_skeletons_are_trees():
    # n rooms attached one at a time to an existing room gives n-1 connections.
    np.random.seed(0)
    skeleton = generate_celeskeleton(10, p=0.5, room_size=(40, 23))

    graph = skeleton.get_room_graph()
    nb_connections = sum(len(neighbours) for neighbours in graph.values()) // 2

    assert nb_connections == skeleton.get_nb_rooms() - 1


def test_traversal_path_is_not_room_numbering_in_a_labyrinth():
    # The reason this API exists: with p > 0 a new room attaches to a random
    # earlier one, so the numbering is insertion order rather than the route.
    off_numbering = 0
    for seed in range(20):
        np.random.seed(seed)
        skeleton = generate_celeskeleton(8, p=0.6, room_size=(40, 23))
        if skeleton.get_traversal_path() != skeleton.list_all_rooms():
            off_numbering += 1

    assert off_numbering > 0
