from seleCte.celeskeleton.celeskeleton import Celeskeleton, Room


def build_two_rooms_skeleton() -> Celeskeleton:
    """Two 10x10 rooms side by side, connected by one exit each."""
    skeleton = Celeskeleton()
    left_room = Room(width=10, height=10, origin_x=0, origin_y=0)
    right_room = Room(width=10, height=10, origin_x=10, origin_y=0)
    left_room.add_exit(side="right", c1=2, c2=5)
    right_room.add_exit(side="left", c1=2, c2=5)
    skeleton.add_room(left_room, "room_1")
    skeleton.add_room(right_room, "room_2")
    skeleton.set_start_end()

    return skeleton


def test_get_special_rooms():
    skeleton = build_two_rooms_skeleton()

    assert skeleton.get_special_rooms() == {"room_1": "start", "room_2": "end"}


def test_get_special_rooms_empty_skeleton():
    assert Celeskeleton().get_special_rooms() == {}


def test_show_skeleton_draws_rooms_exits_and_names():
    skeleton = build_two_rooms_skeleton()

    ax = skeleton.show_skeleton(show=False)

    assert len(ax.patches) == 2  # one rectangle per room
    assert len(ax.lines) == 2  # one segment per exit
    assert {text.get_text() for text in ax.texts} == {"room_1", "room_2"}
    assert ax.get_xlim()[0] < 0 and ax.get_xlim()[1] > 20
    assert ax.get_aspect() == 1.0


def test_show_skeleton_without_names():
    ax = build_two_rooms_skeleton().show_skeleton(show_names=False, show=False)

    assert not ax.texts


def test_show_skeleton_saves_figure(tmp_path):
    output_file = tmp_path / "skeleton.png"

    build_two_rooms_skeleton().show_skeleton(save_path=output_file, show=False)

    assert output_file.exists() and output_file.stat().st_size > 0


def test_show_skeleton_empty_skeleton():
    ax = Celeskeleton().show_skeleton(show=False)

    assert not ax.patches
