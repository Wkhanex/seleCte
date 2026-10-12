import numpy as np
import pandas as pd
import pytest

from seleCte.pcg.rooms_generation import LAYER_SAMPLING, Args, main
from seleCte.utils import ALL_TILES_AND_ENTITIES, VALID_BG_TILES
from seleCte.utils.utils import (
    BG_PALETTE,
    FG_PALETTE,
    color_map,
    generate_empty_room_data,
    generate_new_tile,
    generate_room_data,
    update_room_array,
    visualize_room,
)

FG_ONLY_SYMBOLS = {"^", "<", "v", ">", "S", "D", "C", "_", "O", "Q", "W", "B", "R", "F"}


def uniform_dpe(vocabulary, pattern_length=3):
    """A DPE over one pattern only, so every other lookup is an unseen state."""
    seed = vocabulary[0] * pattern_length

    return {
        seed: {symbol: 1.0 if symbol == vocabulary[0] else 0.0 for symbol in vocabulary}
    }


# ------------------------------------------------------------- empty room seed


def test_generate_empty_room_data_borders_with_solid_by_default():
    room = generate_empty_room_data(width=5, height=4)

    assert set(room[0, :]) == {"1"}
    assert set(room[:, 0]) == {"1"}
    assert set(room[-1, :]) == {"1"}
    assert set(room[:, -1]) == {"1"}
    assert room[1, 1] == "0"


def test_generate_empty_room_data_takes_a_border_symbol():
    """bg has no walls, so it seeds with void instead."""
    room = generate_empty_room_data(width=5, height=4, border="0")

    assert set(room.ravel()) == {"0"}


def test_generate_empty_room_data_keeps_its_shape():
    assert generate_empty_room_data(width=40, height=23).shape == (23, 40)


# ------------------------------------------------------------------ vocabulary


def test_generate_new_tile_draws_from_the_given_vocabulary_when_unseen():
    """An empty distribution means an unseen pattern: the fallback draw."""
    drawn = {
        generate_new_tile(proba_dist={}, all_symbols=VALID_BG_TILES) for _ in range(200)
    }

    assert drawn <= set(VALID_BG_TILES)
    assert not drawn & FG_ONLY_SYMBOLS


def test_update_room_array_honours_the_vocabulary_on_an_unseen_pattern():
    """The regression this threading exists for: an unseen bg pattern used to
    fall back on the foreground vocabulary and could write a spinner."""
    room = generate_empty_room_data(width=6, height=5, border="0")

    for _ in range(50):
        update_room_array(
            array=room,
            x=0,
            y=0,
            d_proba_estimation={},
            verbose=False,
            all_symbols=VALID_BG_TILES,
        )
        assert room[2, 2] in VALID_BG_TILES


def test_generate_room_data_stays_inside_the_bg_vocabulary():
    room = generate_room_data(
        d_proba_estimation=uniform_dpe(VALID_BG_TILES),
        room_size=(12, 9),
        all_symbols=VALID_BG_TILES,
        border="0",
    )

    assert set(room.ravel()) <= set(VALID_BG_TILES)
    assert not set(room.ravel()) & FG_ONLY_SYMBOLS


def test_generate_room_data_defaults_to_the_fg_vocabulary():
    room = generate_room_data(
        d_proba_estimation=uniform_dpe(ALL_TILES_AND_ENTITIES), room_size=(12, 9)
    )

    assert set(room.ravel()) <= set(ALL_TILES_AND_ENTITIES)
    assert set(room[0, :]) == {"1"}


def test_generate_room_data_returns_height_by_width():
    room = generate_room_data(
        d_proba_estimation=uniform_dpe(VALID_BG_TILES),
        room_size=(40, 23),
        all_symbols=VALID_BG_TILES,
        border="0",
    )

    assert room.shape == (23, 40)


def test_generate_room_data_rejects_a_dict_trained_with_another_mask():
    with pytest.raises(ValueError, match="trained with a different mask"):
        generate_room_data(
            d_proba_estimation={"00": {"0": 1.0}},
            room_size=(8, 8),
            all_symbols=VALID_BG_TILES,
            border="0",
        )


# --------------------------------------------------------------------- palette


def test_bg_palette_covers_the_whole_bg_vocabulary():
    """Without this, `visualize_room` raises on the first unmapped tileset."""
    assert set(VALID_BG_TILES) <= set(BG_PALETTE)


def test_fg_palette_covers_the_whole_fg_vocabulary():
    assert set(ALL_TILES_AND_ENTITIES) <= set(FG_PALETTE)


def test_color_map_defaults_to_the_fg_palette():
    assert color_map(val="1") == "background-color: salmon"


def test_visualize_room_draws_a_bg_room():
    df = pd.DataFrame(np.array([["0", "a"], ["e", "8"]]))

    assert visualize_room(df=df, layer="bg").to_html()


# ------------------------------------------------------------------------- CLI


def test_layer_sampling_pairs_each_vocabulary_with_its_border():
    assert LAYER_SAMPLING["fg"] == (ALL_TILES_AND_ENTITIES, "1")
    assert LAYER_SAMPLING["bg"] == (VALID_BG_TILES, "0")


def test_main_rejects_a_bg_skeleton(tmp_path, monkeypatch):
    """Exits and respawn points are foreground concepts; integration is deferred."""
    dicts = tmp_path / "dicts"
    dicts.mkdir()
    (dicts / "bg_6_000011012.json").write_text(
        '{"#mdmc_matrix": "000011012", "000": {"0": 1.0}}'
    )
    monkeypatch.setattr("seleCte.pcg.rooms_generation.PE_DICTS_DIR", dicts)

    args = Args()
    args.nb_rooms = 1
    args.experiment_type = "skel"
    args.room_size = (12, 9)
    args.bt_depth = 0
    args.proba = 0.2
    args.output_dir = str(tmp_path / "out")
    args.dict_proba_estimation = "bg_6_000011012"
    args.layer = "bg"
    args.verbose = False

    with pytest.raises(ValueError, match="foreground-only"):
        main(args)


def test_main_writes_bg_rooms_in_the_bg_alphabet(tmp_path, monkeypatch):
    dicts = tmp_path / "dicts"
    dicts.mkdir()
    (dicts / "bg_6_000011012.json").write_text(
        '{"#mdmc_matrix": "000011012", "000": {"0": 0.5, "a": 0.5}}'
    )
    monkeypatch.setattr("seleCte.pcg.rooms_generation.PE_DICTS_DIR", dicts)
    out = tmp_path / "out"

    args = Args()
    args.nb_rooms = 2
    args.experiment_type = "rooms"
    args.room_size = (12, 9)
    args.bt_depth = 0
    args.proba = 0.2
    args.output_dir = str(out)
    args.dict_proba_estimation = "bg_6_000011012"
    args.layer = "bg"
    args.verbose = False

    main(args)

    written = sorted(out.iterdir())
    assert len(written) == 2
    for path in written:
        rows = [line.split(";") for line in path.read_text().splitlines()]
        assert len(rows) == 9
        assert all(len(row) == 12 for row in rows)
        assert {cell for row in rows for cell in row} <= set(VALID_BG_TILES)
