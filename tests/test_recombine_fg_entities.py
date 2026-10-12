import json
import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from seleCte.pcg.preprocessing.corpus import invalid_symbol_counts, output_filename
from seleCte.pcg.preprocessing.recombine_fg_entities import (
    Args,
    MappedEntity,
    _tile_span,
    collapse_fg_tiles,
    discover_room_pairs,
    filter_entities,
    integrate_entities_to_fg,
    main,
    recombine_room,
    stamp_entity,
)
from seleCte.utils import ALL_TILES_AND_ENTITIES, ENTITY_SYMBOLS, VALID_FG_TILES


def build_room(rows: list[str]) -> pd.DataFrame:
    """A room from one string per line: build_room(["0000", "0110"])."""
    return pd.DataFrame([list(row) for row in rows])


def write_room(tmp_path, chapter, name, rows, entities):
    """Writes a room's two files into a chapter dir; returns the fg path."""
    chapter_dir = tmp_path / chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    fg_path = chapter_dir / f"lvl_{name}_fg.csv"
    fg_path.write_text("\n".join(";".join(row) for row in rows) + "\n")
    (chapter_dir / f"lvl_{name}_entities.json").write_text(json.dumps(entities))

    return fg_path


def make_args(levels_dir, output_dir, dry_run=False) -> Args:
    args = Args()
    args.levels_dir = str(levels_dir)
    args.output_dir = str(output_dir)
    args.dry_run = dry_run

    return args


def spinner(x, y):
    return {"__name": "spinner", "id": 1, "x": x, "y": y}


# _tile_span: the geometry the whole stage rests on


def test_tile_span_sized_entity_covers_its_box():
    assert _tile_span(origin_px=0, size_px=24, limit=10) == (0, 3)


def test_tile_span_sized_entity_clips_at_the_far_edge():
    assert _tile_span(origin_px=64, size_px=64, limit=10) == (8, 10)


def test_tile_span_sized_entity_clips_at_the_near_edge():
    assert _tile_span(origin_px=-8, size_px=24, limit=10) == (0, 2)


def test_tile_span_point_entity_shifts_back_one_tile():
    assert _tile_span(origin_px=40, size_px=None, limit=10) == (4, 5)


def test_tile_span_point_entity_on_the_first_tile_is_clamped_not_dropped():
    # The bug this stage used to have: the back-shift produced -1, the label
    # slice matched nothing, and the entity vanished.
    assert _tile_span(origin_px=0, size_px=None, limit=10) == (0, 1)


def test_tile_span_point_entity_past_the_room_is_dropped():
    # The clamp is deliberately asymmetric: past the far edge there is no tile
    # inside the room to recover.
    assert _tile_span(origin_px=88, size_px=None, limit=10) is None


def test_tile_span_zero_sized_entity_is_dropped():
    assert _tile_span(origin_px=16, size_px=0, limit=10) is None


def test_tile_span_never_returns_a_negative_bound():
    # Positional indexing reads a negative bound as counting from the end, so
    # this is a hard contract rather than a nicety.
    for origin in range(-64, 128, 8):
        for size in (None, 0, 8, 40):
            span = _tile_span(origin_px=origin, size_px=size, limit=10)
            assert span is None or (0 <= span[0] < span[1] <= 10)


# filter_entities


def test_filter_entities_keeps_only_mapped_names_in_order():
    entities = [
        {"__name": "player", "id": 0, "x": 8, "y": 8},
        {"__name": "spikesUp", "id": 1, "x": 16, "y": 24, "width": 16},
        {"__name": "strawberry", "id": 2, "x": 8, "y": 8},
        spinner(x=32, y=32),
    ]

    mapped = filter_entities(entities=entities)

    assert [entity.name for entity in mapped] == ["spikesUp", "spinner"]
    assert [entity.symbol for entity in mapped] == ["^", "S"]


def test_filter_entities_splits_boosters_by_the_red_flag():
    entities = [
        {"__name": "booster", "id": 1, "x": 8, "y": 8, "red": True},
        {"__name": "booster", "id": 2, "x": 16, "y": 8, "red": False},
    ]

    mapped = filter_entities(entities=entities)

    assert [entity.name for entity in mapped] == ["redBooster", "greenBooster"]
    assert [entity.symbol for entity in mapped] == ["Q", "O"]


def test_filter_entities_rejects_a_booster_without_a_red_flag():
    with pytest.raises(ValueError, match="red"):
        filter_entities(entities=[{"__name": "booster", "id": 1, "x": 8, "y": 8}])


def test_filter_entities_on_an_empty_list_returns_empty():
    assert filter_entities(entities=[]) == []


def test_filter_entities_honours_a_custom_map():
    # Regression: the notebook took a map argument but assigned symbols from the
    # module-level default, so a custom map silently produced NaN symbols.
    entities = [
        spinner(x=8, y=8),
        {"__name": "spikesUp", "id": 2, "x": 8, "y": 8, "width": 8},
    ]

    mapped = filter_entities(entities=entities, entity_map={"spinner": "X"})

    assert len(mapped) == 1
    assert mapped[0].symbol == "X"


def test_filter_entities_rejects_an_entity_without_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        filter_entities(entities=[{"__name": "spinner", "id": 7}])


def test_filter_entities_reads_optional_sizes_independently():
    entities = [{"__name": "spikesLeft", "id": 1, "x": 8, "y": 8, "height": 24}]

    (mapped,) = filter_entities(entities=entities)

    assert mapped.height == 24
    assert mapped.width is None


# collapse_fg_tiles / invalid_symbol_counts


def test_collapse_fg_tiles_maps_every_variant_to_solid():
    room = build_room(["".join(VALID_FG_TILES)])

    collapsed = collapse_fg_tiles(df_fg=room)

    assert set(collapsed.to_numpy().flatten()) == {"1"}


def test_collapse_fg_tiles_leaves_void_and_unknown_symbols_alone():
    collapsed = collapse_fg_tiles(df_fg=build_room(["03Z0"]))

    assert list(collapsed.iloc[0]) == ["0", "1", "Z", "0"]


def test_collapse_fg_tiles_does_not_mutate_its_input():
    room = build_room(["33", "33"])

    collapse_fg_tiles(df_fg=room)

    assert list(room.iloc[0]) == ["3", "3"]


def test_invalid_symbol_counts_is_empty_for_a_clean_room():
    assert invalid_symbol_counts(df_room=build_room(["0110", "1001"])) == {}


def test_invalid_symbol_counts_reports_the_offenders():
    # Regression: the notebook's warning reported the count of *valid* tiles
    # while claiming they were invalid.
    assert invalid_symbol_counts(df_room=build_room(["0Z10", "1Z0Y"])) == {
        "Z": 2,
        "Y": 1,
    }


# stamping


def test_stamp_entity_fills_the_whole_box():
    room = build_room(["0000", "0000", "0000"])
    entity = MappedEntity(name="dreamBlock", symbol="D", x=16, y=8, width=16, height=16)

    assert stamp_entity(df_room=room, entity=entity) is True
    assert [list(row) for row in room.to_numpy()] == [
        ["0", "0", "0", "0"],
        ["0", "0", "D", "D"],
        ["0", "0", "D", "D"],
    ]


def test_stamp_entity_at_the_room_origin_is_not_dropped():
    # End-to-end form of the clamp fix: this spinner used to disappear.
    room = build_room(["00", "00"])

    assert stamp_entity(df_room=room, entity=_point("S", x=0, y=0)) is True
    assert room.iat[0, 0] == "S"


def test_stamp_entity_outside_the_room_returns_false_and_changes_nothing():
    room = build_room(["00", "00"])

    assert stamp_entity(df_room=room, entity=_point("S", x=800, y=800)) is False
    assert set(room.to_numpy().flatten()) == {"0"}


def test_integrate_entities_applies_them_in_order():
    room = build_room(["000", "000"])
    entities = [_point("S", x=8, y=8), _point("R", x=8, y=8)]

    result, n_outside = integrate_entities_to_fg(df_fg=room, entities=entities)

    assert result.iat[0, 0] == "R"
    assert n_outside == 0


def test_integrate_entities_counts_the_ones_that_fall_outside():
    room = build_room(["000", "000"])

    _, n_outside = integrate_entities_to_fg(
        df_fg=room, entities=[_point("S", x=8, y=8), _point("S", x=800, y=8)]
    )

    assert n_outside == 1


def test_integrate_entities_never_mutates_the_foreground():
    room = build_room(["000", "000"])

    result, _ = integrate_entities_to_fg(df_fg=room, entities=[])

    assert result is not room
    assert result.equals(room)


def _point(symbol, x, y):
    return MappedEntity(
        name="spinner", symbol=symbol, x=x, y=y, width=None, height=None
    )


# discovery and naming


def test_output_filename_uses_the_chapter_prefix():
    expected = {
        ("1-ForsakenCity", "1"): "1_lvl_1_fg.csv",
        ("1H-ForsakenCity", "00"): "1H_lvl_00_fg.csv",
        ("LostLevels", "a"): "LostLevels_lvl_a_fg.csv",
        # Only the *directory* is split on the dash: room names contain them too.
        ("0-Intro", "-1"): "0_lvl_-1_fg.csv",
    }
    for (chapter, name), name_out in expected.items():
        fg_path = Path("levels") / chapter / f"lvl_{name}_fg.csv"
        assert output_filename(room_path=fg_path) == name_out


def test_discover_room_pairs_finds_and_sorts_pairs(tmp_path):
    write_room(tmp_path, "2-OldSite", "b", [["0"]], [])
    write_room(tmp_path, "1-ForsakenCity", "a", [["0"]], [])

    pairs = discover_room_pairs(levels_dir=tmp_path)

    assert [fg.parent.name for fg, _ in pairs] == ["1-ForsakenCity", "2-OldSite"]
    assert all(entities.is_file() for _, entities in pairs)


def test_discover_room_pairs_skips_a_room_without_entities(tmp_path, caplog):
    write_room(tmp_path, "1-ForsakenCity", "a", [["0"]], [])
    (tmp_path / "1-ForsakenCity" / "lvl_b_fg.csv").write_text("0\n")

    with caplog.at_level(logging.WARNING):
        pairs = discover_room_pairs(levels_dir=tmp_path)

    assert len(pairs) == 1
    assert "no entities file" in caplog.text


def test_discover_room_pairs_raises_on_a_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        discover_room_pairs(levels_dir=tmp_path / "nope")


def test_discover_room_pairs_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match=re.escape("No `*_fg.csv` found")):
        discover_room_pairs(levels_dir=tmp_path)


# recombine_room and main


def test_recombine_room_matches_a_hand_written_room(tmp_path):
    rows = [list("333333"), list("300003"), list("333333")]
    # `spikesUp` carries only a width, so its y axis is a point axis and gets the
    # one-tile back-shift: y=16 lands on row 1, not row 2.
    entities = [
        {"__name": "spikesUp", "id": 1, "x": 8, "y": 16, "width": 16},
        spinner(x=32, y=16),
    ]
    fg_path = write_room(tmp_path, "1-ForsakenCity", "a", rows, entities)

    df_room, report = recombine_room(
        fg_path=fg_path, entities_path=fg_path.with_name("lvl_a_entities.json")
    )

    assert [list(row) for row in df_room.to_numpy()] == [
        list("111111"),
        list("1^^S01"),
        list("111111"),
    ]
    assert report.n_entities_kept == 2
    assert report.n_entities_outside == 0


def test_recombine_room_reports_unknown_tiles_and_returns_nothing(tmp_path):
    fg_path = write_room(tmp_path, "1-ForsakenCity", "a", [list("3Z3")], [])

    df_room, report = recombine_room(
        fg_path=fg_path, entities_path=fg_path.with_name("lvl_a_entities.json")
    )

    assert df_room is None
    assert report.invalid_symbols == {"Z": 1}


def test_main_writes_one_csv_per_room_with_the_exact_bytes(tmp_path):
    levels, output = tmp_path / "levels", tmp_path / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("303"), list("333")], [])

    main(make_args(levels_dir=levels, output_dir=output))

    # Pins separator, absence of header and index, and the trailing newline.
    assert (output / "1_lvl_a_fg.csv").read_text() == "1;0;1\n1;1;1\n"


def test_main_creates_the_output_directory(tmp_path):
    levels, output = tmp_path / "levels", tmp_path / "deep" / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("3")], [])

    main(make_args(levels_dir=levels, output_dir=output))

    assert output.is_dir()


def test_main_dry_run_writes_nothing(tmp_path):
    levels, output = tmp_path / "levels", tmp_path / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("3")], [])

    main(make_args(levels_dir=levels, output_dir=output, dry_run=True))

    assert not output.exists()


def test_main_skips_a_room_with_unknown_tiles(tmp_path, caplog):
    levels, output = tmp_path / "levels", tmp_path / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("3Z3")], [])
    write_room(levels, "1-ForsakenCity", "b", [list("333")], [])

    with caplog.at_level(logging.WARNING):
        main(make_args(levels_dir=levels, output_dir=output))

    assert not (output / "1_lvl_a_fg.csv").exists()
    assert (output / "1_lvl_b_fg.csv").exists()
    assert "'Z': 1" in caplog.text


def test_main_raises_when_two_chapters_collide(tmp_path):
    levels, output = tmp_path / "levels", tmp_path / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("3")], [])
    write_room(levels, "1-SomethingElse", "a", [list("3")], [])

    with pytest.raises(ValueError, match="both map to"):
        main(make_args(levels_dir=levels, output_dir=output))


def test_main_is_idempotent(tmp_path):
    levels, output = tmp_path / "levels", tmp_path / "out"
    write_room(levels, "1-ForsakenCity", "a", [list("303")], [spinner(x=8, y=8)])
    args = make_args(levels_dir=levels, output_dir=output)

    main(args)
    first = (output / "1_lvl_a_fg.csv").read_text()
    main(args)

    assert (output / "1_lvl_a_fg.csv").read_text() == first


# vocabulary invariant


def test_every_entity_symbol_is_in_the_model_vocabulary():
    # If a symbol recombination emits is missing from the vocabulary, the DPE
    # never learns a distribution over it and generation can never sample it.
    assert set(ENTITY_SYMBOLS.values()) <= set(ALL_TILES_AND_ENTITIES)


def test_entity_symbols_are_single_characters():
    assert all(len(symbol) == 1 for symbol in ENTITY_SYMBOLS.values())
