import logging
import re
from pathlib import Path

import pytest

from seleCte.pcg.preprocessing.collect_bg_rooms import (
    Args,
    collect_room,
    discover_bg_rooms,
    main,
)
from seleCte.utils import ALL_TILES_AND_ENTITIES, VALID_BG_TILES


def write_bg_room(tmp_path, chapter, name, rows) -> Path:
    """Writes one background room into a chapter dir; returns its path."""
    chapter_dir = tmp_path / chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    bg_path = chapter_dir / f"lvl_{name}_bg.csv"
    bg_path.write_text("\n".join(";".join(row) for row in rows) + "\n")

    return bg_path


def make_args(levels_dir, output_dir, dry_run=False) -> Args:
    args = Args()
    args.levels_dir = str(levels_dir)
    args.output_dir = str(output_dir)
    args.dry_run = dry_run

    return args


# ---------------------------------------------------------------- discovery


def test_discover_bg_rooms_finds_every_chapter(tmp_path):
    write_bg_room(tmp_path, "1-ForsakenCity", "1", ["00", "11"])
    write_bg_room(tmp_path, "6-Reflection", "a", ["aa", "0a"])

    found = discover_bg_rooms(levels_dir=tmp_path)

    assert [p.name for p in found] == ["lvl_1_bg.csv", "lvl_a_bg.csv"]


def test_discover_bg_rooms_ignores_the_other_layers(tmp_path):
    chapter = tmp_path / "1-ForsakenCity"
    chapter.mkdir(parents=True)
    (chapter / "lvl_1_fg.csv").write_text("1;1\n")
    (chapter / "lvl_1_obj.csv").write_text("0;0\n")
    (chapter / "lvl_1_entities.json").write_text("[]")
    write_bg_room(tmp_path, "1-ForsakenCity", "1", ["00", "11"])

    assert [p.name for p in discover_bg_rooms(levels_dir=tmp_path)] == ["lvl_1_bg.csv"]


def test_discover_bg_rooms_rejects_a_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        discover_bg_rooms(levels_dir=tmp_path / "nope")


def test_discover_bg_rooms_rejects_an_empty_tree(tmp_path):
    (tmp_path / "1-ForsakenCity").mkdir()

    with pytest.raises(ValueError, match=re.escape("No `*_bg.csv` found")):
        discover_bg_rooms(levels_dir=tmp_path)


# ------------------------------------------------------------------ reading


def test_collect_room_keeps_every_symbol_untouched(tmp_path):
    """The whole point of the bg layer: no collapsing, unlike the fg stage."""
    rows = ["0a1", "89e", "45d"]
    bg_path = write_bg_room(tmp_path, "7-Summit", "1", rows)

    df_room, report = collect_room(bg_path=bg_path)

    assert df_room is not None
    assert [list(row) for row in rows] == df_room.to_numpy().tolist()
    assert report.invalid_symbols == {}


def test_collect_room_accepts_the_whole_vocabulary(tmp_path):
    bg_path = write_bg_room(tmp_path, "7-Summit", "1", [list(VALID_BG_TILES)])

    df_room, report = collect_room(bg_path=bg_path)

    assert df_room is not None
    assert report.invalid_symbols == {}


def test_collect_room_rejects_symbols_outside_the_vocabulary(tmp_path):
    bg_path = write_bg_room(tmp_path, "1-ForsakenCity", "1", ["0Z1", "ZZ0"])

    df_room, report = collect_room(bg_path=bg_path)

    assert df_room is None
    assert report.invalid_symbols == {"Z": 3}


def test_collect_room_rejects_foreground_entity_symbols(tmp_path):
    """`^` is a spike: a fg-only symbol that must not pass as a bg tile."""
    bg_path = write_bg_room(tmp_path, "1-ForsakenCity", "1", ["0^0", "000"])

    df_room, report = collect_room(bg_path=bg_path)

    assert df_room is None
    assert report.invalid_symbols == {"^": 1}


# --------------------------------------------------------------------- main


def test_main_writes_one_csv_per_room_with_the_exact_bytes(tmp_path):
    write_bg_room(tmp_path / "levels", "6-Reflection", "a", ["0aa", "a0a"])
    write_bg_room(tmp_path / "levels", "4-GoldenRidge", "b", ["666", "060"])
    out = tmp_path / "out"

    main(make_args(tmp_path / "levels", out))

    assert sorted(p.name for p in out.iterdir()) == ["4_lvl_b_bg.csv", "6_lvl_a_bg.csv"]
    assert (out / "6_lvl_a_bg.csv").read_text() == "0;a;a\na;0;a\n"
    assert (out / "4_lvl_b_bg.csv").read_text() == "6;6;6\n0;6;0\n"


def test_main_creates_the_output_directory(tmp_path):
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "1", ["01", "10"])
    out = tmp_path / "nested" / "out"

    main(make_args(tmp_path / "levels", out))

    assert (out / "1_lvl_1_bg.csv").is_file()


def test_main_dry_run_writes_nothing(tmp_path, caplog):
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "1", ["01", "10"])
    out = tmp_path / "out"

    with caplog.at_level(logging.INFO):
        main(make_args(tmp_path / "levels", out, dry_run=True))

    assert not out.exists()
    assert "1 file(s) would change" in caplog.text


def test_main_is_idempotent(tmp_path, caplog):
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "1", ["01", "10"])
    out = tmp_path / "out"
    main(make_args(tmp_path / "levels", out))

    with caplog.at_level(logging.INFO):
        main(make_args(tmp_path / "levels", out, dry_run=True))

    assert "0 file(s) would change" in caplog.text


def test_main_skips_a_room_with_unknown_tiles(tmp_path, caplog):
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "good", ["01", "10"])
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "bad", ["0Z", "10"])
    out = tmp_path / "out"

    with caplog.at_level(logging.WARNING):
        main(make_args(tmp_path / "levels", out))

    assert [p.name for p in out.iterdir()] == ["1_lvl_good_bg.csv"]
    assert "valid_bg_tiles" in caplog.text


def test_main_raises_when_two_chapters_collide(tmp_path):
    write_bg_room(tmp_path / "levels", "1-ForsakenCity", "1", ["01"])
    write_bg_room(tmp_path / "levels", "1-SomethingElse", "1", ["10"])

    with pytest.raises(ValueError, match="both map to"):
        main(make_args(tmp_path / "levels", tmp_path / "out"))


def test_main_keeps_room_names_containing_dashes(tmp_path):
    """`lvl_-1_bg.csv` is a real room name: only the directory may be split."""
    write_bg_room(tmp_path / "levels", "0-Intro", "-1", ["01"])
    out = tmp_path / "out"

    main(make_args(tmp_path / "levels", out))

    assert [p.name for p in out.iterdir()] == ["0_lvl_-1_bg.csv"]


# ------------------------------------------------------------- vocabularies


def test_bg_and_fg_vocabularies_disagree_about_shared_symbols():
    """Why the two layers get separate corpus directories and separate dicts.

    `0` and `1` are in both alphabets and mean different things: solid ground in
    the recombined fg grid, a tileset in bg. Mixing the corpora would train one
    distribution over two meanings.
    """
    shared = set(VALID_BG_TILES) & set(ALL_TILES_AND_ENTITIES)

    assert shared == {"0", "1"}
    assert not set(VALID_BG_TILES) >= set(ALL_TILES_AND_ENTITIES)
