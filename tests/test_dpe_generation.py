import json

import pytest

from seleCte.pcg.dpe_generation import (
    LAYERS,
    Args,
    estimate_dpe,
    extract_dataset_from_filepath,
    main,
)
from seleCte.utils import (
    ALL_TILES_AND_ENTITIES,
    MDMC_KEY,
    VALID_BG_TILES,
    parse_mdmc_matrix,
)

DEFAULT_MASK = "000011012"


def write_corpus_room(corpus_dir, name, rows) -> None:
    """Writes one room into a corpus directory, in the pipeline's csv dialect."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / name).write_text("\n".join(";".join(row) for row in rows) + "\n")


def make_args(dataset, layer, mdmc_matrix=DEFAULT_MASK) -> Args:
    args = Args()
    args.dataset = dataset
    args.layer = layer
    args.mdmc_matrix = mdmc_matrix

    return args


def estimate(corpus_dir, dataset, vocabulary, mask=DEFAULT_MASK):
    return estimate_dpe(
        corpus_dir=corpus_dir,
        dataset=dataset,
        mdmc_matrix=parse_mdmc_matrix(mask),
        vocabulary=vocabulary,
    )


# ----------------------------------------------------------------- estimation


def test_estimate_dpe_counts_a_uniform_room(tmp_path):
    """An all-void room yields one pattern, `000`, that always predicts void."""
    write_corpus_room(tmp_path, "1_lvl_1_fg.csv", ["000", "000", "000"])

    dpe = estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)

    assert list(dpe) == ["000"]
    assert dpe["000"]["0"] == 1.0


def test_estimate_dpe_normalizes_each_pattern_independently(tmp_path):
    write_corpus_room(tmp_path, "1_a_fg.csv", ["000", "001", "001"])

    dpe = estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)

    for distribution in dpe.values():
        assert sum(distribution.values()) == pytest.approx(1.0)


def test_estimate_dpe_zero_fills_the_whole_vocabulary(tmp_path):
    write_corpus_room(tmp_path, "1_a_fg.csv", ["000", "000", "000"])

    dpe = estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)

    assert set(dpe["000"]) == set(ALL_TILES_AND_ENTITIES)


def test_estimate_dpe_holds_only_observed_patterns(tmp_path):
    """The dict is the corpus's patterns, not the vocabulary's permutations."""
    write_corpus_room(tmp_path, "1_a_fg.csv", ["000", "000", "000"])

    dpe = estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)

    assert len(dpe) == 1
    assert len(dpe) < len(ALL_TILES_AND_ENTITIES) ** 3


def test_estimate_dpe_emits_patterns_in_vocabulary_order(tmp_path):
    """Not corpus order: `iterdir` is filesystem-dependent, runs must not be."""
    write_corpus_room(tmp_path, "1_a_fg.csv", ["111", "101", "110"])
    write_corpus_room(tmp_path, "1_b_fg.csv", ["000", "010", "001"])

    dpe = estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)
    order = {symbol: i for i, symbol in enumerate(ALL_TILES_AND_ENTITIES)}

    assert list(dpe) == sorted(dpe, key=lambda p: tuple(order[c] for c in p))


def test_estimate_dpe_selects_by_dataset_id(tmp_path):
    write_corpus_room(tmp_path, "1_a_fg.csv", ["000", "000", "000"])
    write_corpus_room(tmp_path, "2_a_fg.csv", ["111", "111", "111"])

    assert list(estimate(tmp_path, "1", ALL_TILES_AND_ENTITIES)) == ["000"]
    assert list(estimate(tmp_path, "2", ALL_TILES_AND_ENTITIES)) == ["111"]
    assert sorted(estimate(tmp_path, "12", ALL_TILES_AND_ENTITIES)) == ["000", "111"]


def test_estimate_dpe_rejects_a_missing_corpus(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        estimate(tmp_path / "nope", "1", ALL_TILES_AND_ENTITIES)


def test_estimate_dpe_rejects_a_dataset_matching_no_room(tmp_path):
    write_corpus_room(tmp_path, "1_a_fg.csv", ["000", "000", "000"])

    with pytest.raises(ValueError, match="belongs to dataset"):
        estimate(tmp_path, "7", ALL_TILES_AND_ENTITIES)


def test_extract_dataset_from_filepath_reads_the_first_character(tmp_path):
    assert extract_dataset_from_filepath(path=tmp_path / "6_lvl_-1_bg.csv") == "6"


# --------------------------------------------------------------------- layers


def test_layers_pair_each_corpus_with_its_own_vocabulary():
    assert LAYERS["fg"].vocabulary == ALL_TILES_AND_ENTITIES
    assert LAYERS["bg"].vocabulary == VALID_BG_TILES
    assert LAYERS["fg"].corpus_dir != LAYERS["bg"].corpus_dir


def test_only_the_bg_layer_is_prefixed():
    """fg keeps its historical name, so no committed dict is invalidated."""
    assert LAYERS["fg"].output_prefix == ""
    assert LAYERS["bg"].output_prefix == "bg_"


def test_bg_dict_name_still_ends_in_the_mask():
    """`load_dpe` recovers the mask from the name's tail for older dicts."""
    name = f"{LAYERS['bg'].output_prefix}6_{DEFAULT_MASK}.json"

    assert name.removesuffix(".json").rsplit("_", 1)[-1] == DEFAULT_MASK


def test_estimate_dpe_on_bg_never_admits_a_foreground_symbol(tmp_path):
    """The vocabulary is what stops a bg room ever sampling a spike."""
    write_corpus_room(tmp_path, "6_lvl_a_bg.csv", ["0aa", "a0a", "aa0"])

    dpe = estimate(tmp_path, "6", VALID_BG_TILES)
    sampled = {symbol for distribution in dpe.values() for symbol in distribution}

    assert sampled == set(VALID_BG_TILES)
    assert not sampled & {"^", "<", "v", ">", "S", "D", "C", "_", "W", "R", "F", "L"}


# ----------------------------------------------------------------------- main


def test_main_writes_the_bg_dict_under_its_prefixed_name(tmp_path, monkeypatch):
    corpus = tmp_path / "all_rooms_bg"
    write_corpus_room(corpus, "6_lvl_a_bg.csv", ["0aa", "a0a", "aa0"])
    out = tmp_path / "dicts"
    out.mkdir()
    monkeypatch.setattr("seleCte.pcg.dpe_generation.PE_DICTS_DIR", out)
    monkeypatch.setattr(
        "seleCte.pcg.dpe_generation.LAYERS",
        {**LAYERS, "bg": type(LAYERS["bg"])(corpus, VALID_BG_TILES, "bg_")},
    )

    main(make_args(dataset="6", layer="bg"))

    written = out / f"bg_6_{DEFAULT_MASK}.json"
    assert written.is_file()
    assert json.loads(written.read_text())[MDMC_KEY] == DEFAULT_MASK


def test_main_embeds_the_mask_it_trained_with(tmp_path, monkeypatch):
    corpus = tmp_path / "all_rooms"
    write_corpus_room(corpus, "1_a_fg.csv", ["000", "001", "011"])
    out = tmp_path / "dicts"
    out.mkdir()
    monkeypatch.setattr("seleCte.pcg.dpe_generation.PE_DICTS_DIR", out)
    monkeypatch.setattr(
        "seleCte.pcg.dpe_generation.LAYERS",
        {**LAYERS, "fg": type(LAYERS["fg"])(corpus, ALL_TILES_AND_ENTITIES, "")},
    )

    main(make_args(dataset="1", layer="fg", mdmc_matrix="000001012"))

    payload = json.loads((out / "1_000001012.json").read_text())
    assert payload[MDMC_KEY] == "000001012"
    assert all(len(k) == 2 for k in payload if k != MDMC_KEY)
