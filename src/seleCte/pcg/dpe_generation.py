"""Builds the probability estimation dict (DPE) the room generators sample from.

Run as a script, this is the offline training step of the PCG pipeline. It
slides a 3x3 window over every level csv in the corpus, reads the neighbourhood
cells the MdMC matrix marks with a `1`, and counts which tile was observed at the
window's bottom-right corner for each pattern. Those counts are normalized per
pattern, zero-filled over the full tile vocabulary, and written to
`probability_estimation_dicts/`.

Three things identify a run, all of them CLI arguments: `--mdmc-matrix` is the
flattened neighbourhood window (default `000011012`), `--dataset` the level ids to
train on, and `--layer` which of the two tile layers to read.

The layer picks the corpus, the vocabulary and the output prefix together, because
those three cannot be mixed and matched - see `LAYERS`. Foreground dicts keep
their historical `{dataset}_{mask}.json` name; background dicts are prefixed,
`bg_{dataset}_{mask}.json`, so that `load_dpe`'s fallback of recovering the mask
from the end of the file name still works for both.

**Background dicts should be trained one chapter at a time.** A background tileset
is a property of the chapter rather than of the room, so `--layer bg --dataset
0123456789L` mixes tilesets that never co-occur in the game - the reasoning is
recorded in `preprocessing.collect_bg_rooms`.
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from seleCte.paths import ALL_ROOMS_BG_DIR, ALL_ROOMS_DIR, PE_DICTS_DIR
from seleCte.utils import (
    ALL_TILES_AND_ENTITIES,
    MDMC_KEY,
    VALID_BG_TILES,
    get_all_submatrices,
    parse_mdmc_matrix,
    submatrix_to_count,
)

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Everything that differs between the two tile layers.

    Bundled rather than passed as three arguments because they are not
    independent: reading the background corpus with the foreground vocabulary
    would validate nothing and would write a dict under a name the foreground
    generator then picks up.
    """

    corpus_dir: Path
    vocabulary: list[str]
    output_prefix: str


# Immutable so a caller cannot mutate the shared table, and so it can be a default
# argument without tripping B006.
LAYERS: Mapping[str, LayerSpec] = MappingProxyType(
    {
        "fg": LayerSpec(
            corpus_dir=ALL_ROOMS_DIR,
            vocabulary=ALL_TILES_AND_ENTITIES,
            output_prefix="",
        ),
        "bg": LayerSpec(
            corpus_dir=ALL_ROOMS_BG_DIR,
            vocabulary=VALID_BG_TILES,
            output_prefix="bg_",
        ),
    }
)


class Args:
    """Parsed command-line arguments, one attribute per `add_argument` call.

    Deliberately not an `argparse.Namespace` subclass: `Namespace.__getattr__`
    returns `Any`, so a misspelled `args.datset` would type-check and only blow
    up mid-run. A plain class has no such escape hatch and the checker rejects
    unknown attributes. `parse_args` still fills it in - it only ever calls
    `setattr` on the namespace it is handed.
    """

    mdmc_matrix: str
    dataset: str
    layer: str


def extract_dataset_from_filepath(path: Path) -> str:
    """Extract the dataset to which a filepath belongs.

    Datasets are identified by their level ids, being a number
    between 0-9, or the letter L for LostLevels.
    """
    # TODO: This might need some refactoring to include StrawberryJam
    return path.as_posix().split(sep="/")[-1][0]


def estimate_dpe(
    corpus_dir: Path,
    dataset: str,
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
    vocabulary: list[str],
) -> dict[str, dict[str, int | float]]:
    """Counts tile patterns across a corpus and normalizes them per pattern.

    Only patterns the corpus actually contains end up in the result. That matters
    for more than tidiness: enumerating the vocabulary to the power of the
    neighbourhood size grows past a million entries for a five-cell mask, while
    the corpus itself holds a couple of hundred distinct patterns - 162 of the
    3375 possible ones, for the background layer under the default mask.

    Patterns are emitted in vocabulary order rather than in the order the corpus
    happened to present them, so two runs over the same corpus produce the same
    file whatever order the filesystem lists it in.

    Args:
        corpus_dir: the directory of room csvs to train on.
        dataset: the level ids to keep, matched against the first character of
            each file name.
        mdmc_matrix: the neighbourhood mask, see `parse_mdmc_matrix`.
        vocabulary: every symbol the layer can hold, used to zero-fill each
            distribution so generation can never sample outside it.

    Returns:
        The `{pattern: {symbol: probability}}` dict.

    Raises:
        ValueError: the corpus directory does not exist, or no file in it belongs
            to the requested dataset.
    """
    if not corpus_dir.is_dir():
        raise ValueError(
            f"Corpus directory {corpus_dir} does not exist. Build it first with "
            f"the matching module in `seleCte.pcg.preprocessing`."
        )

    d_patterns: defaultdict[str, list[str]] = defaultdict(list)
    n_rooms: int = 0
    for file_path in corpus_dir.iterdir():
        if extract_dataset_from_filepath(path=file_path) not in dataset:
            continue

        n_rooms += 1
        df_temp: DataFrame = pd.read_csv(
            filepath_or_buffer=file_path, header=None, sep=";", dtype=str
        ).map(func=lambda x: "0" if x == "" or x == " " else x)
        level_data_temp: np.ndarray[tuple[int, int], np.dtype[Any]] = df_temp.to_numpy(
            dtype=str
        )

        for submatrix in get_all_submatrices(array=level_data_temp):
            pattern, tiletype = submatrix_to_count(
                array=submatrix, mdmc_matrix=mdmc_matrix
            )
            d_patterns[pattern].append(tiletype)

    if not n_rooms:
        raise ValueError(
            f"No room in {corpus_dir} belongs to dataset {dataset!r}. Corpus files "
            f"are named `{{chapter_id}}_lvl_...`, and the id is their first "
            f"character."
        )

    symbol_order: dict[str, int] = {symbol: i for i, symbol in enumerate(vocabulary)}
    d_proba_estimation: dict[str, dict[str, int | float]] = {}
    for pattern in sorted(
        d_patterns,
        key=lambda p: tuple(symbol_order.get(c, len(symbol_order)) for c in p),
    ):
        # Observed symbols keep the order the corpus presented them in, and the
        # zero-fill is appended in vocabulary order behind them.
        counts: dict[str, int] = dict(Counter(d_patterns[pattern]))
        total: int = sum(counts.values())
        distribution: dict[str, int | float] = {k: v / total for k, v in counts.items()}
        for symbol in vocabulary:
            if symbol not in distribution:
                distribution[symbol] = 0.0
        d_proba_estimation[pattern] = distribution

    logger.info(
        msg=f"Estimated {len(d_proba_estimation)} patterns over {n_rooms} rooms "
        f"from {corpus_dir.name}."
    )

    return d_proba_estimation


def main(args: Args) -> None:
    """Trains one DPE and writes it to `PE_DICTS_DIR`.

    The mask is written into the result under `MDMC_KEY`: generation needs it to
    read patterns back the same way, and inferring it from the file name only
    works for dicts that still follow the `{dataset}_{mask}.json` convention.

    Args:
        args: parsed arguments, see `Args`.
    """
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]] = parse_mdmc_matrix(
        args.mdmc_matrix
    )
    layer: LayerSpec = LAYERS[args.layer]

    d_proba_estimation: dict[str, dict[str, int | float]] = estimate_dpe(
        corpus_dir=layer.corpus_dir,
        dataset=args.dataset,
        mdmc_matrix=mdmc_matrix,
        vocabulary=layer.vocabulary,
    )

    # The mask rides along with the patterns so generation cannot silently read
    # them back with a different one - see `check_dpe_matches_mask`. It is mixed
    # in only here, at the serialization boundary, to keep `d_proba_estimation`
    # itself homogeneous; `load_dpe` pops the key straight back out.
    result: dict[str, str | dict[str, float]] = {MDMC_KEY: args.mdmc_matrix}
    result.update(d_proba_estimation)

    fn_result_dpe: Path = (
        PE_DICTS_DIR / f"{layer.output_prefix}{args.dataset}_{args.mdmc_matrix}.json"
    )
    with fn_result_dpe.open(mode="w") as file:
        json.dump(obj=result, fp=file, indent=4)

    logger.info(msg=f"Wrote {fn_result_dpe}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdmc-matrix",
        "-mm",
        required=False,
        type=str,
        help="MdMC matrix chosen for DPE generation",
        default="000011012",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=False,
        type=str,
        help="Number of levels considered for training (9 for LostLevels)",
        default="0123456789L",
    )
    parser.add_argument(
        "--layer",
        "-l",
        required=False,
        choices=sorted(LAYERS),
        type=str,
        help="Tile layer to train on. Background dicts want a single-chapter "
        "--dataset: the tileset is a property of the chapter",
        default="fg",
    )
    args: Args = parser.parse_args(namespace=Args())

    main(args)
