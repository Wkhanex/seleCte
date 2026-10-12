"""Import-anchored locations of the data directories shipped inside the package.

Every path derives from this module's own file, so they resolve the same way
whatever the working directory is and whichever entry point does the importing.
"""

from pathlib import Path

# This file lives AT the package root, so its parent IS the package root —
# regardless of which script imports it or what the working directory is.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

PCG_DIR: Path = PACKAGE_ROOT / "pcg"
PE_DICTS_DIR: Path = PCG_DIR / "preprocessing" / "probability_estimation_dicts"
DATA_PCG_DIR: Path = PCG_DIR / "preprocessing" / "data_pcg_ready"

# Named `GAME_DATA_DIR` rather than `DATA_DIR`: `seleCte.utils.utils` already binds
# `DATA_DIR` to the package's own `utils/data`, and two same-named data directories
# one import apart is a trap. This one holds data extracted from the game.
GAME_DATA_DIR: Path = PACKAGE_ROOT / "data"
LEVELS_DIR: Path = GAME_DATA_DIR / "levels" / "original"

# The recombined corpus: written by `pcg.preprocessing.recombine_fg_entities`, read
# by `pcg.dpe_generation`. Shared so producer and consumer cannot drift apart.
ALL_ROOMS_DIR: Path = DATA_PCG_DIR / "all_rooms"

# The background corpus, the same contract one layer down: written by
# `pcg.preprocessing.collect_bg_rooms`, read by `pcg.dpe_generation --layer bg`. Kept
# in its own directory rather than mixed into `ALL_ROOMS_DIR`, because
# `extract_dataset_from_filepath` selects on the chapter id alone and would otherwise
# train one dict over both layers - whose alphabets overlap without agreeing on what
# a symbol means.
ALL_ROOMS_BG_DIR: Path = DATA_PCG_DIR / "all_rooms_bg"
