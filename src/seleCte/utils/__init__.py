"""Shared room-level helpers: MdMC sampling, A* pathfinding and metrics.

See `seleCte.utils.utils` for the full picture; this re-exports the parts the
pipeline stages use.
"""

from seleCte.utils.utils import (
    ALL_TILES_AND_ENTITIES,
    DEFAULT_MDMC_MATRIX,
    ENTITY_SYMBOLS,
    MDMC_KEY,
    VALID_BG_TILES,
    VALID_FG_TILES,
    astar,
    check_dpe_matches_mask,
    generate_room_batch,
    generate_room_data,
    generate_temp_room,
    get_all_submatrices,
    load_dpe,
    mdmc_offsets,
    mdmc_target_offset,
    parse_mdmc_matrix,
    submatrix_to_count,
)

__all__: list[str] = [
    "ALL_TILES_AND_ENTITIES",
    "DEFAULT_MDMC_MATRIX",
    "ENTITY_SYMBOLS",
    "MDMC_KEY",
    "VALID_BG_TILES",
    "VALID_FG_TILES",
    "astar",
    "check_dpe_matches_mask",
    "generate_room_batch",
    "generate_room_data",
    "generate_temp_room",
    "get_all_submatrices",
    "load_dpe",
    "mdmc_offsets",
    "mdmc_target_offset",
    "parse_mdmc_matrix",
    "submatrix_to_count",
]
