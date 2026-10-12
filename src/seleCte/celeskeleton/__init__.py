"""The level skeleton: room sizes, positions and the exits linking them.

This is the structural stage of the pipeline, run before any tile is sampled -
it lays out where the rooms of a level sit and how they connect, leaving the
contents of each to `pcg`.
"""

from seleCte.celeskeleton.celeskeleton import (
    Celeskeleton,
    generate_celeskeleton,
)

__all__: list[str] = [
    "Celeskeleton",
    "generate_celeskeleton",
]
