"""Procedural generation of room contents, from estimation through to levels.

`dpe_generation` counts tile patterns over a corpus of rooms into a
probability dict, `rooms_generation` samples raw rooms back out of one, and
`level_generation` wraps that sampling in the playability checks and retries
that turn a skeleton into a finished level.

The first two take a `--layer {fg,bg}`, which picks the corpus, the vocabulary
and the empty-room seed together. Only the foreground layer reaches
`level_generation`: a skeleton carves exits and places respawn points, which a
decorative layer has no equivalent of, so how the two are stitched into one level
is still an open question.
"""
