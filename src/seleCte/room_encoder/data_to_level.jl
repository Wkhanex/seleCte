# Encode a generated level back into a map Celeste can load. This is the tail of
# the pipeline, the mirror image of `data_loader/data_loader.jl`.
#
#     julia --project=. src/seleCte/room_encoder/data_to_level.jl \
#         <level-dir> [map-name] [output-dir]
#
# `<level-dir>` is a folder written by `Celeskeleton.save`: one `room_<n>.csv`
# per room, space-separated (`np.savetxt`'s default), plus a `data.json` giving
# each room its `origin`, `size` and `exits`. Rooms are read as `room_1` through
# `room_<N>`, N being the number of entries in `data.json`.
#
# Each csv is a single grid in the post-recombination alphabet, and Maple turns
# that one grid into both of a room's layers: `FgTiles` keeps the characters
# that name a real tileset and blanks everything else to air, while `entityMap`
# reads exactly the characters `FgTiles` blanked and turns them into entities.
# The alphabet was chosen to make that work - see ALPHABET.md for the full
# symbol-by-symbol correspondence, and for the round-trip caveats.
#
# The `.bin` is written as a `Side`, not a bare `Map`: alongside the rooms it carries
# a `meta` element holding the chapter settings, without which the crystal heart in the
# end room is a collectible rather than the end of the level. See `chapter_metadata`.
#
# There is a ready-made fixture to try it on:
#
#     julia --project=. src/seleCte/room_encoder/data_to_level.jl \
#         src/seleCte/room_encoder/fixtures/round_trip_room round_trip_room

using DelimitedFiles
using JSON
using Maple

const DEFAULT_MAP_NAME = "default_map"
const DEFAULT_OUTPUT_DIRNAME = "generated_bin_files"

# Celeste positions everything in pixels; the whole project thinks in tiles.
const TILE_SIZE = 8

const USAGE = """
Usage: julia --project=. src/seleCte/room_encoder/data_to_level.jl <level-dir> [map-name] [output-dir]

  <level-dir>    folder written by `Celeskeleton.save`: room_<n>.csv files plus
                 a data.json.
  [map-name]     name of the map, and the stem of the .bin. Defaults to
                 $(DEFAULT_MAP_NAME).
  [output-dir]   where the .bin goes, created if missing.
                 Defaults to ./$(DEFAULT_OUTPUT_DIRNAME) under the current directory.
"""


"""
    room_origin(room_name, metadata)

The room's origin from `data.json`, in tiles, as `[x, y]`.

Reported rather than returned as `nothing`, because a missing entry means the
`data.json` and the csvs on disk disagree about which rooms exist, and the
resulting `.bin` would be silently short a room.
"""
function room_origin(room_name::AbstractString, metadata::AbstractDict)
    entry = get(metadata, room_name, nothing)
    entry === nothing && error("data.json has no entry for $room_name")

    origin = get(entry, "origin", nothing)
    origin === nothing && error("data.json entry for $room_name has no `origin`")

    return origin
end


"""
    create_room(room_nb, metadata, lvl_data_folder)

Build one Maple `Room` from `room_<room_nb>.csv` and its `data.json` entry.

Both layers come out of the same matrix: `FgTiles` for the solids, `entityMap`
for everything the alphabet spells as an entity.
"""
function create_room(room_nb::Integer, metadata::AbstractDict, lvl_data_folder::AbstractString)
    name = "room_$room_nb"
    path = joinpath(lvl_data_folder, "$name.csv")
    isfile(path) || error("data.json lists $name but $path does not exist")

    # `Celeskeleton.save` writes these with `np.savetxt`, i.e. space-separated.
    # `Char` is the parse target, so a cell that is not exactly one character
    # fails here rather than becoming an unencodable tile further down.
    data = readdlm(path, ' ', Char)
    fg_tiles = FgTiles(data)
    origin = room_origin(name, metadata)

    # Maple's `size(::Tiles)` already returns (width, height) in pixels.
    width, height = size(fg_tiles)

    return Room(
        name=name,
        fgTiles=fg_tiles,
        # The skeleton's y axis points up and `origin` is the room's *bottom*-left
        # corner; Celeste's y axis points down and `position` is its *top*-left one.
        # Flipping the sign alone moves the bottom edge to where the top belongs,
        # which drops every room by its own height - a level whose rooms are all the
        # same height survives that as a uniform translation, which is why the
        # single-room fixture never caught it, but rooms of differing heights end up
        # overlapping. Subtracting the height is what makes the flip rigid.
        position=(origin[1] * TILE_SIZE, -(origin[2] * TILE_SIZE + height)),
        size=(width, height),
        entities=entityMap(data),
    )
end


"""
    chapter_metadata()

The chapter settings Celeste keeps in the map's `meta` element, shaped the way `Side`
wants them: `meta` carries the chapter-wide attributes, `mode` the per-side ones, and
`audiostate` the music and ambience nested under `mode`.

A bare `Map` has no metadata at all - `encodeMap` writes `levels`, `Style` and `Filler`
and stops - so a map encoded that way loads, but as a chapter with every setting left
at Everest's default. `HeartIsEnd` is the one that matters today: `Celeskeleton` places
exactly one crystal heart, in the end room, and that heart is how a generated level is
meant to finish. Without the flag, collecting it just pockets a collectible and the
level never ends.

The values come from Maple's own `default_meta`/`default_mode`/`default_audiostate`,
which is where the fork records this project's choices - `Project.toml` explains why
there is a fork. They are not exported, hence the qualified names. Against upstream
Maple `HeartIsEnd` is `false`, so a level encoded there will not end on the heart -
which is moot, since upstream cannot encode the heart in the first place.
"""
function chapter_metadata()
    mode = merge(
        Maple.default_mode,
        Dict{String,Any}("audiostate" => Maple.default_audiostate),
    )

    return Dict{String,Any}(
        "meta" => merge(Maple.default_meta, Dict{String,Any}("mode" => mode)),
    )
end


"""
    create_map_file(metadata, lvl_data_folder, map_name, output_dir)

Encode every room into `<output_dir>/<map_name>.bin`, returning its path.

`metadata` here is the `data.json` payload - the rooms' geometry - and is unrelated to
the chapter metadata `chapter_metadata` builds.
"""
function create_map_file(
    metadata::AbstractDict,
    lvl_data_folder::AbstractString,
    map_name::AbstractString,
    output_dir::AbstractString,
)
    rooms = [create_room(i, metadata, lvl_data_folder) for i in 1:length(metadata)]

    mkpath(output_dir)
    output_path = joinpath(output_dir, "$map_name.bin")
    # Named `celeste_map`: `map` is `Base.map`, and shadowing it here is how the
    # notebook this script came from kept failing.
    celeste_map = Map(map_name, rooms)
    # A `Side` rather than the bare `Map`, because metadata only rides along on a
    # Side - `encodeSide` is `encodeMap` plus the `meta` element.
    encodeSide(Side(celeste_map, chapter_metadata()), output_path)

    return output_path
end


function main(args)
    if isempty(args) || first(args) in ("-h", "--help")
        println(USAGE)
        return nothing
    end

    lvl_data_folder = abspath(args[1])
    map_name = length(args) >= 2 ? args[2] : DEFAULT_MAP_NAME
    output_dir = length(args) >= 3 ? abspath(args[3]) : joinpath(pwd(), DEFAULT_OUTPUT_DIRNAME)

    metadata_path = joinpath(lvl_data_folder, "data.json")
    isfile(metadata_path) || error("No data.json in $lvl_data_folder")

    metadata = JSON.parsefile(metadata_path)
    output_path = create_map_file(metadata, lvl_data_folder, map_name, output_dir)
    println("Done - encoded $(length(metadata)) rooms into $output_path")

    return nothing
end


main(ARGS)
