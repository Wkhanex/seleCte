# Decode Celeste's binary maps into the per-room csv/json dump every later stage
# reads. This is the head of the pipeline: nothing upstream of it is ours.
#
#     julia --project=. src/seleCte/data_loader/data_loader.jl <maps-dir> [output-dir]
#
# `<maps-dir>` is a directory of `.bin` maps - a chapter folder from the game's
# `Content/Maps`, or a mod dump such as Strawberry Jam. Anything that is not a
# `.bin` is skipped, so the `.texturecache.txt` files sitting next to them are
# harmless. Each map becomes one subdirectory of `<output-dir>`, holding four
# files per room:
#
#     <output-dir>/<MapName>/lvl_<id>_fg.csv        foreground tiles
#     <output-dir>/<MapName>/lvl_<id>_bg.csv        background tiles
#     <output-dir>/<MapName>/lvl_<id>_obj.csv       object tiles
#     <output-dir>/<MapName>/lvl_<id>_entities.json entities, one dict each
#
# Two naming contracts downstream depend on the shape of those names, so resist
# the urge to tidy them:
#
#   - `pcg.preprocessing.recombine_fg_entities` pairs `_fg.csv` with
#     `_entities.json` by stripping those exact suffixes, and reads the chapter
#     id off the *directory* name up to the first dash (`1-ForsakenCity` -> `1`).
#     Room names contain dashes of their own (`lvl_-1`), which is why the split
#     happens on the directory and not on the file.
#   - the csvs are `;`-separated with one character per tile, which is what the
#     Python side reads them back as.
#
# Only `_fg.csv` and `_entities.json` are consumed downstream. `_bg.csv` is
# decorative and deliberately ignored, and nothing reads `_obj.csv` at all; both
# are still written so that a dump is a full record of the map rather than of
# the subset today's model happens to want.

using DelimitedFiles
using JSON
using Maple

# The Python side reads these back with `pd.read_csv(..., sep=";")`.
const CSV_DELIMITER = ';'

const DEFAULT_OUTPUT_DIRNAME = "levels"

const USAGE = """
Usage: julia --project=. src/seleCte/data_loader/data_loader.jl <maps-dir> [output-dir]

  <maps-dir>     directory holding the .bin maps to decode. Non-.bin files are
                 skipped.
  [output-dir]   where the dump goes, one subdirectory per map.
                 Defaults to ./$(DEFAULT_OUTPUT_DIRNAME) under the current directory.

To dump the vanilla chapters into the tree the rest of the pipeline expects:

  julia --project=. src/seleCte/data_loader/data_loader.jl \\
      "<Celeste>/Content/Maps" src/seleCte/data/levels/original
"""


"""
    map_name_from_path(path)

Map name from a path: `.../1-ForsakenCity.bin` -> `1-ForsakenCity`. The name
becomes a directory name, and the corpus builder reads the chapter id off its
first character, so the leading `1`/`2`/`L` has to survive intact.
"""
map_name_from_path(path::AbstractString) = first(splitext(basename(path)))


"""
    map_files(dir)

Every `.bin` directly inside `dir`, sorted so two runs over the same directory
produce the same order.
"""
function map_files(dir::AbstractString)
    isdir(dir) || error("Not a directory: $dir")
    bin_names = filter(n -> lowercase(last(splitext(n))) == ".bin", readdir(dir))
    return sort([joinpath(dir, name) for name in bin_names])
end


"""
    write_room(room, out_dir)

Dump one room as its fg/bg/obj csv triplet plus its entities json.

The entity dicts are written exactly as Maple hands them over: keys are sparse
and vary per room, since a Celeste entity only carries the attributes it
actually uses. Anything reading them back has to treat every key as optional.
"""
function write_room(room, out_dir::AbstractString)
    stem = joinpath(out_dir, room.name)

    open("$(stem)_entities.json", "w") do io
        write(io, JSON.json([Dict(entity) for entity in room.entities]))
    end

    writedlm("$(stem)_fg.csv", room.fgTiles.data, CSV_DELIMITER)
    writedlm("$(stem)_bg.csv", room.bgTiles.data, CSV_DELIMITER)
    writedlm("$(stem)_obj.csv", room.objTiles.data, CSV_DELIMITER)

    return nothing
end


"""
    dump_map(bin_path, out_root)

Decode one `.bin` into `<out_root>/<MapName>/`, returning the number of rooms.
"""
function dump_map(bin_path::AbstractString, out_root::AbstractString)
    name = map_name_from_path(bin_path)
    out_dir = joinpath(out_root, name)
    mkpath(out_dir)

    rooms = loadMap(bin_path).rooms
    for room in rooms
        write_room(room, out_dir)
    end

    println("  $name: $(length(rooms)) rooms -> $out_dir")
    return length(rooms)
end


function main(args)
    if isempty(args) || first(args) in ("-h", "--help")
        println(USAGE)
        return nothing
    end

    maps_dir = abspath(args[1])
    out_root = length(args) >= 2 ? abspath(args[2]) : joinpath(pwd(), DEFAULT_OUTPUT_DIRNAME)

    files = map_files(maps_dir)
    isempty(files) && error("No .bin map found in $maps_dir")

    println("Decoding $(length(files)) map(s) from $maps_dir")
    total = sum(dump_map(file, out_root) for file in files)
    println("Done - $total rooms written under $out_root.")

    return nothing
end


main(ARGS)
