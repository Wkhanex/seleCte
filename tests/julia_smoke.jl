# Smoke test for the two Julia scripts, run from the repo root:
#
#     julia --project=. tests/julia_smoke.jl
#
# Deliberately small. It does not try to test Maple, and it does not re-check things
# the Python suite already covers - it pins down the handful of contracts that span a
# boundary nothing else executes, which is where every bug in these scripts has come
# from so far:
#
#   - the alphabet correspondence in room_encoder/ALPHABET.md, which is prose and a
#     table until something encodes the fixture and reads the entities back;
#   - the chapter metadata, without which the crystal heart is a collectible and the
#     generated level cannot be finished;
#   - the skeleton -> Celeste coordinate flip, which is invisible on a level whose rooms
#     are all one height and silently overlaps them as soon as they are not;
#   - the file names and separator the Python recombination stage reads;
#   - the entity names in tiles_info.json, which have to be the ones Maple emits.
#
# `Test` comes from `@stdlib` rather than `[deps]`, so the runtime environment stays the
# three packages the scripts actually import.

using Test
using JSON
using Maple

const REPO = dirname(@__DIR__)
const ENCODER = joinpath(REPO, "src", "seleCte", "room_encoder", "data_to_level.jl")
const LOADER = joinpath(REPO, "src", "seleCte", "data_loader", "data_loader.jl")
const FIXTURE = joinpath(REPO, "src", "seleCte", "room_encoder", "fixtures", "round_trip_room")
const VOCABULARY = joinpath(REPO, "src", "seleCte", "utils", "data", "tiles_info.json")

# The scripts are CLIs that end in `main(ARGS)`, so drive them as CLIs: a subprocess
# exercises argument handling and the `--project` resolution the docs tell people to
# use, where `include` would quietly skip both.
function run_script(script::AbstractString, args::AbstractString...)
    cmd = `$(Base.julia_cmd()) --project=$REPO $script $(collect(args))`
    run(pipeline(cmd, stdout=devnull))

    return nothing
end

"""
    entity_rows(room)

`(name, x, y, width, height)` per entity, in the order they come back, with `nothing`
for a dimension the entity does not carry. Comparable against a literal table.
"""
function entity_rows(room)
    return map(room.entities) do entity
        data = Dict(entity)
        (entity.name, data["x"], data["y"], get(data, "width", nothing), get(data, "height", nothing))
    end
end


@testset "data_to_level.jl" begin
    mktempdir() do out
        run_script(ENCODER, FIXTURE, "round_trip_room", out)
        side = loadSide(joinpath(out, "round_trip_room.bin"))
        room = only(side.map.rooms)

        @testset "the fixture encodes to the table in ALPHABET.md" begin
            # Reading order, since `entityMap` scans row-major and `encodeMap` keeps it.
            # `blackGem` is the eleventh: upstream Maple has no `H` glyph and would give
            # ten, so this doubles as an assertion that the pin is the fork.
            @test entity_rows(room) == [
                ("refill", 164, 28, nothing, nothing),
                ("blackGem", 256, 32, nothing, nothing),
                ("refill", 148, 52, nothing, nothing),
                ("refill", 116, 76, nothing, nothing),
                ("booster", 212, 84, nothing, nothing),
                ("refill", 204, 116, nothing, nothing),
                ("infiniteStar", 116, 140, nothing, nothing),
                ("refill", 196, 140, nothing, nothing),
                ("player", 48, 152, nothing, nothing),
                ("jumpThru", 32, 160, 24, nothing),
                ("spikesUp", 112, 176, 192, nothing),
            ]

            # `size(::Tiles)` is (width, height) in pixels, off a 40x23 room.
            @test Int.(room.size) == (320, 184)
        end

        @testset "the heart is what ends the level" begin
            # A bare `Map` carries no metadata at all, so this is really asserting that
            # the encoder still writes a `Side`.
            @test haskey(side.data, "meta")
            @test side.data["meta"]["mode"]["HeartIsEnd"] == true
        end
    end
end


@testset "rooms of different heights stay adjacent" begin
    # The regression guard the committed fixture cannot be: it is one room, and a room
    # on its own absorbs a wrong y-flip as a translation of the whole map. Two rooms of
    # different heights do not - they overlap.
    mktempdir() do level
        # 40x23 below, 40x46 directly above it, exactly as `Celeskeleton.connect` places
        # them: origin is the bottom-left corner and y grows upwards.
        for (name, w, h, origin) in (("room_1", 40, 23, [0, 0]), ("room_2", 40, 46, [0, 23]))
            open(joinpath(level, "$name.csv"), "w") do io
                for _ in 1:h
                    println(io, join(fill('1', w), ' '))
                end
            end
        end

        open(joinpath(level, "data.json"), "w") do io
            write(io, JSON.json(Dict(
                "room_1" => Dict("origin" => [0, 0], "size" => [40, 23], "exits" => Dict()),
                "room_2" => Dict("origin" => [0, 23], "size" => [40, 46], "exits" => Dict()),
            )))
        end

        mktempdir() do out
            run_script(ENCODER, level, "two_rooms", out)
            rooms = sort(loadMap(joinpath(out, "two_rooms.bin")).rooms, by=r -> r.name)
            spans = [(Int(r.position[2]), Int(r.position[2]) + Int(r.size[2])) for r in rooms]

            # Celeste's y points down, so the room that was on top ends up first.
            @test spans == [(-184, 0), (-552, -184)]
            # Touching, not overlapping: the top of one is the bottom of the other.
            @test spans[1][1] == spans[2][2]
        end
    end
end


@testset "data_loader.jl keeps the contract the Python side reads" begin
    mktempdir() do maps
        mktempdir() do dump
            run_script(ENCODER, FIXTURE, "1-TestChapter", maps)
            run_script(LOADER, maps, dump)

            # `recombine_fg_entities` globs `*/*_fg.csv`, pairs each with the
            # `_entities.json` beside it, and takes the chapter id off the directory
            # name up to the first dash.
            chapter = joinpath(dump, "1-TestChapter")
            @test isdir(chapter)
            @test all(isfile, joinpath.(chapter, "room_1" .* ["_fg.csv", "_bg.csv", "_obj.csv", "_entities.json"]))

            # Read back with `pd.read_csv(..., header=None, sep=";")`: one character per
            # cell, no header, semicolons.
            fg = readlines(joinpath(chapter, "room_1_fg.csv"))
            @test length(fg) == 23
            @test all(row -> length(split(row, ';')) == 40, fg)
            @test all(row -> all(cell -> length(cell) == 1, split(row, ';')), fg)

            # Sparse per-room keys, but `__name`/`x`/`y` are what recombination needs.
            entities = JSON.parsefile(joinpath(chapter, "room_1_entities.json"))
            @test !isempty(entities)
            @test all(e -> haskey(e, "__name") && haskey(e, "x") && haskey(e, "y"), entities)
        end
    end
end


@testset "tiles_info.json names entities the way Maple does" begin
    # Both sides key entities by Celeste's `__name`, so where both know a glyph they have
    # to agree - a name matching nothing is dropped exactly like an unmapped entity, which
    # is how `feather` and `bumper` stayed dead symbols without anything failing.
    symbols = JSON.parsefile(VOCABULARY)["entity_symbols"]
    maple_names = Dict(glyph => func(0, 0).name for (glyph, func) in Maple.default_entity_map)

    for (name, glyph) in symbols
        char = only(glyph)

        # `O`/`Q` are both Maple `booster`s; the Python side is what splits them, on the
        # `red` flag. `L` has no upstream glyph at all - see ALPHABET.md.
        char in ('O', 'Q') && continue
        haskey(maple_names, char) || continue

        @test name == maple_names[char]
    end
end
