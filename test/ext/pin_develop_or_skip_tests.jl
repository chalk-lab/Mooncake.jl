using Pkg, TOML

@testset "pin_develop_or_skip" begin
    for failure in (:compat, :update, :pin, :develop)
        mktempdir() do root
            suite = joinpath(root, "test", "ext", "function_wrappers")
            mkpath(suite)
            mooncake_uuid = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
            for (dir, name, uuid, version) in (
                (root, "Mooncake", mooncake_uuid, "2.0.0"),
                (joinpath(root, "old"), "Mooncake", mooncake_uuid, "1.0.0"),
                (
                    joinpath(root, "target"),
                    "FunctionWrappers",
                    "069b7b12-0de2-55c6-9aab-29f3d0a68a2e",
                    "1.0.0",
                ),
            )
                mkpath(joinpath(dir, "src"))
                write(joinpath(dir, "src", "$name.jl"), "module $name end")
                project = Dict{String,Any}(
                    "name" => name, "uuid" => uuid, "version" => version
                )
                if name == "FunctionWrappers"
                    project["deps"] = Dict("Mooncake" => mooncake_uuid)
                    project["compat"] = Dict(
                        "Mooncake" => failure == :compat ? "1" : "1, 2"
                    )
                elseif dir == root && failure == :develop
                    project["deps"] = Dict(
                        "Random" => "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
                    )
                    project["compat"] = Dict("Random" => "999")
                end
                open(joinpath(dir, "Project.toml"), "w") do io
                    TOML.print(io, project)
                end
            end
            cp(
                joinpath(@__DIR__, "pin_develop_or_skip.jl"),
                joinpath(root, "test", "ext", "pin_develop_or_skip.jl"),
            )
            entry = read(
                joinpath(@__DIR__, "function_wrappers", "function_wrappers.jl"), String
            )
            failure == :pin &&
                (entry = replace(entry, "\"FunctionWrappers\"" => "\"MissingTarget\""))
            write(joinpath(suite, "function_wrappers.jl"), entry)
            script = """
                using Pkg, TOML
                Pkg.offline(true)
                Pkg.activate($(repr(suite)))
                Pkg.develop([
                    PackageSpec(path=$(repr(joinpath(root, "old")))),
                    PackageSpec(path=$(repr(joinpath(root, "target")))),
                ])
                if $(failure == :update)
                    p = TOML.parsefile(Base.active_project())
                    p["deps"]["Random"] = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
                    p["compat"] = Dict("Random" => "999")
                    open(Base.active_project(), "w") do io
                        TOML.print(io, p)
                    end
                end
                include($(repr(joinpath(suite, "function_wrappers.jl"))))
                """
            output = IOBuffer()
            cmd = addenv(
                `$(Base.julia_cmd()) --startup-file=no --project=$suite -e $script`,
                "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            )
            process = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output))
            log = String(take!(output))
            @test success(process) == (failure == :compat)
            @test occursin("skipped: incompatible with Mooncake", log) ==
                (failure == :compat)
            if failure != :compat
                @test occursin("ERROR", log)
            end
        end
    end
end
