using Pkg

@testset "pin_develop_or_skip" begin
    @testset "$failure" for failure in (
        :compat,
        :transitive,
        :transitive_reverse,
        :update,
        :pin,
        :develop,
        :develop_with_cap,
        :nested_with_cap,
    )
        transitive = failure in (:transitive, :transitive_reverse)
        capped = failure in (:compat, :develop_with_cap, :nested_with_cap)
        should_skip = failure == :compat || transitive
        mktempdir() do root
            suite = joinpath(root, "test", "ext", "function_wrappers")
            mkpath(suite)
            mooncake_uuid = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
            # Newer Pkg versions ignore impossible stdlib compat, so use a local package.
            dependency_uuid = "11111111-2222-3333-4444-555555555555"
            for (dir, name, uuid, version) in (
                (root, "Mooncake", mooncake_uuid, "2.0.0"),
                (joinpath(root, "dependency"), "Dependency", dependency_uuid, "1.0.0"),
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
                if name == "FunctionWrappers" && transitive
                    project["deps"] = Dict("Dependency" => dependency_uuid)
                elseif name == "Dependency" && transitive
                    project["weakdeps"] = Dict("Mooncake" => mooncake_uuid)
                    project["compat"] = Dict("Mooncake" => "1")
                elseif name == "FunctionWrappers"
                    project["deps"] = Dict("Mooncake" => mooncake_uuid)
                    project["compat"] = Dict("Mooncake" => capped ? "1" : "1, 2")
                elseif dir == root && failure in (:develop, :develop_with_cap)
                    project["deps"] = Dict("Dependency" => dependency_uuid)
                    project["compat"] = Dict("Dependency" => "999")
                end
                open(joinpath(dir, "Project.toml"), "w") do io
                    Pkg.TOML.print(io, project)
                end
            end
            cp(
                joinpath(@__DIR__, "pin_develop_or_skip.jl"),
                joinpath(root, "test", "ext", "pin_develop_or_skip.jl"),
            )
            entry = read(
                joinpath(@__DIR__, "function_wrappers", "function_wrappers.jl"), String
            )
            entry =
                first(split(entry, "\n\n")) * "\nerror(\"setup unexpectedly proceeded\")"
            failure == :pin &&
                (entry = replace(entry, "\"FunctionWrappers\"" => "\"MissingTarget\""))
            write(joinpath(suite, "function_wrappers.jl"), entry)
            # Exercise both orientations of Pkg's diagnostic, and a nested Mooncake mention.
            prefix = if failure == :nested_with_cap
                " └─restricted by compatibility requirements with Other [22222222] to versions: uninstalled — no versions left\n   └─Other [22222222] log:\n    "
            else
                ""
            end
            message =
                "Unsatisfiable requirements detected for package Dependency [11111111]:\n" *
                " Dependency [11111111] log:\n" *
                prefix *
                " └─restricted by compatibility requirements with Mooncake [da2b9cff] to versions: uninstalled — no versions left\n"
            script = """
                using Pkg
                Pkg.offline(true)
                Pkg.activate($(repr(suite)))
                Pkg.develop([
                    PackageSpec(path=$(repr(joinpath(root, "dependency")))),
                    PackageSpec(path=$(repr(joinpath(root, "old")))),
                    PackageSpec(path=$(repr(joinpath(root, "target")))),
                ])
                if $(failure == :update)
                    p = Pkg.TOML.parsefile(Base.active_project())
                    p["compat"] = Dict("Dependency" => "999")
                    open(Base.active_project(), "w") do io
                        Pkg.TOML.print(io, p)
                    end
                end
                if $(failure in (:transitive_reverse, :nested_with_cap))
                    @eval Pkg develop(; path) = throw(Resolve.ResolverError($(repr(message))))
                end
                include($(repr(joinpath(suite, "function_wrappers.jl"))))
                """
            output = IOBuffer()
            color = transitive ? "yes" : "no"
            cmd = addenv(
                `$(Base.julia_cmd()) --startup-file=no --color=$color --project=$suite -e $script`,
                "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            )
            process = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output))
            log = String(take!(output))
            @test success(process) == should_skip
            @test occursin("skipped: incompatible with Mooncake", log) == should_skip
            if !should_skip
                @test occursin("ERROR", log)
                @test !occursin("setup unexpectedly proceeded", log)
                if failure in (:develop, :develop_with_cap)
                    @test occursin(
                        "Unsatisfiable requirements detected for package Dependency [11111111]:",
                        log,
                    )
                end
            end
        end
    end
end
