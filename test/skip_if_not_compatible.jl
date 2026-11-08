using Pkg

function skip_if_not_compatible(target::AbstractString, mooncake::AbstractString)
    project = Base.active_project()
    try
        Pkg.activate(; temp=true)
        Pkg.add(target)
        Pkg.pin(target)
        Pkg.develop(; path=mooncake)
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        @warn "$target skipped: incompatible with Mooncake\n$(err.msg)"
        if haskey(ENV, "GITHUB_STEP_SUMMARY")
            println("::warning title=Skipped::$target incompatible with Mooncake")
            open(ENV["GITHUB_STEP_SUMMARY"], "a") do io
                println(io, "**$target** skipped: incompatible with Mooncake")
            end
        end
        exit(0)
    finally
        Pkg.activate(project)
    end
end
