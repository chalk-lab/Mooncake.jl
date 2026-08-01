using Pkg

"""
    pin_develop_or_skip(dir::AbstractString, targets::AbstractString...)

Activate the test environment in `dir`, update it, pin the target package(s), then `develop`
the checked-out Mooncake into it. `include` this file from an ext/integration test entry point
(adjusting the relative path) and call it first, passing `@__DIR__` and the package(s) that suite
targets:

    pin_develop_or_skip(@__DIR__, "Flux")  # a single target
    pin_develop_or_skip(@__DIR__, "OrdinaryDiffEq", "SciMLSensitivity")  # several targets

Pinning stops the resolver from silently downgrading a target to accommodate an incompatible
Mooncake. If a pinned target cannot coexist with the checked-out Mooncake, the resulting
`Pkg.Resolve.ResolverError` is caught, a warning is logged, and the process exits successfully
(`exit(0)`) so the suite is skipped instead of failing CI; any other error is re-raised.
"""
function pin_develop_or_skip(dir::AbstractString, targets::AbstractString...)
    Pkg.activate(dir)
    try
        # `update` (not just `resolve`) so the pin locks the target's *current* version: a stale
        # manifest left by a cached CI environment could otherwise pin a pre-downgraded target and
        # hide the incompatibility. It also populates the manifest so the pin can find the target.
        Pkg.update()
        Pkg.pin(collect(targets))
        Pkg.develop(; path=joinpath(@__DIR__, "..", ".."))
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        name = basename(dir)
        @warn "$name skipped: incompatible with Mooncake"
        if haskey(ENV, "GITHUB_STEP_SUMMARY")
            println("::warning title=Skipped::$name incompatible with Mooncake")
            open(ENV["GITHUB_STEP_SUMMARY"], "a") do io
                println(io, "**$name** skipped: incompatible with Mooncake")
            end
        end
        exit(0)
    end
end
