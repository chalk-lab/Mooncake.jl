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
Mooncake. A develop-time `Pkg.Resolve.ResolverError` is skipped only when a pinned target's
declared Mooncake compat excludes the checked-out version. A warning is logged and the process
exits successfully (`exit(0)`). Update and pin failures, and conflicts without this evidence
(including transitive conflicts), fail loudly.

!!! warning "A skipped suite is indistinguishable from a passing one"
    `exit(0)` means a skip shows up as green. That is deliberate — a downstream package that has
    not yet widened its Mooncake compat should not turn CI red — but it is worth knowing that the
    signal is a log line and a GitHub annotation, not a test result, and neither appears in the
    aggregate summary.

    This matters most immediately after a breaking release, when many downstreams still cap the
    previous version and several suites skip at once. Measured on this branch at Mooncake 0.6.0,
    `test/ext/differentiation_interface` already takes the skip path. Do not read a green
    ecosystem run as coverage without checking which suites actually ran.
"""
function pin_develop_or_skip(dir::AbstractString, targets::AbstractString...)
    Pkg.activate(dir)
    # Update before pinning so a cached manifest cannot lock in an old target version.
    Pkg.update()
    Pkg.pin(collect(targets))
    pinned_targets = filter(
        p -> p.name in targets && p.is_pinned, collect(values(Pkg.dependencies()))
    )
    mooncake_path = joinpath(@__DIR__, "..", "..")
    mooncake = Pkg.Types.read_project(joinpath(mooncake_path, "Project.toml"))
    try
        Pkg.develop(; path=mooncake_path)
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        incompatible = any(pinned_targets) do target
            project = Pkg.Types.read_project(joinpath(target.source, "Project.toml"))
            depends =
                get(project.deps, "Mooncake", nothing) == mooncake.uuid ||
                get(project.weakdeps, "Mooncake", nothing) == mooncake.uuid
            return depends &&
                   haskey(project.compat, "Mooncake") &&
                   !(mooncake.version in project.compat["Mooncake"].val)
        end
        incompatible || rethrow()
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
