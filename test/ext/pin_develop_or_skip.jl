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
Mooncake. A develop-time `Pkg.Resolve.ResolverError` is skipped only when Mooncake has
unsatisfiable requirements (including transitive compat constraints), or the conflicting package
has no versions left compatible with Mooncake. A warning is logged and the process exits
successfully (`exit(0)`). Update and pin failures, unrelated resolver conflicts, and all other
develop failures fail loudly.

!!! warning "A skipped suite is indistinguishable from a passing one"
    `exit(0)` means a skip shows up as green. That is deliberate — a downstream package that has
    not yet widened its Mooncake compat should not turn CI red — but it is worth knowing that the
    signal is a log line and a GitHub annotation, not a test result, and neither appears in the
    aggregate summary.

    This matters most immediately after a breaking release, when many downstreams still cap the
    previous version and several suites skip at once. Do not read a green ecosystem run as
    coverage without checking which suites actually ran.
"""
function pin_develop_or_skip(dir::AbstractString, targets::AbstractString...)
    Pkg.activate(dir)
    # Update before pinning so a cached manifest cannot lock in an old target version.
    Pkg.update()
    Pkg.pin(collect(targets))
    mooncake_path = joinpath(@__DIR__, "..", "..")
    try
        Pkg.develop(; path=mooncake_path)
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        # ResolverError has no structured package identity; match name and short UUID after
        # stripping Pkg's colors. Pkg can report either end of a compat conflict, so
        # also accept Mooncake as the final top-level constraint, never a nested mention.
        message = replace(err.msg, r"\e\[[0-9;]*m" => "")
        conflict = match(
            r"\AUnsatisfiable requirements detected for package ([^\n]+):\n", message
        )
        isnothing(conflict) && rethrow()
        conflict[1] == "Mooncake [da2b9cff]" ||
            occursin(
                r"^ └─restricted by compatibility requirements with Mooncake \[da2b9cff\] to versions: [^\n]+ — no versions left$"m,
                message,
            ) ||
            rethrow()
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
