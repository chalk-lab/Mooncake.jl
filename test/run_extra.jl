# Used in the `extra` CI job.
using Test
using Pkg
using TOML

project_dir = joinpath(@__DIR__, ENV["TEST_TYPE"], ENV["LABEL"])
Pkg.activate(project_dir)

# TODO: temporary — [sources] in Mooncake's Project.toml doesn't propagate
# through Pkg.develop, so projects that depend on JET need the fork pin here.
# Only required on Julia 1.13+ where upstream JET hasn't published a compatible
# version yet; the fork itself drops < 1.12 in its compat bounds and would fail
# to install on the buildkite 1.10/1.12 lanes. Remove once JET supports 1.13.
if VERSION >= v"1.13-"
    let proj = TOML.parsefile(joinpath(project_dir, "Project.toml"))
        if any(haskey(get(proj, k, Dict()), "JET") for k in ("deps", "weakdeps", "extras"))
            Pkg.add(Pkg.PackageSpec(;
                url="https://github.com/sunxd3/JET.jl",
                rev="67290628ffb9a36b65bc3f3a749cce019ece21d6",
            ))
        end
    end
end

Pkg.develop(; path=joinpath(@__DIR__, ".."))

# TODO: temporary — JET's analysis is unreliable on Julia 1.13 prerelease.
# Remove once JET supports 1.13.
using Mooncake
Mooncake.TestUtils.test_hook(::Any, ::typeof(Mooncake.TestUtils.test_opt), ::Any...) = nothing
Mooncake.TestUtils.test_hook(::Any, ::typeof(Mooncake.TestUtils.report_opt), tt) = nothing

include(joinpath(project_dir, ENV["LABEL"] * ".jl"))
