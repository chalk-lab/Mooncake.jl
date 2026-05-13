# Used in the `extra` CI job.
using Test
using Pkg

project_dir = joinpath(@__DIR__, ENV["TEST_TYPE"], ENV["LABEL"])
Pkg.activate(project_dir)
Pkg.develop(; path=joinpath(@__DIR__, ".."))

# TODO: temporary — JET's analysis is unreliable on Julia 1.13 prerelease.
# Remove once JET supports 1.13.
using Mooncake
Mooncake.TestUtils.test_hook(::Any, ::typeof(Mooncake.TestUtils.test_opt), ::Any...) = nothing
Mooncake.TestUtils.test_hook(::Any, ::typeof(Mooncake.TestUtils.report_opt), tt) = nothing

include(joinpath(project_dir, ENV["LABEL"] * ".jl"))
