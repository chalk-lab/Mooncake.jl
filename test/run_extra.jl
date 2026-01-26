# Uses in the `extra` CI job.
using Test

@static if VERSION < v"1.13-"
    using Pkg
    Pkg.add("JET")
    using JET
end

include(joinpath(@__DIR__, ENV["TEST_TYPE"], ENV["LABEL"], ENV["LABEL"] * ".jl"))
