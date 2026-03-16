# Used in the `extra` CI job.
using Test
include(joinpath(@__DIR__, ENV["TEST_TYPE"], ENV["LABEL"], ENV["LABEL"] * ".jl"))
