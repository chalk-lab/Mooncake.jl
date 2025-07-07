using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake.TestUtils: DD_ENABLED

@test DD_ENABLED

include(joinpath(@__DIR__, "..", "front_matter.jl"))

include(joinpath(@__DIR__, "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "tangents.jl"))
include(joinpath(@__DIR__, "..", "codual.jl"))
include(joinpath(@__DIR__, "..", "stack.jl"))
