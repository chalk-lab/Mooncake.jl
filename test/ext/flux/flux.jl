include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "Flux")

using Mooncake, StableRNGs, Test, Flux
using Mooncake.TestUtils: test_rule

@testset "flux" begin
    @testset "$f, $(typeof(fargs))" for (
        interface_only, perf_flag, is_primitive, f, fargs...
    ) in vcat(
        map([Float32, Float64]) do P
            return (
                false,
                :none,
                true,
                Flux.Losses.mse,
                randn(StableRNG(1), P, 3),
                randn(StableRNG(2), P, 3),
            )
        end,
    )
        rng = StableRNG(123)
        mode = Mooncake.ReverseMode
        test_rule(rng, f, fargs...; interface_only, perf_flag, is_primitive, mode)
    end
end
