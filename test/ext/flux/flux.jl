include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "Flux")

using Mooncake, StableRNGs, Test, Flux
using Mooncake.TestUtils: TestCase, test_rule

@testset "flux" begin
    # This suite overrides the mode at construction: these cases cover reverse rules.
    test_cases = vcat(
        map([Float32, Float64]) do P
            return TestCase(
                Flux.Losses.mse,
                randn(StableRNG(1), P, 3),
                randn(StableRNG(2), P, 3);
                mode=Mooncake.ReverseMode,
            )
        end,
    )
    for (tc, name) in zip(test_cases, Mooncake.TestUtils._test_case_names(test_cases))
        rng = StableRNG(123)
        test_rule(rng, tc; name)
    end
end
