using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using BridgeStan, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

# A trivial Stan model with a standard normal prior on one parameter.
# This exercises log_density with and without the propto/jacobian kwargs.
const STAN_CODE = """
data {}
parameters { real x; }
model { x ~ normal(0, 1); }
"""

@testset "bridgestan" begin
    mktempdir() do dir
        stan_file = joinpath(dir, "test_model.stan")
        write(stan_file, STAN_CODE)
        sm = BridgeStan.StanModel(stan_file=stan_file, data="")

        q = [0.5]
        rng = StableRNG(123)

        # Direct call (no kwargs) — uses the positional rrule!! with defaults.
        test_rule(
            rng, BridgeStan.log_density, sm, q;
            is_primitive=true, mode=Mooncake.ReverseMode,
        )

        # Kwarg call — exercises the Core.kwcall rrule!! with explicit kwargs.
        for (propto, jacobian) in [(true, true), (false, true), (true, false)]
            test_rule(
                rng,
                Core.kwcall,
                (propto=propto, jacobian=jacobian),
                BridgeStan.log_density,
                sm,
                q;
                is_primitive=true,
                mode=Mooncake.ReverseMode,
            )
        end
    end
end
