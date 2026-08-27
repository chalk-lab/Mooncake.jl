using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, Distances, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@testset "Distances" begin
    rng = StableRNG(123456)

    # The rules sit on `_pairwise!`, which always takes observations as columns.
    @testset "_pairwise!, $metric, $P" for metric in (SqEuclidean(), Euclidean()),
        P in (Float64, Float32)

        X = randn(rng, P, 5, 7)
        Y = randn(rng, P, 5, 3)
        test_rule(
            rng, Distances._pairwise!, metric, zeros(P, 7, 7), X; perf_flag=:stability
        )
        test_rule(
            rng, Distances._pairwise!, metric, zeros(P, 7, 3), X, Y; perf_flag=:stability
        )
    end

    # `pairwise` and `pairwise!` reach those rules through their own `dims` handling, which
    # is all these add: element types are covered above.
    @testset "$f, $metric, dims=$dims" for f in (pairwise, pairwise!),
        metric in (SqEuclidean(), Euclidean()),
        dims in (1, 2)

        X = randn(rng, 5, 7)
        Y = randn(rng, dims == 1 ? (3, 7) : (5, 3))
        nX = size(X, dims)
        for (n, xs) in ((nX, (X,)), (size(Y, dims), (X, Y)))
            args = f === pairwise ? xs : (zeros(nX, n), xs...)
            test_rule(
                rng,
                Core.kwcall,
                (; dims),
                f,
                metric,
                args...;
                perf_flag=:none,
                is_primitive=false,
            )
        end
    end

    # A non-zero `thresh` sends the primal down its recomputation branch.
    test_rule(
        rng,
        Core.kwcall,
        (; dims=2),
        pairwise,
        SqEuclidean(1e-12),
        randn(rng, 5, 7);
        perf_flag=:none,
        is_primitive=false,
    )
end
