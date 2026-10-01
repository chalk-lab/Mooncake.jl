@testset "linear_algebra" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:linear_algebra))
end

if Base.get_extension(Mooncake, :MooncakeChainRulesExt) !== nothing
    rng = StableRNG(123)
    @testset "svd, $P, $m×$n" for P in [Float64, Float32], (m, n) in [(3, 3), (5, 3)]
        TestUtils.test_rule(rng, svd, randn(rng, P, m, n); mode=ReverseMode)
    end

    @testset "svdvals, $P, $m×$n" for P in [Float32, Float64, ComplexF32, ComplexF64],
        (m, n) in [(3, 3), (5, 3), (3, 5)]

        TestUtils.test_rule(rng, svdvals, randn(rng, P, m, n); mode=ReverseMode)
    end

    @testset "eigvals, $P, $uplo, view=$use_view" for P in [Float32, Float64],
        uplo in [:U, :L],
        use_view in [false, true]

        A = randn(rng, P, 5, 5)
        S = Symmetric(use_view ? view(A, 2:4, 2:4) : A, uplo)
        TestUtils.test_rule(rng, eigvals, S; mode=ReverseMode)
    end

    @testset "spectral compositions" for f in (
        A -> opnorm(A, 2), A -> sum(eigvals(Symmetric(A)))
    )
        TestUtils.test_rule(rng, f, randn(rng, 3, 3); mode=ReverseMode, is_primitive=false)
    end
end

# The registry cannot replicate a pinned struct seed into selected lanes.
@testset "MulAddMul inactive lanes" for P in (Float64, ComplexF64), a in (1, 2), b in (0, 2)
    p = LinearAlgebra.MulAddMul(P(a), P(b))
    da = Mooncake._scalar_ndual(P(a), (P(3), ntuple(_ -> zero(P), 7)...))
    db = Mooncake._scalar_ndual(P(b), (P(5), ntuple(_ -> zero(P), 7)...))
    d = Mooncake.Lifted{typeof(p),8}(p, Mooncake.ImmutableDual((; alpha=da, beta=db)))
    for args in ((P(7),), (P(7), P(11)))
        out = Mooncake.frule!!(d, map(x -> Mooncake.zero_lifted(Val(8), x), args)...)
        @test tangent(out, 1) == P(length(args) == 1 ? 21 : 76)
        @test all(k -> iszero(tangent(out, k)), 2:8)
    end
end
