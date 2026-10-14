@testset "linear_algebra" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:linear_algebra))

    # Points where the derivative doesn't exist: a rank-deficient non-square `A \ b` and
    # `logabsdet` of a singular matrix.
    @testset "not differentiable" begin
        A = [1.0 2.0 1.0; 3.0 4.0 3.0; 5.0 6.0 5.0; 7.0 8.0 7.0]
        rng = StableRNG(123)
        TestUtils.test_rule_throws(rng, \, A, ones(4); err=ArgumentError)
        TestUtils.test_rule_throws(rng, \, collect(A'), ones(3); err=ArgumentError)
        TestUtils.test_rule_throws(rng, logabsdet, [1.0 2.0; 2.0 4.0]; err=ArgumentError)
    end

    # Finite differences are too inaccurate to test det in Float32, so compare with Float64.
    @testset "det, Float32" for A in [
        [2.0 1.0 0.5; 0.3 3.0 1.0; 0.2 0.1 4.0], [2.0 1.0; 0.0 3.0], [1.0 2.0; 2.0 4.0]
    ]
        g(A) = Mooncake.value_and_gradient!!(
            Mooncake.prepare_gradient_cache(det, A), det, A
        )
        @test g(Float32.(A))[2][2] ≈ g(A)[2][2] rtol = 1e-5
    end
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
