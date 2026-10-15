@testset "linear_algebra" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:linear_algebra))

    # Points where the derivative doesn't exist: a rank-deficient non-square `A \ b` and
    # `logabsdet` of a singular matrix.
    @testset "not differentiable" begin
        A = [1.0 2.0 1.0; 3.0 4.0 3.0; 5.0 6.0 5.0; 7.0 8.0 7.0]
        rng = StableRNG(123)
        TestUtils.test_rule_throws(rng, \, A, ones(4); err=ArgumentError)
        TestUtils.test_rule_throws(rng, \, collect(A'), ones(3); err=ArgumentError)
        f(A) = first(logabsdet(A))
        TestUtils.test_rule_throws(rng, f, [1.0 2.0; 2.0 4.0]; err=ArgumentError)
    end

    # Finite differences are too inaccurate to test det and non-square \ in Float32, so
    # compare with Float64.
    grad(f, x...) = Mooncake.value_and_gradient!!(
        Mooncake.prepare_gradient_cache(f, x...), f, x...
    )
    # A matrix with a NaN has a NaN determinant and gradient, as without the rule.
    @test all(isnan, grad(det, [1.0 NaN; 2.0 3.0])[2][2])
    @testset "det, Float32" for A in [
        [2.0 1.0 0.5; 0.3 3.0 1.0; 0.2 0.1 4.0], [2.0 1.0; 0.0 3.0], [1.0 2.0; 2.0 4.0]
    ]
        @test grad(det, Float32.(A))[2][2] ≈ grad(det, A)[2][2] rtol = 1e-5
    end
    @testset "\\, Float32, $(size(A))" for A in [
        [2.0 1.0 0.5; 0.3 3.0 1.0; 0.2 0.1 4.0; 1.0 0.0 2.0; 0.5 1.5 0.0],
        [2.0 1.0 0.5 1.0 0.5; 0.3 3.0 1.0 0.0 1.5; 0.2 0.1 4.0 2.0 0.0],
    ]
        f(A, b) = sum(abs2, A \ b)
        b = collect(1.0:size(A, 1))
        g32, g64 = grad(f, Float32.(A), Float32.(b))[2], grad(f, A, b)[2]
        @test g32[2] ≈ g64[2] rtol = 1e-4
        @test g32[3] ≈ g64[3] rtol = 1e-4
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
