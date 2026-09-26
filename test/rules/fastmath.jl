@testset "fastmath" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:fastmath))

    # Regression: integer-exponent pow_fast must use the guarded NDual scale so an inactive
    # (zero-partial) lane stays zero where the gradient diverges (x == 0, n < 0).
    @testset "pow_fast inactive lane at x = 0, n < 0 (width $N)" for N in (2, 3)
        parts = ntuple(k -> Float64(k == 1), N)
        x0 = Lifted{Float64,N}(0.0, Mooncake.Nfwd.NDual{Float64,N}(0.0, parts))
        r = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), Base.FastMath.pow_fast),
            x0,
            Mooncake.zero_lifted(Val(N), -2),
        )
        @test tangent(r).value == Inf
        @test all(iszero, tangent(r).partials[2:end])
    end
end
