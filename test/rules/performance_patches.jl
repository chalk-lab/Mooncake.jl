@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))
    @testset "Float16 mul! derivatives and restoration" for b in (Float16(0), Float16(0.25))
        P = Float16
        C, A, B = fill(P(4), 1, 1), fill(P(2), 1, 1), fill(P(3), 1, 1)
        a = P(0.5)
        result = Mooncake.frule!!(
            zero_dual(mul!),
            Dual(copy(C), ones(P, 1, 1)),
            Dual(A, ones(P, 1, 1)),
            Dual(B, ones(P, 1, 1)),
            Dual(a, one(P)),
            Dual(b, one(P)),
        )
        @test primal(result) == fill(P(3) + P(4) * b, 1, 1)
        @test tangent(result) == fill(P(12.5) + b, 1, 1)

        args = map(zero_fcodual, (mul!, C, A, B, a, b))
        result, pb = rrule!!(args...)
        @test primal(result) == fill(P(3) + P(4) * b, 1, 1)
        fill!(tangent(result), P(2))
        @test pb(NoRData()) == (NoRData(), NoRData(), NoRData(), NoRData(), P(12), P(8))
        @test C == fill(P(4), 1, 1)
        @test tangent(args[2]) == fill(P(2) * b, 1, 1)
        @test tangent(args[3]) == fill(P(3), 1, 1)
        @test tangent(args[4]) == fill(P(2), 1, 1)
    end
end
