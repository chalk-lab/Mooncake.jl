@testset "linear_algebra" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:linear_algebra))
end

if Base.get_extension(Mooncake, :MooncakeChainRulesExt) !== nothing
    rng = StableRNG(123)
    @testset "svd, $P, $m×$n" for P in [Float64, Float32], (m, n) in [(3, 3), (5, 3)]
        TestUtils.test_rule(rng, svd, randn(rng, P, m, n); mode=ReverseMode)
    end
end
