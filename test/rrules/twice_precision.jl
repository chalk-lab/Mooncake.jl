@testset "twice_precision" begin
    rng = sr(123)
    p = Base.TwicePrecision{Float64}(5.0, 4.0)
    TestUtils.test_tangent_consistency(rng, p)
    TestUtils.test_fwds_rvs_data(rng, p)
    TestUtils.run_rule_test_cases(StableRNG, Val(:twice_precision))
end
