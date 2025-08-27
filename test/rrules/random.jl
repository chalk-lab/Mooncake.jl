@testset "randn" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:random))
end
