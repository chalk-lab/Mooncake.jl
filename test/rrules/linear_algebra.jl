@testset "linear_algebra" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:linear_algebra))
end
