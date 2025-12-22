@testset "lapack" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:lapack))
end
