@testset "new" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:new))
end
