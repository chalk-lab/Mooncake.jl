@testset "avoiding_non_differentiable_code" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:avoiding_non_differentiable_code))
end
