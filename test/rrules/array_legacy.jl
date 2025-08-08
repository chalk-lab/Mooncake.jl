@testset "array_legacy" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:array_legacy))
end
