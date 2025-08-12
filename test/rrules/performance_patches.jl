@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))
end
