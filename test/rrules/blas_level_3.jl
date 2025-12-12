@testset "blas (level 3)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3))
end
