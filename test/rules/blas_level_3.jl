# Split into sequential calls so GC can reclaim each set before the next is built,
# reducing peak memory. See src/rules/blas.jl for details.
@testset "blas (level 3)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3a))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3b))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3c))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3d))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3e))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3f))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3))
end
