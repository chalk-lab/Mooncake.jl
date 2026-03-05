# Split into sequential sets so GC can reclaim each before the next is built,
# reducing peak memory. See src/rules/blas.jl for details.
@testset "blas (level 3a)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3a))
end
@testset "blas (level 3b)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3b))
end
@testset "blas (level 3c)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3c))
end
@testset "blas (level 3d)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3d))
end
@testset "blas (level 3, derived)" begin
    TestUtils.run_derived_rule_test_cases(StableRNG, Val(:blas_level_3), ForwardMode)
    TestUtils.run_derived_rule_test_cases(StableRNG, Val(:blas_level_3), ReverseMode)
end
