# Split by element type so GC can reclaim each precision's arrays before the next is
# built, reducing peak memory. See src/rules/blas.jl for details.
@testset "blas (level 3)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3_Float64))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3_Float32))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3_ComplexF64))
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_level_3_ComplexF32))
end
