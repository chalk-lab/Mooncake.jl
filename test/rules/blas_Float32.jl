@testset "blas (Float32)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float32))
end
