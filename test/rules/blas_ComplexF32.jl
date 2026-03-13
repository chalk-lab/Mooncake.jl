@testset "blas (ComplexF32)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_ComplexF32))
end
