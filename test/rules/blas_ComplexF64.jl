@testset "blas (ComplexF64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_ComplexF64))
end
