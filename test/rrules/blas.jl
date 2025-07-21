@testset "blas" begin
    @test_throws "Encountered unexpected array type" Mooncake.arrayify(5, 4)
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas))
end
