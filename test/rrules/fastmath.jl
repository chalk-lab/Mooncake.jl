@testset "fastmath" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:fastmath))
end
