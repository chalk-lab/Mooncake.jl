@testset "threads" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:threads))
end
