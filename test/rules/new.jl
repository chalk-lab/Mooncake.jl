@testset "new" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:new))
    include("tangent_world_age_regression.jl")
end
