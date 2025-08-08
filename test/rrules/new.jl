@testset "new" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:new))
    include("build_fdata_world_age_regression.jl")
end
