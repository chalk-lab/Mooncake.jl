@testset "performance_patches" begin
    @test !Mooncake.may_match_primitive(
        MinimalCtx, ForwardMode, Tuple{typeof(mul!),Any,Any,Any,Int,Int}
    )
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))
end
