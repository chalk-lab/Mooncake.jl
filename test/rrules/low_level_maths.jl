@testset "low_level_maths" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:low_level_maths))

    # These are all examples of signatures which we do _not_ want to make primitives,
    # because they are very shallow wrappers around lower-level primitives for which we
    # already have rules.
    world = Base.get_world_counter()
    @testset "$T, $C, $M" for T in [Float16, Float32, Float64],
        C in [DefaultCtx, MinimalCtx],
        M in [ForwardMode, ReverseMode]

        @test !is_primitive(C, M, Tuple{typeof(+),T}, world)
        @test !is_primitive(C, M, Tuple{typeof(-),T}, world)
        @test !is_primitive(C, M, Tuple{typeof(abs2),T}, world)
        @test !is_primitive(C, M, Tuple{typeof(inv),T}, world)
        @test !is_primitive(C, M, Tuple{typeof(abs),T}, world)

        @test !is_primitive(C, M, Tuple{typeof(+),T,T}, world)
        @test !is_primitive(C, M, Tuple{typeof(-),T,T}, world)
        @test !is_primitive(C, M, Tuple{typeof(*),T,T}, world)
        @test !is_primitive(C, M, Tuple{typeof(/),T,T}, world)
        @test !is_primitive(C, M, Tuple{typeof(\),T,T}, world)
    end
end
