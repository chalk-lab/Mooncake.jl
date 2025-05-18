@testset "low_level_maths" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:low_level_maths))

    # These are all examples of signatures which we do _not_ want to make primitives,
    # because they are very shallow wrappers around lower-level primitives for which we
    # already have rules.
    @testset "$T, $C" for T in [Float16, Float32, Float64], C in [DefaultCtx, MinimalCtx]
        @test !is_primitive(C, Tuple{typeof(+),T})
        @test !is_primitive(C, Tuple{typeof(-),T})
        @test !is_primitive(C, Tuple{typeof(abs2),T})
        @test !is_primitive(C, Tuple{typeof(inv),T})
        @test !is_primitive(C, Tuple{typeof(abs),T})

        @test !is_primitive(C, Tuple{typeof(+),T,T})
        @test !is_primitive(C, Tuple{typeof(-),T,T})
        @test !is_primitive(C, Tuple{typeof(*),T,T})
        @test !is_primitive(C, Tuple{typeof(/),T,T})
        @test !is_primitive(C, Tuple{typeof(\),T,T})
    end
end
