@testset "low_level_maths" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:low_level_maths))
    @testset "NaN handling in rrules" begin
        test_cases = vcat(
            map([Float16, Float32, Float64]) do T
                cases = [
                    (log, T(0)),
                    (log, (T(0), T(0))),
                    (sqrt, T(0)),
                    (cbrt, T(0)),
                    (log10, T(0)),
                    (log2, T(0)),
                    (log1p, T(-1)),
                    (hypot, T(0)),
                    (hypot, (T(0), T(0))),
                    (hypot, (T(0), T(0), T(0))),
                    # builtins
                    (Base.sqrt_llvm, T(0)),
                    (Base.sqrt_llvm_fast, T(0)),
                ]
                return cases
            end...,
        )
        for (f, args) in test_cases
            cache = prepare_gradient_cache(f, args...)
            _, grad = value_and_gradient!!(cache, f, args...)
            @test all(iszero, grad[2:end])
        end
    end

    # These are all examples of signatures which we do _not_ want to make primitives,
    # because they are very shallow wrappers around lower-level primitives for which we
    # already have rules.
    @testset "$T, $C, $M" for T in [Float16, Float32, Float64],
        C in [DefaultCtx, MinimalCtx],
        M in [ForwardMode, ReverseMode]

        @test !is_primitive(C, M, Tuple{typeof(+),T})
        @test !is_primitive(C, M, Tuple{typeof(-),T})
        @test !is_primitive(C, M, Tuple{typeof(abs2),T})
        @test !is_primitive(C, M, Tuple{typeof(inv),T})
        @test !is_primitive(C, M, Tuple{typeof(abs),T})

        @test !is_primitive(C, M, Tuple{typeof(+),T,T})
        @test !is_primitive(C, M, Tuple{typeof(-),T,T})
        @test !is_primitive(C, M, Tuple{typeof(*),T,T})
        @test !is_primitive(C, M, Tuple{typeof(/),T,T})
        @test !is_primitive(C, M, Tuple{typeof(\),T,T})
    end
end
