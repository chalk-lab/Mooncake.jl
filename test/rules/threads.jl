@testset "threads" begin
    test_cases, memory = TestUtils.test_hook(
        Mooncake.hand_written_rule_test_cases, StableRNG, Val(:threads)
    ) do
        Mooncake.hand_written_rule_test_cases(StableRNG, Val(:threads))
    end
    GC.@preserve memory @testset "$f, $(Mooncake._typeof(x))" for (
            interface_only, perf_flag, _, f, x...
        ) in test_cases

        TestUtils.test_rule(
            StableRNG(123),
            f,
            x...;
            interface_only,
            perf_flag,
            is_primitive=false,
            mode=ReverseMode,
        )
    end

    derived_cases, derived_memory = TestUtils.test_hook(
        Mooncake.derived_rule_test_cases, StableRNG, Val(:threads), nothing
    ) do
        Mooncake.derived_rule_test_cases(StableRNG, Val(:threads))
    end
    GC.@preserve derived_memory @testset "$f, $(Mooncake._typeof(x))" for (
            interface_only, perf_flag, _, f, x...
        ) in derived_cases

        TestUtils.test_rule(
            StableRNG(123),
            f,
            x...;
            interface_only,
            perf_flag,
            is_primitive=false,
            mode=ReverseMode,
        )
    end

    @testset "pullback_type inference" begin
        f_scale = Mooncake.ThreadedScale(2.0)

        cases = [
            ("sin", (typeof(sin), Float64)),
            ("exp32", (typeof(exp), Float32)),
            ("plus", (typeof(+), Float64, Float64)),
            ("callable_struct", (typeof(f_scale), Float64)),
            ("Float64", (Type{Float64}, Float32)),
            ("lambda_plus", (typeof((x, y) -> x + y), Float32, Float64)),
        ]

        @testset "$name" for (name, sig) in cases
            rule = Mooncake.build_rrule(Tuple{sig...})
            @test Mooncake.pullback_type(Mooncake._typeof(rule), sig) !== Any
        end
    end

    @testset "primitive builder" begin
        rule = Mooncake.build_rrule(
            Tuple{
                typeof(Mooncake.threaded_map!),
                typeof(sin),
                Vector{Float64},
                Vector{Float64},
            },
        )
        @test rule isa Mooncake.ThreadedMapReverseRule
    end
end
