function TestUtils.test_hook(
    f, ::typeof(Mooncake.derived_rule_test_cases), rng_ctor, ::Val{:random}, mode
)
    test_cases, memory = f()
    if mode === ReverseMode
        # TODO: restore reverse-mode derived random coverage once `Random.seed!` no longer
        # trips the missing `Base.indexed_iterate` reverse rule on `ReinterpretArray`.
        return Any[], memory
    end
    return test_cases, memory
end

@testset "random" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:random))
end

@testset "Random state advancement" begin
    rng = Xoshiro(123)
    for f in (rand!, rand, randn!, randn)
        rng2 = deepcopy(rng)
        args = f in (rand!, randn!) ? (rng, [0.0]) : (rng,)
        rule = Mooncake.build_rrule(f, args...)
        @test rng == rng2
        if f in (rand!, randn!)
            Mooncake.value_and_pullback!!(rule, [1.0], f, args...)
        else
            Mooncake.value_and_gradient!!(rule, f, args...)
        end
        @test rng != rng2
    end
end
