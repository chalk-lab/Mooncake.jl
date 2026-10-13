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

# test_rule's reverse correctness check requires restoration of every input, including
# the RNG. Random fills deliberately advance it, so check their state contract here,
# using the registered interface/performance cases above.
@testset "Random fill aliases, restoration, and subsequent draw" begin
    cases, _ = Mooncake.hand_written_rule_test_cases(StableRNG, Val(:random))
    for (_, _, _, f, args...) in cases
        f in (rand!, randn!, randexp!) || continue
        rng, x, sampler... = args
        if !isconcretetype(eltype(x))
            for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
                @test !Mooncake.is_primitive(
                    Mooncake.DefaultCtx,
                    mode,
                    Mooncake._typeof((f, args...)),
                    Base.get_world_counter(),
                )
            end
            continue
        end
        replayed_next = rand(Random.seed!(deepcopy(rng), 123))
        expected_rng = Random.seed!(deepcopy(rng), 123)
        expected_x = f(expected_rng, copy(x), sampler...)
        expected_next = rand(expected_rng)
        dx = similar(x)
        @testset "$mode $(typeof((f, args...)))" for mode in (
            Mooncake.ForwardMode, Mooncake.ReverseMode
        )
            r = Random.seed!(deepcopy(rng), 123)
            fill!(x, 2)
            fill!(dx, 3)
            if mode === Mooncake.ForwardMode
                rule = Mooncake.build_frule(f, r, x, sampler...)
                out = rule(
                    Mooncake.zero_dual(f),
                    Mooncake.zero_dual(r),
                    Mooncake.Dual(x, dx),
                    map(Mooncake.zero_dual, sampler)...,
                )
            else
                rule = Mooncake.build_rrule(f, r, x, sampler...)
                out, pb = rule(
                    Mooncake.zero_fcodual(f),
                    Mooncake.zero_fcodual(r),
                    Mooncake.CoDual(x, dx),
                    map(Mooncake.zero_fcodual, sampler)...,
                )
            end
            @test Mooncake.primal(out) === x
            @test Mooncake.tangent(out) === dx
            @test x == expected_x
            @test all(iszero, dx)
            if mode === Mooncake.ReverseMode
                fill!(dx, 7)
                pb(Mooncake.NoRData())
                @test all(==(2), x)
                @test all(==(3), dx)
            end
            next = rand(r)
            @test next == expected_next
            @test next != replayed_next
        end
    end
end
