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
                ]
                return cases
            end...,
        )

        # Test cases for avoiding `NaN` poisoning. 
        #  See https://github.com/chalk-lab/Mooncake.jl/issues/807
        function low_level_maths_nantester(f, args)
            a = f(args...)
            b = args
            return sum(b)
        end

        for (f, args) in test_cases
            cache = prepare_gradient_cache(low_level_maths_nantester, f, args)
            _, grad = value_and_gradient!!(cache, low_level_maths_nantester, f, args)
            @test all(map(isone, grad[3:end]...))
        end
    end

    @testset "hypot singular-point consistency across arities" begin
        for T in (Float16, Float32, Float64)
            x = Dual(zero(T), one(T))
            y = Dual(zero(T), one(T))
            z = Dual(zero(T), one(T))

            @test tangent(Mooncake.frule!!(zero_dual(hypot), x)) === zero(T)
            @test tangent(Mooncake.frule!!(zero_dual(hypot), x, y)) === zero(T)
            @test tangent(Mooncake.frule!!(zero_dual(hypot), x, y, z)) === zero(T)

            _, pb1 = Mooncake.rrule!!(zero_fcodual(hypot), zero_fcodual(zero(T)))
            _, dx1 = pb1(one(T))
            @test dx1 === zero(T)

            _, pb2 = Mooncake.rrule!!(
                zero_fcodual(hypot), zero_fcodual(zero(T)), zero_fcodual(zero(T))
            )
            _, dx2, dy2 = pb2(one(T))
            @test dx2 === zero(T)
            @test dy2 === zero(T)

            _, pb3 = Mooncake.rrule!!(
                zero_fcodual(hypot),
                zero_fcodual(zero(T)),
                zero_fcodual(zero(T)),
                zero_fcodual(zero(T)),
            )
            _, dx3, dy3, dz3 = pb3(one(T))
            @test dx3 === zero(T)
            @test dy3 === zero(T)
            @test dz3 === zero(T)
        end
    end

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
