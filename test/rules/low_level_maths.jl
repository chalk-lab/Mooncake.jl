@testset "low_level_maths" begin
    # Test helper: lift each bare `Dual` arg to its canonical `Lifted` slot,
    # invoke the `Lifted`-typed `frule!!` body, and unwrap the result back to
    # a width-1 `Dual` so existing `tangent` checks still apply. Mirrors the
    # runtime path used by `__forwards` / `_wrap_oc_args` — these tests
    # exercise the public lifted-API rules directly.
    @inline _to_lifted(d::Mooncake.Dual{P}) where {P} = Mooncake.Lifted{P,1}(
        primal(d), tangent(d)
    )
    @inline function _frule_lifted(f, args::Vararg{Mooncake.Dual,N}) where {N}
        fl = Mooncake.zero_lifted(Val(1), f)
        ls = ntuple(i -> _to_lifted(args[i]), Val(N))
        return Mooncake._ndual_output_to_width1(Mooncake.frule!!(fl, ls...))
    end

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

            @test tangent(_frule_lifted(hypot, x)) === zero(T)
            @test tangent(_frule_lifted(hypot, x, y)) === zero(T)
            @test tangent(_frule_lifted(hypot, x, y, z)) === zero(T)

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

    @testset "nfwd-backed non-smooth scalar rules" begin
        for T in (Float16, Float32, Float64)
            @test tangent(
                _frule_lifted(^, Dual(zero(T), one(T)), Dual(one(T), zero(T)))
            ) === one(T)
            @test tangent(_frule_lifted(^, Dual(zero(T), one(T)), Dual(T(2), zero(T)))) ===
                zero(T)
            @test isinf(
                tangent(_frule_lifted(^, Dual(zero(T), one(T)), Dual(T(0.5), zero(T))))
            )

            @test isnan(
                tangent(_frule_lifted(mod, Dual(T(4), one(T)), Dual(T(2), zero(T))))
            )
            @test isnan(tangent(_frule_lifted(mod2pi, Dual(T(2π), one(T)))))

            @test tangent(
                _frule_lifted(max, Dual(one(T), one(T)), Dual(one(T), zero(T)))
            ) === zero(T)
            @test tangent(
                _frule_lifted(min, Dual(one(T), one(T)), Dual(one(T), zero(T)))
            ) === one(T)

            @test tangent(_frule_lifted(Base.eps, Dual(one(T), one(T)))) === zero(T)
            @test tangent(_frule_lifted(nextfloat, Dual(one(T), one(T)))) === one(T)
            @test tangent(_frule_lifted(prevfloat, Dual(one(T), one(T)))) === one(T)
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
