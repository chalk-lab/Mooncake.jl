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

    @testset "trig pole guard (inactive lane stays 0, not NaN)" begin
        # Direct NDual calls cover the scalar arithmetic path, which tan/tand frules bypass.
        # Mixed active/inactive lanes also pin the singular derivative in the active lane.
        for T in (Float16, Float32, Float64)
            for (f, xs) in ((tand, (90, 270)), (tanpi, (0.5, 1.5)), (secd, (90, 270)))
                for x in T.(xs)
                    @test tangent(
                        Mooncake.frule!!(zero_dual(f), Mooncake.lift(x, zero(T))), 1
                    ) === zero(T)
                    d = f(Mooncake.Nfwd.NDual{T,2}(x, (zero(T), one(T))))
                    @test d.partials[1] === zero(T)
                    @test isinf(d.partials[2])
                end
            end
        end
        for f in (tan, sec)
            d = f(Mooncake.Nfwd.NDual{Float16,2}(Float16(π / 2), (Float16(0), Float16(1))))
            @test isfinite(d.value)
            @test d.partials[1] === Float16(0)
            @test isinf(d.partials[2])
        end
    end

    @testset "nfwd-backed non-smooth scalar rules" begin
        for T in (Float16, Float32, Float64)
            @test tangent(
                Mooncake.frule!!(
                    zero_dual(^),
                    Mooncake.lift(zero(T), one(T)),
                    Mooncake.lift(one(T), zero(T)),
                ),
                1,
            ) === one(T)
            @test tangent(
                Mooncake.frule!!(
                    zero_dual(^),
                    Mooncake.lift(zero(T), one(T)),
                    Mooncake.lift(T(2), zero(T)),
                ),
                1,
            ) === zero(T)
            @test isinf(
                tangent(
                    Mooncake.frule!!(
                        zero_dual(^),
                        Mooncake.lift(zero(T), one(T)),
                        Mooncake.lift(T(0.5), zero(T)),
                    ),
                    1,
                ),
            )

            @test isnan(
                tangent(
                    Mooncake.frule!!(
                        zero_dual(mod),
                        Mooncake.lift(T(4), one(T)),
                        Mooncake.lift(T(2), zero(T)),
                    ),
                    1,
                ),
            )
            @test isnan(
                tangent(
                    Mooncake.frule!!(zero_dual(mod2pi), Mooncake.lift(T(2π), one(T))), 1
                ),
            )
            # Reverse mod2pi must also be NaN at wrap points, including zero.
            let (_, pb) = Mooncake.rrule!!(zero_codual(mod2pi), zero_codual(zero(T)))
                @test isnan(pb(one(T))[2])
            end
            let (_, pb) = Mooncake.rrule!!(zero_codual(mod2pi), zero_codual(T(0.7)))
                @test pb(one(T))[2] === one(T)
            end

            @test tangent(
                Mooncake.frule!!(
                    zero_dual(max),
                    Mooncake.lift(one(T), one(T)),
                    Mooncake.lift(one(T), zero(T)),
                ),
                1,
            ) === zero(T)
            @test tangent(
                Mooncake.frule!!(
                    zero_dual(min),
                    Mooncake.lift(one(T), one(T)),
                    Mooncake.lift(one(T), zero(T)),
                ),
                1,
            ) === one(T)

            @test tangent(
                Mooncake.frule!!(zero_dual(Base.eps), Mooncake.lift(one(T), one(T))), 1
            ) === zero(T)
            @test tangent(
                Mooncake.frule!!(zero_dual(nextfloat), Mooncake.lift(one(T), one(T))), 1
            ) === one(T)
            @test tangent(
                Mooncake.frule!!(zero_dual(prevfloat), Mooncake.lift(one(T), one(T))), 1
            ) === one(T)
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

    @testset "near-boundary domain-restricted functions" begin
        test_rule(StableRNG(123), sqrt, 0.005; is_primitive=true, max_fd_step=1e-3)
    end

    # Saturated tanh is flat in floating point, so FD accepts a spurious zero gradient.
    # Compare analytically to pin the nonzero sech² derivative.
    @testset "tanh gradient survives saturation" begin
        for x in (15.0, 19.0, 20.0, 25.0, 8.0f0, 9.0f0, 10.0f0)
            P = typeof(x)
            u = exp(-2 * abs(x))
            want = 4u / (one(P) + u)^2
            cache = Mooncake.prepare_gradient_cache(tanh, x)
            got = Mooncake.value_and_gradient!!(cache, tanh, x)[2][2]
            @test got == want
            @test !iszero(got)
        end
    end
end
