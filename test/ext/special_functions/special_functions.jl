include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "SpecialFunctions")

using AllocCheck, JET, Mooncake, SpecialFunctions, StableRNGs, Test
using Mooncake.Nfwd: NDual
using Mooncake: ForwardMode, ReverseMode, map_prod
using Mooncake.TestUtils: test_rule

# On Julia 1.10, a subset of upstream SpecialFunctions scalar primals allocate in ways
# that are independent of Mooncake's imported rule path. Keep the type-stability checks,
# but skip the zero-allocation assertion for those known cases only.
function _sf_perf_flag(::Type{P}, name::Symbol, default::Symbol) where {P}
    VERSION < v"1.11" || return default
    name in (:digamma, :erfinv, :invdigamma, :trigamma, :expintx) && return :stability
    P === Float32 || return default
    name in (:logerfc, :logerfcx, :beta, :logbeta, :logabsgamma, :loggamma) &&
        return :stability
    return default
end

function _sf_nonprimitive_perf_flag(::Type{P}, name::Symbol, default::Symbol) where {P}
    VERSION < v"1.11" || return default
    P === Float32 && name === :logabsbeta && return :none
    return default
end

function _sf_nonprimitive_perf_flag(name::Symbol, default::Symbol)
    VERSION < v"1.11" || return default
    name in (:gammax, :rgammax) && return :none
    return default
end

# Rules in this file are only lightly tested, because they are all just @from_rrule rules.
@testset "special_functions" begin
    @testset "$perf_flag, $(typeof((f, x...)))" for (perf_flag, f, x...) in vcat(
        map([Float64, Float32]) do P
            return Any[
                (:stability, airyai, P(0.1)),
                (:stability, airyaix, P(0.1)),
                (:stability, airyaiprime, P(0.1)),
                (:stability, airyaiprimex, P(0.1)),
                (:stability, airybi, P(0.1)),
                (:stability, airybiprime, P(0.1)),
                (:stability_and_allocs, besselj0, P(0.1)),
                (:stability_and_allocs, besselj1, P(0.1)),
                (:stability_and_allocs, bessely0, P(0.1)),
                (:stability_and_allocs, bessely1, P(0.1)),
                (:stability_and_allocs, dawson, P(0.1)),
                (_sf_perf_flag(P, :digamma, :stability_and_allocs), digamma, P(0.1)),
                (:stability_and_allocs, erf, P(0.1)),
                (:stability_and_allocs, erf, P(0.1), P(0.5)),
                (:stability_and_allocs, erfc, P(0.1)),
                (_sf_perf_flag(P, :logerfc, :stability_and_allocs), logerfc, P(0.1)),
                (:stability_and_allocs, erfcinv, P(0.1)),
                (:stability_and_allocs, erfcx, P(0.1)),
                (_sf_perf_flag(P, :logerfcx, :stability_and_allocs), logerfcx, P(0.1)),
                (:stability_and_allocs, erfi, P(0.1)),
                (_sf_perf_flag(P, :erfinv, :stability_and_allocs), erfinv, P(0.1)),
                (:stability_and_allocs, gamma, P(0.1)),
                (_sf_perf_flag(P, :invdigamma, :stability_and_allocs), invdigamma, P(0.1)),
                (_sf_perf_flag(P, :trigamma, :stability_and_allocs), trigamma, P(0.1)),
                (:stability_and_allocs, polygamma, 3, P(0.1)),
                (_sf_perf_flag(P, :beta, :stability_and_allocs), beta, P(0.3), P(0.1)),
                (
                    _sf_perf_flag(P, :logbeta, :stability_and_allocs),
                    logbeta,
                    P(0.3),
                    P(0.1),
                ),
                (
                    _sf_perf_flag(P, :logabsgamma, :stability_and_allocs),
                    logabsgamma,
                    P(0.3),
                ),
                (_sf_perf_flag(P, :loggamma, :stability_and_allocs), loggamma, P(0.3)),
                (:stability_and_allocs, expint, P(0.3)),
                (_sf_perf_flag(P, :expintx, :stability_and_allocs), expintx, P(0.3)),
                (:stability_and_allocs, expinti, P(0.3)),
                (:stability_and_allocs, sinint, P(0.3)),
                (:stability_and_allocs, cosint, P(0.3)),
                (:stability_and_allocs, ellipk, P(0.3)),
                (:stability_and_allocs, ellipe, P(0.3)),
            ]
        end...,
        (:stability_and_allocs, logfactorial, 3),
    )
        test_rule(StableRNG(123456), f, x...; perf_flag)
    end

    @testset "$perf_flag, $(typeof((f, x...)))" for (perf_flag, f, x...) in vcat(
        map([Float64, Float32]) do P
            return Any[
                (:none, logerf, P(0.3), P(0.5)), # first branch
                (:none, logerf, P(1.1), P(1.2)), # second branch
                (:none, logerf, P(-1.2), P(-1.1)), # third branch
                (:none, logerf, P(0.3), P(1.1)), # fourth branch
                (:allocs, SpecialFunctions.loggammadiv, P(1.0), P(9.0)),
                (
                    _sf_nonprimitive_perf_flag(P, :logabsbeta, :allocs),
                    logabsbeta,
                    P(0.3),
                    P(0.1),
                ),
            ]
        end...,

        # Functions which only support Float64.
        (_sf_nonprimitive_perf_flag(:gammax, :allocs), SpecialFunctions.gammax, 1.0),
        (_sf_nonprimitive_perf_flag(:rgammax, :allocs), SpecialFunctions.rgammax, 3.0, 6.0),
        (:allocs, SpecialFunctions.rgamma1pm1, 0.1),
        (:allocs, SpecialFunctions.auxgam, 0.1),
        (:allocs, SpecialFunctions.loggamma1p, 0.3),
        (:allocs, SpecialFunctions.loggamma1p, -0.3),
        (:none, SpecialFunctions.lambdaeta, 5.0),
    )
        test_rule(StableRNG(123456), f, x...; perf_flag, is_primitive=false)
    end

    @testset "beta_inc" begin
        for T in (Float32, Float64),
            (a, b, x) in (
                (1, 2, 0.5),
                (2, 2, 0.5),
                (2, 4, 0.4),
                (0.5, 0.5, 0.1),
                (1.3, 2.1, 0.8),
                (100, 100, 0.5),
            )

            test_rule(
                StableRNG(123), beta_inc, T(a), T(b), T(x); perf_flag=:stability_and_allocs
            )
        end
        test_rule(
            StableRNG(123), beta_inc, 1.0f0, 2.0, 0.5f0; perf_flag=:stability_and_allocs
        )
        for x in (0.25, 0.75)
            test_rule(
                StableRNG(123),
                beta_inc,
                2.0,
                2.0,
                x,
                1-x;
                interface_only=true,
                perf_flag=:stability_and_allocs,
            )
            # Finite differences must preserve y = 1-x.
            test_rule(
                StableRNG(123),
                (a, b, x) -> beta_inc(a, b, x, 1-x),
                2.0,
                2.0,
                x;
                is_primitive=false,
                perf_flag=:allocs,
            )
        end

        for T in (Float16, Float32, Float64),
            (a, b, x, A, B, D) in (
                (1, 2, 0.5, 1/4-3log(2)/4, log(2)/4, 1),
                (2, 1, 0.5, -log(2)/4, 3log(2)/4-1/4, 1),
                (2, 2, 0.5, 1/8-log(2)/2, log(2)/2-1/8, 1.5),
                (0.1, 0.1, 0.5, -2.532522082864052, 2.532522082864052, 0.17663027797787373),
                (
                    1000,
                    1000,
                    0.5,
                    -0.008922479336000142,
                    0.008922479336000142,
                    35.67802229170864,
                ),
                (1, 2, 0, 0, 0, 2),
                (2, 1, 1, 0, 0, 2),
                (2, 2, 0, 0, 0, 0),
                (2, 2, 1, 0, 0, 0),
                (1000, 1000, 0, 0, 0, 0),
                (1000, 1000, 1, 0, 0, 0),
                (0.5, 2, 0, 0, 0, Inf),
                (2, 0.5, 1, 0, 0, Inf),
            )

            a, b, x = T.((a, b, x))
            y, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, (beta_inc, a, b, x))...)
            @test Mooncake.primal(y) === beta_inc(a, b, x)
            tol = max(16eps(T), 1e-11)
            for (dp, dq) in ((one(T), zero(T)), (zero(T), one(T)), (one(T), one(T)))
                d = dp-dq
                g = pb((dp, dq))[2:end]
                expected = iszero(d) ? (zero(T), zero(T), zero(T)) : T.(d .* (A, B, D))
                @test all(isapprox.(g, expected; rtol=tol))
            end
            for i in 1:3
                seeds = ntuple(k -> T(k==i), 3)
                args = map(Mooncake.lift, (a, b, x), seeds)
                out = Mooncake.frule!!(Mooncake.zero_dual(beta_inc), args...)
                @test only(Mooncake.tangent(out)[1].partials) ≈ T((A, B, D)[i]) rtol=tol
                @test only(Mooncake.tangent(out)[2].partials) ≈ -T((A, B, D)[i]) rtol=tol
            end
            args = ntuple(i -> NDual{T,3}((a, b, x)[i], ntuple(k -> T(k==i), 3)), 3)
            p, q = beta_inc(args...)
            @test all(isapprox.(p.partials, T.((A, B, D)); rtol=tol))
            @test q.partials == map(-, p.partials)
        end

        for args in ((1.0, 2.0, 0.5), (2.0, 3.0, 0.75)),
            f in (
                (a, b, x) -> beta_inc(a, 2.0, 0.5),
                (a, b, x) -> beta_inc(1.0, b, 0.5),
                (a, b, x) -> beta_inc(a, b, 0.5),
                (a, b, x) -> beta_inc(1.0, 2.0, x),
                (a, b, x) -> beta_inc(a, 2.0, x),
                (a, b, x) -> beta_inc(1.0, b, x),
                beta_inc,
            )

            test_rule(StableRNG(123), f, args...; is_primitive=false, mode=ReverseMode)
        end

        y = NDual{Float64,1}(1e-20, (1.0,))
        p, q = beta_inc(2.0, 1.0, 1-y, y)
        @test q.value > 0
        @test q.partials[1] ≈ 2.0
        for (a, b) in ((0.0, 1.0), (1.0, 0.0), (Inf, 1.0))
            @test_throws DomainError Mooncake.rrule!!(
                map(Mooncake.zero_fcodual, (beta_inc, a, b, 0.5))...
            )
        end
        f(a) = first(beta_inc(a, 2.0, 0.5))
        cache = Mooncake.prepare_hvp_cache(f, 1.0)
        @test Mooncake.value_and_hvp!!(cache, f, 1.0, 1.0)[3] ≈ 3log(2)^2/4-log(2)/2
    end

    @testset "gamma_inc" begin
        for T in (Float32, Float64),
            (a, x) in ((0.1, 0.1), (3, 2), (1, 2), (3, 50), (1000, 1000))

            test_rule(
                StableRNG(123), gamma_inc, T(a), T(x), 0; perf_flag=:stability_and_allocs
            )
        end
        test_rule(StableRNG(123), gamma_inc, 3.0, 2.0; is_primitive=false)
        for (T, S) in ((Float32, Float64), (Float64, Float32))
            test_rule(
                StableRNG(123), gamma_inc, T(3), S(2), 0; perf_flag=:stability_and_allocs
            )
        end
        for T in (Float16, Float32, Float64),
            (a, x, da, dx) in (
                (3, 2, -0.2318486720439896, 0.2706705664732254),
                (1, 2, -0.2208254262118595, 0.1353352832366127),
                (1000, 2000, -4.756465407623239e-136, 3.430501706332743e-136),
                (2, 1e-200, 0, 1e-200),
                (3, 50, -7.552056917446367e-19, 2.410937309954897e-19),
                (1000, 1000, -0.012616713994069625, 0.012614611348721499),
                # `gamma_inc` underflows p to 0 here; the partials come from log space.
                (170, 1, -2.620172596340683e-307, 8.617324446353426e-306),
                (0, 2, -0.04890051070806112, 0),
                (1, 0, 0, 1),
                (2, 0, 0, 0),
                (0.5, 0, 0, Inf),
                (3, Inf, 0, 0),
            ),
            ind in (0, 1, 2)

            a, x = T(a), T(x)
            y, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, (gamma_inc, a, x, ind))...)
            @test Mooncake.primal(y) == gamma_inc(a, x, ind)
            for (dp, dq) in ((one(T), zero(T)), (zero(T), one(T)), (one(T), one(T)))
                d = dp - dq
                expected_a = iszero(d) ? zero(T) : T(d * da)
                expected_x = iszero(d) ? zero(T) : T(d * dx)
                grad = pb((dp, dq))
                @test grad[2] ≈ expected_a rtol=max(4eps(T), 1e-12)
                @test grad[3] ≈ expected_x rtol=max(4eps(T), 1e-12)
            end
            for (adot, xdot) in ((one(T), zero(T)), (zero(T), one(T)), (zero(T), zero(T)))
                result = Mooncake.frule!!(
                    Mooncake.zero_dual(gamma_inc),
                    Mooncake.lift(a, adot),
                    Mooncake.lift(x, xdot),
                    Mooncake.zero_dual(ind),
                )
                expected =
                    (iszero(adot) ? zero(T) : T(da)) + (iszero(xdot) ? zero(T) : T(dx))
                dp, dq = Mooncake.tangent(result)
                @test Mooncake.primal(result) == gamma_inc(a, x, ind)
                @test only(dp.partials) ≈ expected rtol=max(4eps(T), 1e-12)
                @test only(dq.partials) ≈ -expected rtol=max(4eps(T), 1e-12)
            end
        end

        for T in (Float16, Float32, Float64)
            a, x = nextfloat(zero(T)), T(2)
            _, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, (gamma_inc, a, x, 0))...)
            @test pb((one(T), zero(T)))[2] ≈ -expint(x) rtol=max(4eps(T), 1e-12)
        end

        for (a, x, seed, expected) in (
            (Float16(0.05), nextfloat(Float16(0)), Float16(0.01), Float16(3752)),
            (Float16(1), Float16(2), floatmax(Float16), Float16(8864)),
            (0.05f0, nextfloat(0.0f0), 1.0f-5, 2.0961772f36),
            (1e-10, 1e-310, 1.0, 9.9999992867763e299),
            (3.0, 1e160, 1.0, 0.0),
            (3.0, 1e308, 1.0, 0.0),
        )
            _, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, (gamma_inc, a, x, 0))...)
            @test pb((seed, zero(seed)))[3] ≈ expected rtol=max(4eps(typeof(a)), 1e-12)
            @test pb((seed, -seed))[3] ≈ 2expected rtol=max(4eps(typeof(a)), 1e-12)
            result = Mooncake.frule!!(
                Mooncake.zero_dual(gamma_inc),
                Mooncake.lift(a, zero(a)),
                Mooncake.lift(x, seed),
                Mooncake.zero_dual(0),
            )
            dp = Mooncake.tangent(result)[1]
            @test only(dp.partials) ≈ expected rtol=max(4eps(typeof(a)), 1e-12)
        end

        @test_throws DomainError Mooncake.rrule!!(
            map(Mooncake.zero_fcodual, (gamma_inc, Inf, 1.0, 0))...
        )
        @test_throws ErrorException Mooncake.rrule!!(
            map(Mooncake.zero_fcodual, (gamma_inc, 1e12, 1e12, 0))...
        )
    end

    @testset "Primitive SpecialFunctions with `NotImplemented` gradients" begin
        first_arg_types = [Float64, Float32]
        second_arg_types = [Float64, Float32]

        # Check gradients while excluding those marked as `NotImplemented`.
        @testset "$perf_flag, $(typeof((f, x...)))" for (perf_flag, f, x...) in vcat(
            map_prod(first_arg_types, second_arg_types) do (T, P)
                return Any[
                    # 2-arg standard Bessel/Hankel (1st arg gradient is `NotImplemented`)
                    (:none, x -> besselj(T(3), x), P(1.5)),
                    (:none, x -> besseli(T(3), x), P(1.5)),
                    (:none, x -> bessely(T(3), x), P(1.5)),
                    (:none, x -> besselk(T(3), x), P(1.5)),
                    (:none, x -> hankelh1(T(3), x), P(1.5)),
                    (:none, x -> hankelh2(T(3), x), P(1.5)),

                    # 2-arg scaled Bessel/Hankel (1st arg gradient is `NotImplemented`)
                    (:none, x -> besselix(P(0.5), x), P(1.5)),
                    (:none, x -> besseljx(P(0.5), x), P(1.5)),
                    (:none, x -> besselkx(P(0.5), x), P(1.5)),
                    (:none, x -> besselyx(P(0.5), x), P(1.5)),
                    (:none, x -> hankelh1x(T(2), x), P(1.5)),
                    (:none, x -> hankelh2x(T(2), x), P(1.5)),

                    # 2-arg Gamma & exponential integrals (1st arg gradient is `NotImplemented`)
                    (:none, x -> gamma(T(3), x), P(1.5)),
                    (:none, x -> loggamma(T(3), x), P(1.5)),
                    (:none, x -> expintx(T(3), x), P(0.5)),
                    (:none, x -> expint(T(3), x), P(0.5)),

                    # Complex arguments
                    (:none, x -> besselj(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> besseli(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> bessely(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> besselk(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> hankelh1(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> hankelh2(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> besselix(P(0.5), Complex(x, x)), P(1.5)),
                    (:none, x -> besseljx(P(0.5), Complex(x, x)), P(1.5)),
                    (:none, x -> besselkx(P(0.5), Complex(x, x)), P(1.5)),
                    (:none, x -> besselyx(P(0.5), Complex(x, x)), P(1.5)),
                    (:none, x -> hankelh1x(T(0.5), Complex(x, x)), P(1.5)),
                    (:none, x -> hankelh2x(T(0.5), Complex(x, x)), P(1.5)),

                    # Both arguments for the functions below can be complex
                    (:none, x -> gamma(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> loggamma(T(3), Complex(x, x)), P(1.5)),
                    (:none, x -> expintx(T(3), Complex(x, x)), P(0.5)),
                    (:none, x -> expint(T(3), Complex(x, x)), P(0.5)),
                    (:none, x -> gamma(Complex(T(3), T(3)), x), P(1.5)),
                    (:none, x -> loggamma(Complex(T(3), T(3)), x), P(1.5)),
                    (:none, x -> expintx(Complex(T(3), T(3)), x), P(0.5)),
                    (:none, x -> expint(Complex(T(3), T(3)), x), P(0.5)),
                ]
            end...,
        )
            # Use `is_primitive = false` when testing closures over `SpecialFunctions`
            Mooncake.TestUtils.test_rule(
                StableRNG(123456), f, x...; perf_flag, is_primitive=false
            )
        end
    end

    @testset "NDual unsupported parameter directions" begin
        ν_active = NDual{Float64,1}(3.0, (1.0,))
        x_active = NDual{Float64,1}(1.5, (1.0,))
        @test_throws ArgumentError besselj(ν_active, x_active)
    end
end
