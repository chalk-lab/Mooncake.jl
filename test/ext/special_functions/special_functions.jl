using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

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

# Helper methods to enable mixed Float32/Float64 operations. 
# Required for compatibility with Julia 1.12+.
Union{Float32,Float64}(x) = Float64(x)
Mooncake.increment!!(x::Float32, y::Float64) = Float32(x + y)
Mooncake.increment!!(x::Float64, y::Float32) = Float64(x + y)

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
                (VERSION >= v"1.11" ? :stability_and_allocs : :none, bessely1, P(0.1)),
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

    @testset "Primitive SpecialFunctions with `NotImplemented` gradients" begin
        first_arg_types = [Float64, Float32]
        second_arg_types = [Float64, Float32]

        # Check gradients while excluding those marked as `NotImplemented`.
        @testset "$perf_flag, $(typeof((f, x...)))" for (perf_flag, f, x...) in vcat(
            map_prod(first_arg_types, second_arg_types) do (T, P)
                return Any[
                    # 3-arg gamma_inc (IND is 0/1; tangent(a) is 0 in AD, but approximated in FD)
                    (:none, x -> gamma_inc(T(3), x, 0), P(2)),
                    (:none, x -> gamma_inc(T(3), x, 1), P(2)),

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

        a_active = NDual{Float64,1}(2.0, (1.0,))
        b_zero = NDual{Float64,1}(3.0, (0.0,))
        x_zero = NDual{Float64,1}(0.4, (0.0,))
        @test_throws ArgumentError beta_inc(a_active, b_zero, x_zero)
        @test_throws ArgumentError beta_inc(b_zero, a_active, x_zero)
    end
end
