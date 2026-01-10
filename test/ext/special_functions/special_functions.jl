using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, JET, Mooncake, SpecialFunctions, StableRNGs, Test
using Mooncake: ForwardMode, ReverseMode
using Mooncake.TestUtils: test_rule

# Rules in this file are only lightly tester, because they are all just @from_rrule rules.
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
                (:stability_and_allocs, digamma, P(0.1)),
                (:stability_and_allocs, erf, P(0.1)),
                (:stability_and_allocs, erf, P(0.1), P(0.5)),
                (:stability_and_allocs, erfc, P(0.1)),
                (:stability_and_allocs, logerfc, P(0.1)),
                (:stability_and_allocs, erfcinv, P(0.1)),
                (:stability_and_allocs, erfcx, P(0.1)),
                (:stability_and_allocs, logerfcx, P(0.1)),
                (:stability_and_allocs, erfi, P(0.1)),
                (:stability_and_allocs, erfinv, P(0.1)),
                (:stability_and_allocs, gamma, P(0.1)),
                (:stability_and_allocs, invdigamma, P(0.1)),
                (:stability_and_allocs, trigamma, P(0.1)),
                (:stability_and_allocs, polygamma, 3, P(0.1)),
                (:stability_and_allocs, beta, P(0.3), P(0.1)),
                (:stability_and_allocs, logbeta, P(0.3), P(0.1)),
                (:stability_and_allocs, logabsgamma, P(0.3)),
                (:stability_and_allocs, loggamma, P(0.3)),
                (:stability_and_allocs, expint, P(0.3)),
                (:stability_and_allocs, expintx, P(0.3)),
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
                (:allocs, logabsbeta, P(0.3), P(0.1)),
            ]
        end...,

        # Functions which only support Float64.
        (:allocs, SpecialFunctions.gammax, 1.0),
        (:allocs, SpecialFunctions.rgammax, 3.0, 6.0),
        (:allocs, SpecialFunctions.rgamma1pm1, 0.1),
        (:allocs, SpecialFunctions.auxgam, 0.1),
        (:allocs, SpecialFunctions.loggamma1p, 0.3),
        (:allocs, SpecialFunctions.loggamma1p, -0.3),
        (:none, SpecialFunctions.lambdaeta, 5.0),
    )
        test_rule(StableRNG(123456), f, x...; perf_flag, is_primitive=false)
    end

    @testset "Primitive SpecialFunctions with Intractable Gradients" begin
        @testset "$perf_flag, $(typeof((f, x...)))" for (perf_flag, f, x...) in vcat(
            map([Float64, Float32]) do P
                return Any[
                    # 2 arg Standard Bessel & Hankel (1st arg gradient Intractable)
                    # (:stability, x -> besselj(P(0.5), x), P(1.5)),
                    # (:stability, x -> besseli(P(0.5), x), P(1.5)),
                    # (:stability, x -> bessely(P(0.5), x), P(1.5)),
                    # (:stability, x -> besselk(P(0.5), x), P(1.5)),
                    # (:stability, x -> hankelh1(P(0.5), x), P(1.5)),
                    # (:stability, x -> hankelh2(P(0.5), x), P(1.5)),

                    # 2 arg scaled bessel-i,j,k,y & hankelh1, hankelh2 (1st arg gradient Intractable)
                    # (last arg gradient Intractable)
                    # (:none, x -> besselix(P(0.5), x), P(1.5)),
                    # (:none, x -> besseljx(P(0.5), x), P(1.5)),
                    # (:none, x -> besselkx(P(0.5), x), P(1.5)),
                    # (:none, x -> besselyx(P(0.5), x), P(1.5)),
                    # (:none, x -> hankelh1x(P(0.5), x), P(1.5)),
                    # (:none, x -> hankelh2x(P(0.5), x), P(1.5)),

                    # 2 arg Gamma & Exponential Integrals (1st arg gradient Intractable)
                    (:stability, x -> gamma(P(2.0), x), P(1.5)),
                    (:stability, x -> loggamma(P(2.0), x), P(1.5)),
                    (:stability, x -> expint(P(1.0), x), P(0.5)),
                    (:stability, x -> expintx(P(1.0), x), P(0.5)),

                    # 3 arg gamma_inc (IND is 0/1, tangent a is 0 for AD but an approximation for testing FD)
                    (:stability, x -> gamma_inc(P(2), x, 0), P(2)),
                    (:stability, x -> gamma_inc(P(2), x, 1), P(2)),
                ]
            end...,
        )
            # flag is_primitive = false to test closures over SpecialFunctions.
            # This excludes gradient caclulations for `NotImplemented` fields.
            Mooncake.TestUtils.test_rule(
                StableRNG(123456), f, x...; perf_flag, is_primitive=false
            )
        end
    end
end
