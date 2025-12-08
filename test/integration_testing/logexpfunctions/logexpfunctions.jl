using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    @testset for (perf_flag, interface_only, is_primitive, f, x...) in vcat(
        map([Float64, Float32]) do P
            return Any[
                (:allocs, false, true, xlogx, P(1.1)),
                (:allocs, false, true, xlogy, P(0.3), P(1.2)),
                (:allocs, false, true, xlog1py, P(0.3), -P(0.5)),
                (:allocs, false, true, xexpx, -P(0.5)),
                (:allocs, false, true, xexpy, P(1.0), -P(0.7)),
                (:allocs, false, true, logistic, P(0.5)),
                (:allocs, false, true, logistic, P(1000.0)),
                (:allocs, false, true, logit, P(0.3)),
                (:allocs, false, true, logcosh, P(1.5)),
                (:allocs, false, true, logabssinh, P(0.3)),
                (:allocs, false, true, log1psq, P(0.3)),
                (:allocs, false, true, log1pexp, P(0.1)),
                (:allocs, false, true, log1mexp, -P(0.5)),
                (:allocs, false, true, log2mexp, P(0.1)),
                (:allocs, false, true, logexpm1, P(0.1)),
                (:allocs, false, true, log1pmx, -P(0.95)),
                (:allocs, false, true, logmxp1, P(0.02)),
                (:allocs, false, true, logaddexp, -P(0.5), P(0.4)),
                (:allocs, P == Float32, true, logaddexp, P(1000.0), P(1000.0)),
                (:allocs, false, true, logsubexp, -P(0.5), -P(5.0)),
                (:allocs, false, true, logsumexp, randn(sr(1), P, 5)),
                (:allocs, false, true, logsumexp, randn(sr(2), P, 5, 4)),
                (:allocs, false, true, logsumexp, randn(sr(3), P, 5, 4, 3)),
                # edge case with two equal inputs: see #881 for discussion
                (:allocs, false, true, logsumexp, [1.0, 1.0]),
                (:none, false, false, x -> logsumexp(x; dims=1), randn(sr(4), P, 5, 4)),
                (:none, false, false, x -> logsumexp(x; dims=1), fill(1.0, 2, 2)),
                (:none, false, false, x -> logsumexp(x; dims=2), randn(sr(5), P, 5, 4)),
                (:none, false, false, x -> logsumexp(x; dims=2), fill(1.0, 2, 2)),
                (:none, true, true, logsumexp!, rand(sr(6), 5), randn(sr(7), P, 5, 4)),
                (:none, true, true, logsumexp!, [0.0], [1.0, 1.0]),
                (:none, false, false, softmax, randn(sr(7), P, 10)),
                (:allocs, false, true, cloglog, P(0.5)),
                (:allocs, false, true, cexpexp, -P(0.3)),
                (:allocs, false, false, loglogistic, P(0.5)),
                (:allocs, false, false, logitexp, -P(0.3)),
                (:allocs, false, false, log1mlogistic, -P(0.9)),
                (:allocs, false, false, logit1mexp, -P(0.6)),
            ]
        end...,
    )
        test_rule(sr(123456), f, x...; perf_flag, is_primitive, interface_only)
    end
end
