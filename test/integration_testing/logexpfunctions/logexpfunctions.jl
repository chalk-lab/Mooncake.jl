using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    @testset for (perf_flag, f, x...) in vcat(
        map([Float64, Float32]) do P
            return Any[
                (:allocs, xlogx, P(1.1)),
                (:allocs, xlogy, P(0.3), P(1.2)),
                (:allocs, xlog1py, P(0.3), -P(0.5)),
                (:allocs, xexpx, -P(0.5)),
                (:allocs, xexpy, P(1.0), -P(0.7)),
                (:allocs, logistic, P(0.5)),
                (:allocs, logit, P(0.3)),
                (:allocs, logcosh, P(1.5)),
                (:allocs, logabssinh, P(0.3)),
                (:allocs, log1psq, P(0.3)),
                (:allocs, log1pexp, P(0.1)),
                (:allocs, log1mexp, -P(0.5)),
                (:allocs, log2mexp, P(0.1)),
                (:allocs, logexpm1, P(0.1)),
                (:allocs, log1pmx, -P(0.95)),
                (:allocs, logmxp1, P(0.02)),
                (:allocs, logaddexp, -P(0.5), P(0.4)),
                (:allocs, logsubexp, -P(0.5), -P(5.0)),
                (:allocs, logsumexp, randn(sr(1), P, 5)),
                (:allocs, logsumexp, randn(sr(2), P, 5, 4)),
                (:allocs, logsumexp, randn(sr(3), P, 5, 4, 3)),
                (:none, x -> logsumexp(x; dims=1), randn(sr(4), P, 5, 4)),
                (:none, x -> logsumexp(x; dims=2), randn(sr(5), P, 5, 4)),
                (:none, logsumexp!, rand(sr(6), 5), randn(sr(7), P, 5, 4)),
                (:none, softmax, randn(sr(7), P, 10)),
                (:allocs, cloglog, P(0.5)),
                (:allocs, cexpexp, -P(0.3)),
                (:allocs, loglogistic, P(0.5)),
                (:allocs, logitexp, -P(0.3)),
                (:allocs, log1mlogistic, -P(0.9)),
                (:allocs, logit1mexp, -P(0.6)),
            ]
        end...,
    )
        test_rule(sr(123456), f, x...; perf_flag, is_primitive=false)
    end
end
