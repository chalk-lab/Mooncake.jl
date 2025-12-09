using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    @testset for (perf_flag, rtol, is_primitive, f, x...) in vcat(
        map([Float64, Float32]) do P
            return Any[
                (:allocs, nothing, false, xlogx, P(1.1)),
                (:allocs, nothing, false, xlogy, P(0.3), P(1.2)),
                (:allocs, nothing, false, xlog1py, P(0.3), -P(0.5)),
                (:allocs, nothing, false, xexpx, -P(0.5)),
                (:allocs, nothing, false, xexpy, P(1.0), -P(0.7)),
                (:allocs, nothing, true, logistic, P(0.5)),
                (:allocs, nothing, true, logistic, P(1000.0)),
                (:allocs, nothing, false, logit, P(0.3)),
                (:allocs, nothing, false, logcosh, P(1.5)),
                (:allocs, nothing, false, logabssinh, P(0.3)),
                (:allocs, nothing, false, log1psq, P(0.3)),
                (:allocs, nothing, false, log1pexp, P(0.1)),
                (:allocs, nothing, false, log1mexp, -P(0.5)),
                (:allocs, nothing, false, log2mexp, P(0.1)),
                (:allocs, nothing, false, logexpm1, P(0.1)),
                (:allocs, nothing, false, log1pmx, -P(0.95)),
                (:allocs, nothing, false, logmxp1, P(0.02)),
                (:allocs, nothing, true, logaddexp, -P(0.5), P(0.4)),
                (:allocs, 1e-2, true, logaddexp, P(1000.0), P(1000.0)),
                (:allocs, nothing, false, logsubexp, -P(0.5), -P(5.0)),
                (:allocs, nothing, true, logsumexp, randn(sr(1), P, 5)),
                (:allocs, nothing, true, logsumexp, randn(sr(2), P, 5, 4)),
                (:allocs, nothing, true, logsumexp, randn(sr(3), P, 5, 4, 3)),
                # edge case with two equal inputs: see #881 for discussion
                (:allocs, nothing, true, logsumexp, [1.0, 1.0]),
                (:none, nothing, false, x -> logsumexp(x; dims=1), randn(sr(4), P, 5, 4)),
                (:none, nothing, false, x -> logsumexp(x; dims=1), fill(1.0, 2, 2)),
                (:none, nothing, false, x -> logsumexp(x; dims=2), randn(sr(5), P, 5, 4)),
                (:none, nothing, false, x -> logsumexp(x; dims=2), fill(1.0, 2, 2)),
                (:none, nothing, true, logsumexp!, rand(sr(6), P, 5), randn(sr(7), P, 5, 4)),
                (:none, nothing, true, logsumexp!, [P(1.0)], [P(2.0), P(2.0)]),
                # not a primitive because the two inputs have different eltypes, but we can
                # still check that it runs correctly
                (
                    :none,
                    nothing,
                    false,
                    logsumexp!,
                    rand(sr(6), Float64, 5),
                    randn(sr(7), Float32, 5, 4),
                ),
                (:none, nothing, false, softmax, randn(sr(7), P, 10)),
                (:allocs, nothing, false, cloglog, P(0.5)),
                (:allocs, nothing, false, cexpexp, -P(0.3)),
                (:allocs, nothing, false, loglogistic, P(0.5)),
                (:allocs, nothing, false, logitexp, -P(0.3)),
                (:allocs, nothing, false, log1mlogistic, -P(0.9)),
                (:allocs, nothing, false, logit1mexp, -P(0.6)),
            ]
        end...,
    )
        rtol_kwarg = rtol === nothing ? () : (rtol=rtol,)
        test_rule(sr(123456), f, x...; perf_flag, is_primitive, rtol_kwarg...)
    end
end
