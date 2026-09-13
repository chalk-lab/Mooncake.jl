using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LinearAlgebra, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule
using Mooncake.Nfwd: NDual

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    @testset for (perf_flag, is_primitive, f, x...) in vcat(
        map([Float64, Float32]) do P
            cases = Any[
                (:allocs, false, xlogx, P(1.1)),
                (:allocs, true, xlogy, P(0.3), P(1.2)),
                (:allocs, true, xlogy, P(0), P(3)),
                (:allocs, true, xlogy, P(0), 3),
                (:allocs, false, xlog1py, P(0.3), -P(0.5)),
                (:allocs, false, xexpx, -P(0.5)),
                (:allocs, true, xexpy, P(1.0), -P(0.7)),
                (:allocs, true, xexpy, P(0), P(2)),
                (:allocs, true, xexpy, P(0), 2),
                (:allocs, true, logistic, P(0.5)),
                (:allocs, true, logistic, P(1000.0)),
                (:allocs, false, logit, P(0.3)),
                (:allocs, false, logit, P(0.1)),
                (:allocs, false, logcosh, P(1.5)),
                (:allocs, false, logcosh, P(0.3)),
                (:allocs, false, logabssinh, P(0.3)),
                (:allocs, false, log1psq, P(0.3)),
                (:allocs, false, log1pexp, P(0.1)),
                (:allocs, false, log1mexp, -P(0.5)),
                (:allocs, false, log2mexp, P(0.1)),
                (:allocs, false, logexpm1, P(0.1)),
                (:allocs, false, log1pmx, -P(0.95)),
                (:allocs, false, log1pmx, P(0.1)),
                (:allocs, false, logmxp1, P(0.02)),
                (:allocs, true, logaddexp, -P(0.5), P(0.4)),
                # edge case with two equal inputs: see #881 for discussion
                (:allocs, true, logaddexp, P(1.5), P(1.5)),
                (:allocs, false, logsubexp, -P(0.5), -P(5.0)),
                (:allocs, true, logsumexp, randn(sr(1), P, 5)),
                (:allocs, true, logsumexp, randn(sr(2), P, 5, 4)),
                (:allocs, true, logsumexp, randn(sr(3), P, 5, 4, 3)),
                # subarray/view inputs, see #1035
                (:allocs, true, logsumexp, view(randn(sr(1), P, 5), 1:4)),
                # edge case with two equal inputs: see #881 for discussion
                (:allocs, true, logsumexp, [1.0, 1.0]),
                # Structured tangents: the adjoint is dense, so it has to be projected
                # onto what the wrapper stores. `Symmetric` folds rather than masks.
                (:none, true, logsumexp, UpperTriangular(randn(sr(20), P, 4, 4))),
                (:none, true, logsumexp, Diagonal(randn(sr(21), P, 4))),
                (:none, true, logsumexp, Symmetric(randn(sr(22), P, 4, 4))),
                (:none, false, x -> logsumexp(x; dims=1), randn(sr(4), P, 5, 4)),
                (:none, false, x -> logsumexp(x; dims=1), fill(1.0, 2, 2)),
                (:none, false, x -> logsumexp(x; dims=2), randn(sr(5), P, 5, 4)),
                (:none, false, x -> logsumexp(x; dims=2), fill(1.0, 2, 2)),
                # subarray/view inputs, see #1035
                (:none, false, x -> logsumexp(x; dims=2), view(fill(1.0, 3, 3), 1:2, 1:2)),
                # Non-strided SubArray, see #1296.
                (
                    :none,
                    false,
                    (x, inds) -> logsumexp(view(x,:,:,inds,:); dims=(3, 4))[1],
                    randn(sr(23), P, 2, 3, 5, 2),
                    [1, 3, 4],
                ),
                (:none, true, logsumexp!, rand(sr(6), P, 5), randn(sr(7), P, 5, 4)),
                (
                    :none,
                    true,
                    logsumexp!,
                    rand(sr(6), P, 5),
                    view(randn(sr(7), P, 5, 4), 1:5, 1:4),
                ),
                (:none, true, logsumexp!, [P(1.0)], [P(2.0), P(2.0)]),
                (:none, true, logsumexp!, [P(1.0)], view([P(2.0), P(2.0)], 1:2)),
                (:none, true, logsumexp!, view([P(1.0)], 1:1), view([P(2.0), P(2.0)], 1:2)),
                # not a primitive because the two inputs have different eltypes, but we can
                # still check that it runs correctly
                (
                    :none,
                    false,
                    logsumexp!,
                    rand(sr(6), Float64, 5),
                    randn(sr(7), Float32, 5, 4),
                ),
                (:none, false, softmax, randn(sr(7), P, 10)),
                # subarray/view inputs, see #1035
                (:none, false, softmax, view(randn(sr(7), P, 10), 1:5)),
                (:allocs, false, cloglog, P(0.5)),
                (:allocs, false, cexpexp, -P(0.3)),
                (:allocs, false, loglogistic, P(0.5)),
                (:allocs, false, logitexp, -P(0.3)),
                (:allocs, false, log1mlogistic, -P(0.9)),
                (:allocs, false, logit1mexp, -P(0.6)),
            ]
            @static if isdefined(LogExpFunctions, :logabstanh)
                push!(cases, (:allocs, false, LogExpFunctions.logabstanh, P(0.3)))
                push!(cases, (:allocs, false, LogExpFunctions.logabstanh, P(1.5)))
            end
            return cases
        end...,
    )
        test_rule(sr(123456), f, x...; perf_flag, is_primitive)
    end

    @testset "zero multipliers and inactive directions" begin
        for T in (Float16, Float32, Float64),
            (f, x, y, a, b) in (
                (xlogy, 0, 3, log(T(3)), 0),
                (xlogy, 0, 0, -Inf, 0),
                (xlogy, 0, Inf, Inf, 0),
                (xexpy, 0, 2, exp(T(2)), 0),
                (xexpy, 0, 1000, Inf, 0),
                (xexpy, 1, -Inf, 0, 0),
            )

            x, y, a, b = T(x), T(y), T(a), T(b)
            z, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, (f, x, y))...)
            @test isequal(Mooncake.primal(z), f(x, y))
            @test pb(one(T))[2:3] == (a, b)
            @test pb(zero(T))[2:3] == (zero(T), zero(T))
            for (dx, dy) in ((1, 0), (0, 1), (0, 0), (1, 1))
                dx, dy = T(dx), T(dy)
                expected = (iszero(dx) ? zero(T) : a) + (iszero(dy) ? zero(T) : b)
                result = Mooncake.frule!!(
                    Mooncake.zero_dual(f), Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)
                )
                @test isequal(Mooncake.primal(result), f(x, y))
                @test Mooncake.tangent(result) == expected
                result = f(NDual(x, (dx,)), NDual(y, (dy,)))
                @test isequal(result.value, f(x, y))
                @test only(result.partials) == expected
            end
            @test only(f(NDual(x, (one(T),)), y).partials) == a
            @test only(f(x, NDual(y, (one(T),))).partials) == b
        end
        for f in (xlogy, xexpy)
            test_rule(sr(123), f, 0.0f0, 2.0; is_primitive=true, perf_flag=:allocs)
            for (x, y) in ((0.0f0, 2.0), (0.3, 1.2f0))
                a = f === xlogy ? log(y) : exp(y)
                b = f === xlogy ? x / y : f(x, y)
                result = f(NDual(x, (one(x),)), NDual(y, (zero(y),)))
                @test result.value === f(x, y)
                @test only(result.partials) == a
                result = f(NDual(x, (one(x),)), y)
                @test result.value === f(x, y)
                @test only(result.partials) == a
                result = f(x, NDual(y, (one(y),)))
                @test result.value === f(x, y)
                @test only(result.partials) == b
            end
        end
    end
end
