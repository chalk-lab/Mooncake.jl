using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, LinearAlgebra, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule
using Mooncake: ForwardMode, ReverseMode, zero_tangent
using Mooncake.Nfwd: NDual

sr(n::Int) = StableRNG(n)

@testset "structured projections" for row in Mooncake.TestResources.structured_rule_cases(
    logsumexp
)
    interface_only, perf_flag, opts, f, args... = row
    test_rule(sr(123), f, args...; interface_only, perf_flag, opts...)
end

@testset "structured output" for row in
                                 Mooncake.TestResources.structured_output_cases(logsumexp!)
    interface_only, perf_flag, opts, f, args... = row
    test_rule(sr(123), f, args...; interface_only, perf_flag, opts...)
end

@testset "logexpfunctions" begin
    @testset for (perf_flag, is_primitive, f, x...) in vcat(
        map([Float64, Float32]) do P
            cases = Any[
                (:allocs, true, xlogx, P(1.1)),
                (:allocs, true, xlogy, P(0.3), P(1.2)),
                (:allocs, true, xlogy, P(0), P(3)),
                (:allocs, true, xlogy, P(0), 3),
                (:allocs, true, xlog1py, P(0.3), -P(0.5)),
                (:allocs, true, xlog1py, P(0), -P(0.5)),
                (:allocs, true, xlog1py, P(0), 3),
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
                # `out` aliases `x`, whose first column the rule must read before writing.
                (
                    :none,
                    false,
                    A -> sum(logsumexp!(view(A, :, 1), A)),
                    P[0.3 0.7; -0.2 1.1],
                ),
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

    @testset "derivatives beyond finite differences" begin
        # Rounding hides a unit translation of `2^precision(P)` and `logistic`'s tail slope,
        # and an infinite entry makes every perturbed value infinite. Both modes give `d`.
        derivative_is(d) =
            (mode, f, t) -> begin
                cache = Mooncake.prepare_derivative_cache(f, t)
                fwd = Mooncake.value_and_derivative!!(
                    cache, (f, zero_tangent(f)), (t, one(t))
                )[2]
                rvs = Mooncake.value_and_gradient!!(
                    Mooncake.prepare_gradient_cache(f, t), f, t
                )
                return (mode === ReverseMode || isapprox(fwd, d; rtol=8eps(typeof(d)))) && (
                    mode === ForwardMode || isapprox(rvs[2][2], d; rtol=8eps(typeof(d)))
                )
            end
        # The Hessian at a tie of `n` zeros is `(I - 1/n) / n`; first-order checks miss
        # a rule whose primal pass drops the tied entries' derivatives.
        function hvp_at_tie(_, f, x)
            n = length(x)
            v = [1; zeros(n - 1)]
            _, _, hv = Mooncake.value_and_hvp!!(Mooncake.prepare_hvp_cache(f, x), f, v, x)
            return isapprox(hv, (v .- 1 / n) ./ n)
        end
        lses = (
            logsumexp,
            x -> logsumexp(x; dims=:),
            x -> only(logsumexp(x; dims=1)),
            x -> only(logsumexp!(zeros(eltype(x), 1), x)),
        )
        cases = Any[]
        for P in (Float64, Float32)
            c = P(2)^precision(P)
            for lse in lses
                push!(cases, (derivative_is(one(P)), t -> lse(fill(c + t, 2)), zero(P)))
                # A zero seed on the infinite entry contributes zero.
                push!(cases, (derivative_is(zero(P)), t -> lse([P(Inf), t]), zero(P)))
                # The maximal direction stays finite once normalised.
                f = t -> lse(fill(floatmax(P) * t, 2))
                push!(cases, (derivative_is(floatmax(P)), f, zero(P)))
            end
            # A zero cotangent on the infinite column contributes zero.
            g = t -> logsumexp(P[Inf+t 1; 0 2]; dims=1)[2]
            push!(cases, (derivative_is(zero(P)), g, zero(P)))
            g! = t -> logsumexp!(zeros(P, 1, 2), P[Inf+t 1; 0 2])[2]
            push!(cases, (derivative_is(zero(P)), g!, zero(P)))
            push!(cases, (derivative_is(P(0.5)), t -> logaddexp(c + t, c), zero(P)))
            d = P(exp(-40) / (1 + exp(-40))^2)
            push!(cases, (derivative_is(d), t -> logistic(P(40) + t), zero(P)))
        end
        # Float16 sums saturate at 2048 unit terms and overflow at 65504. At `2^16` zeros,
        # each weight is `2^-16` and half the entries carry half the derivative.
        function weights_are_uniform(mode, f, x)
            dx = Float16.(isodd.(eachindex(x)))
            cache = Mooncake.prepare_derivative_cache(f, x)
            fwd = Mooncake.value_and_derivative!!(cache, (f, zero_tangent(f)), (x, dx))[2]
            cache = Mooncake.prepare_gradient_cache(f, x)
            rvs = Mooncake.value_and_gradient!!(cache, f, x)[2][2]
            return (mode === ReverseMode || fwd == 0.5) &&
                   (mode === ForwardMode || all(==(Float16(2)^-16), rvs))
        end
        for lse in lses
            push!(cases, (weights_are_uniform, lse, zeros(Float16, 2^16)))
        end
        for n in (2, 17), lse in lses
            push!(cases, (hvp_at_tie, lse, zeros(n)))
        end
        push!(cases, (hvp_at_tie, x -> logaddexp(x[1], x[2]), zeros(2)))
        # An abstract element type, whose weights take their type from the result.
        function gradient_is_softmax(_, f, x)
            cache = Mooncake.prepare_gradient_cache(f, x)
            return all(
                isapprox.(Mooncake.value_and_gradient!!(cache, f, x)[2][2], softmax(x))
            )
        end
        push!(cases, (gradient_is_softmax, logsumexp, Union{Float32,Float64}[1.0f0, 2.0]))
        @testset for (oracle, f, x) in cases
            test_rule(sr(1), f, x; is_primitive=false, oracle)
        end
    end

    @testset "zero multipliers and inactive directions" begin
        # Finite differences cannot check the infinite boundary slope or inactive lanes.
        for T in (Float16, Float32, Float64), x in (zero(T), -zero(T), nextfloat(zero(T)))
            d = log(x) + one(T)
            z, pb = Mooncake.rrule!!(Mooncake.zero_fcodual(xlogx), Mooncake.zero_fcodual(x))
            @test isequal(Mooncake.primal(z), xlogx(x))
            @test pb(one(T))[2] == d
            @test pb(zero(T))[2] == zero(T)
            for dx in (one(T), -one(T), zero(T))
                expected = iszero(dx) ? zero(T) : d * dx
                result = Mooncake.frule!!(Mooncake.zero_dual(xlogx), Mooncake.Dual(x, dx))
                @test isequal(Mooncake.primal(result), xlogx(x))
                @test Mooncake.tangent(result) == expected
                for N in (1, 8)
                    result = xlogx(NDual(x, ntuple(k -> isodd(k) ? dx : zero(T), N)))
                    @test isequal(result.value, xlogx(x))
                    @test result.partials == ntuple(k -> isodd(k) ? expected : zero(T), N)
                end
            end
        end
        for T in (Float16, Float32, Float64),
            (f, x, y, a, b) in (
                (xlogy, 0, 3, log(T(3)), 0),
                (xlogy, 0, 0, -Inf, 0),
                (xlogy, 0, Inf, Inf, 0),
                (xlog1py, 0, -0.5, log1p(T(-0.5)), 0),
                (xlog1py, 0, -1, -Inf, 0),
                (xlog1py, 0, Inf, Inf, 0),
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
                for N in (1, 8)
                    result = f(
                        NDual(x, ntuple(k -> isodd(k) ? dx : zero(T), N)),
                        NDual(y, ntuple(k -> isodd(k) ? dy : zero(T), N)),
                    )
                    @test isequal(result.value, f(x, y))
                    @test result.partials == ntuple(k -> isodd(k) ? expected : zero(T), N)
                end
            end
            @test only(f(NDual(x, (one(T),)), y).partials) == a
            @test only(f(x, NDual(y, (one(T),))).partials) == b
        end
        for f in (xlogy, xlog1py, xexpy)
            test_rule(sr(123), f, 0.0f0, 2.0; is_primitive=true, perf_flag=:allocs)
            for (x, y) in ((0.0f0, 2.0), (0.3, 1.2f0))
                a, b = if f === xlogy
                    log(y), x / y
                elseif f === xlog1py
                    log1p(y), x / (one(y) + y)
                else
                    exp(y), f(x, y)
                end
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

    @testset "mixed derivative at a zero multiplier" begin
        # A first-order rule check cannot detect a lost derivative of the zero y-partial.
        function dy(x)
            _, pb = Mooncake.rrule!!(
                Mooncake.zero_fcodual(xlog1py),
                Mooncake.zero_fcodual(x),
                Mooncake.zero_fcodual(oftype(x, -0.5)),
            )
            return pb(one(x))[3]
        end
        for T in (Float32, Float64)
            rule = Mooncake.build_frule(dy, zero(T))
            @test Mooncake.value_and_derivative!!(
                rule, (dy, Mooncake.NoTangent()), (zero(T), one(T))
            ) == (zero(T), T(2))
        end
    end
end
