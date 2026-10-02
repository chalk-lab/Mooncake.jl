include(joinpath(@__DIR__, "..", "..", "ext", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "LogExpFunctions")

using AllocCheck, LinearAlgebra, LogExpFunctions, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: TestCase, test_rule
using Mooncake.Nfwd: NDual

sr(n::Int) = StableRNG(n)

@testset "logexpfunctions" begin
    test_cases = vcat(
        map([Float64, Float32]) do P
            cases = TestCase[
                TestCase(xlogx, P(1.1)),
                TestCase(xlogy, P(0.3), P(1.2)),
                TestCase(xlogy, P(0), P(3)),
                TestCase(xlogy, P(0), 3),
                TestCase(xlog1py, P(0.3), -P(0.5)),
                TestCase(xlog1py, P(0), -P(0.5)),
                TestCase(xlog1py, P(0), 3),
                TestCase(xexpx, -P(0.5); is_primitive=false),
                TestCase(xexpy, P(1.0), -P(0.7)),
                TestCase(xexpy, P(0), P(2)),
                TestCase(xexpy, P(0), 2),
                TestCase(logistic, P(0.5)),
                TestCase(logistic, P(1000.0)),
                TestCase(logit, P(0.3); is_primitive=false),
                TestCase(logit, P(0.1); is_primitive=false),
                TestCase(logcosh, P(1.5); is_primitive=false),
                TestCase(logcosh, P(0.3); is_primitive=false),
                TestCase(logabssinh, P(0.3); is_primitive=false),
                TestCase(log1psq, P(0.3); is_primitive=false),
                TestCase(log1pexp, P(0.1); is_primitive=false),
                TestCase(log1mexp, -P(0.5); is_primitive=false),
                TestCase(log2mexp, P(0.1); is_primitive=false),
                TestCase(logexpm1, P(0.1); is_primitive=false),
                TestCase(log1pmx, -P(0.95); is_primitive=false),
                TestCase(log1pmx, P(0.1); is_primitive=false),
                TestCase(logmxp1, P(0.02); is_primitive=false),
                TestCase(logaddexp, -P(0.5), P(0.4)),
                # edge case with two equal inputs: see #881 for discussion
                TestCase(logaddexp, P(1.5), P(1.5)),
                TestCase(logsubexp, -P(0.5), -P(5.0); is_primitive=false),
                TestCase(logsumexp, randn(sr(1), P, 5)),
                TestCase(logsumexp, randn(sr(2), P, 5, 4)),
                TestCase(logsumexp, randn(sr(3), P, 5, 4, 3)),
                # subarray/view inputs, see #1035
                TestCase(logsumexp, view(randn(sr(1), P, 5), 1:4)),
                # edge case with two equal inputs: see #881 for discussion
                TestCase(logsumexp, [1.0, 1.0]),
                # Structured tangents: the adjoint is dense, so it has to be projected
                # onto what the wrapper stores. `Symmetric` folds rather than masks.
                TestCase(
                    logsumexp, UpperTriangular(randn(sr(20), P, 4, 4)); perf_flag=:none
                ),
                TestCase(logsumexp, UnitUpperTriangular(zeros(P, 2, 2)); perf_flag=:none),
                TestCase(logsumexp, UnitLowerTriangular(zeros(P, 2, 2)); perf_flag=:none),
                TestCase(
                    x -> logsumexp(x; dims=1),
                    UnitUpperTriangular(zeros(P, 2, 2));
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    x -> logsumexp(x; dims=2),
                    UnitLowerTriangular(zeros(P, 2, 2));
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    logsumexp!,
                    zeros(P, 1, 2),
                    UnitUpperTriangular(zeros(P, 2, 2));
                    perf_flag=:none,
                ),
                TestCase(
                    logsumexp!,
                    zeros(P, 2, 1),
                    UnitLowerTriangular(zeros(P, 2, 2));
                    perf_flag=:none,
                ),
                TestCase(logsumexp, Diagonal(randn(sr(21), P, 4)); perf_flag=:none),
                TestCase(logsumexp, Symmetric(randn(sr(22), P, 4, 4)); perf_flag=:none),
                TestCase(
                    x -> logsumexp(x; dims=1),
                    randn(sr(4), P, 5, 4);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    x -> logsumexp(x; dims=1),
                    fill(1.0, 2, 2);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    x -> logsumexp(x; dims=2),
                    randn(sr(5), P, 5, 4);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    x -> logsumexp(x; dims=2),
                    fill(1.0, 2, 2);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                # subarray/view inputs, see #1035
                TestCase(
                    x -> logsumexp(x; dims=2),
                    view(fill(1.0, 3, 3), 1:2, 1:2);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                # Non-strided SubArray, see #1296.
                TestCase(
                    (x, inds) -> logsumexp(view(x,:,:,inds,:); dims=(3, 4))[1],
                    randn(sr(23), P, 2, 3, 5, 2),
                    [1, 3, 4];
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(
                    logsumexp!, rand(sr(6), P, 5), randn(sr(7), P, 5, 4); perf_flag=:none
                ),
                TestCase(
                    logsumexp!,
                    rand(sr(6), P, 5),
                    view(randn(sr(7), P, 5, 4), 1:5, 1:4);
                    perf_flag=:none,
                ),
                TestCase(logsumexp!, [P(1.0)], [P(2.0), P(2.0)]; perf_flag=:none),
                TestCase(
                    logsumexp!,
                    view(UnitUpperTriangular(zeros(P, 2, 2)), 1:1, 2:2),
                    ones(P, 1, 1);
                    perf_flag=:none,
                ),
                TestCase(
                    logsumexp!,
                    view(UnitLowerTriangular(zeros(P, 2, 2)), 2:2, 1:1),
                    ones(P, 1, 1);
                    perf_flag=:none,
                ),
                TestCase(
                    logsumexp!, [P(1.0)], view([P(2.0), P(2.0)], 1:2); perf_flag=:none
                ),
                TestCase(
                    logsumexp!,
                    view([P(1.0)], 1:1),
                    view([P(2.0), P(2.0)], 1:2);
                    perf_flag=:none,
                ),
                # not a primitive because the two inputs have different eltypes, but we can
                # still check that it runs correctly
                TestCase(
                    logsumexp!,
                    rand(sr(6), Float64, 5),
                    randn(sr(7), Float32, 5, 4);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(softmax, randn(sr(7), P, 10); perf_flag=:none, is_primitive=false),
                # subarray/view inputs, see #1035
                TestCase(
                    softmax,
                    view(randn(sr(7), P, 10), 1:5);
                    perf_flag=:none,
                    is_primitive=false,
                ),
                TestCase(cloglog, P(0.5); is_primitive=false),
                TestCase(cexpexp, -P(0.3); is_primitive=false),
                TestCase(loglogistic, P(0.5); is_primitive=false),
                TestCase(logitexp, -P(0.3); is_primitive=false),
                TestCase(log1mlogistic, -P(0.9); is_primitive=false),
                TestCase(logit1mexp, -P(0.6); is_primitive=false),
            ]
            for W in (UnitUpperTriangular, UnitLowerTriangular)
                x = W(randn(sr(24), P, 2, 2))
                V = if W === UnitUpperTriangular
                    UnitLowerTriangular
                else
                    UnitUpperTriangular
                end
                push!(
                    cases,
                    TestCase(logsumexp, view(x, [2, 1, 2], [2, 1]); perf_flag=:none),
                    TestCase(
                        Core.kwcall, (; dims=:), logsumexp, reshape(x, 4); perf_flag=:none
                    ),
                    TestCase(
                        Core.kwcall,
                        (; dims=1),
                        logsumexp,
                        Adjoint(view(x, :, :));
                        perf_flag=:none,
                    ),
                    TestCase(
                        logsumexp!,
                        zeros(P, 1, 2),
                        Transpose(view(x, :, :));
                        perf_flag=:none,
                    ),
                    TestCase(logsumexp, Diagonal(reshape(x, 4)); perf_flag=:none),
                    TestCase(logsumexp, UpperTriangular(x); perf_flag=:none),
                    TestCase(logsumexp, LowerTriangular(x); perf_flag=:none),
                    TestCase(logsumexp, V(view(x, :, :)); perf_flag=:none),
                )
            end
            @static if isdefined(LogExpFunctions, :logabstanh)
                push!(
                    cases, TestCase(LogExpFunctions.logabstanh, P(0.3); is_primitive=false)
                )
                push!(
                    cases, TestCase(LogExpFunctions.logabstanh, P(1.5); is_primitive=false)
                )
            end
            return cases
        end...,
    )
    # First-order checks cannot detect a lost derivative of the structural mask.
    for (T, i) in ((UnitUpperTriangular, 3), (UnitLowerTriangular, 2))
        f(x) = logsumexp(T(reshape(x, 2, 2)))
        v = zeros(4)
        v[i] = 1.0
        p = inv(2 + 2exp(1))
        push!(
            test_cases,
            TestCase(
                f,
                zeros(4);
                name="unit-triangular HVP $T",
                hvp=(
                    check=:reference,
                    directions=(zeros(4), v),
                    reference=(hvp=v -> p * (1-p) * v,),
                    cmp=isapprox,
                ),
            ),
        )
    end
    # Allocation checks are a fallback; allocating test cases opt out explicitly.
    for (tc, name) in zip(test_cases, Mooncake.TestUtils._test_case_names(test_cases))
        test_rule(sr(123456), tc; fallbacks=(perf_flag=:allocs,), name)
    end

    # Assert primitive dispatch only in forward mode; reverse uses derived rules.
    @testset "forward-only primitives" begin
        for P in (Float64, Float32), (f, x) in ((log1psq, P(0.3)), (log2mexp, P(0.1)))
            test_rule(
                sr(123456),
                f,
                x;
                perf_flag=:allocs,
                is_primitive=true,
                mode=Mooncake.ForwardMode,
            )
        end
    end

    @testset "nested unit-triangular reads" begin
        for P in (Float64, Float32),
            W in (UnitUpperTriangular, UnitLowerTriangular),
            wrap in (x -> view(x, :, :), x -> reshape(x, 4), Symmetric)

            test_rule(
                sr(123456),
                logsumexp,
                wrap(W(zeros(P, 2, 2)));
                mode=wrap === Symmetric ? Mooncake.ForwardMode : nothing,
                is_primitive=true,
            )
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
                result = Mooncake.frule!!(Mooncake.zero_dual(xlogx), Mooncake.lift(x, dx))
                @test isequal(Mooncake.primal(result), xlogx(x))
                @test only(Mooncake.tangent(result).partials) == expected
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
                    Mooncake.zero_dual(f), Mooncake.lift(x, dx), Mooncake.lift(y, dy)
                )
                @test isequal(Mooncake.primal(result), f(x, y))
                @test only(Mooncake.tangent(result).partials) == expected
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
