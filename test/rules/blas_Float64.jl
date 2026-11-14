@testset "blas (basic)" begin

    # arrayify tests are not precision-specific; placed here so they run in exactly one
    # CI job. Problems with arrayify tend to surface as confusing failures in the rule
    # tests that use it, so it is worth unit-testing separately.
    @testset "arrayify" begin

        # Verify that an unexpected type throws a sensible error.
        @test_throws "Encountered unexpected array type" Mooncake.arrayify(5, 4)

        # Verify all test cases can be array-ified.
        @testset "$P" for P in [Float32, Float64, ComplexF32, ComplexF64]
            xs = vcat(
                Mooncake.blas_matrices(StableRNG(123), P, 2, 3),
                Mooncake.special_matrices(StableRNG(123), P, 2, 3),
                Mooncake.blas_vectors(StableRNG(123), P, 2),
            )
            @testset "$(typeof(x)), $f" for x in xs, f in [identity, fdata]
                t = f(Mooncake.randn_tangent(StableRNG(123), x))
                _x, _t = Mooncake.arrayify(Mooncake.CoDual(x, t))

                # The primal should be the same thing.
                @test _x === x

                # The data underlying the tangent / fdata returned from arrayify must alias
                # the original. To check that this happens, we check that if we run arrayify a
                # second time on the same input, and mutate the tangent, the values in `_t`
                # are modified in exactly the same way.
                _, _t2 = Mooncake.arrayify(Mooncake.CoDual(x, t))
                _t2 .= zero(P)
                @test _t == _t2
            end
        end
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end

@testset "zero BLAS cotangents" for P in (Float32, Float64, ComplexF32, ComplexF64)
    A, N, x, C = P[2 1; 0 3], fill(P(NaN), 2, 2), ones(P, 2), zeros(P, 2, 2)
    cases = Any[
        (BLAS.scal!, 2, P(NaN), x, 1),
        (BLAS.axpy!, 2, P(NaN), x, 1, zeros(P, 2), 1),
        (BLAS.nrm2, 2, fill(P(NaN), 2), 1),
    ]
    for (f, flags, vector) in (
            (BLAS.gemm!, ('N', 'N'), false),
            (BLAS.symm!, ('L', 'U'), false),
            (BLAS.gemv!, ('N',), true),
            (BLAS.symv!, ('U',), true),
            (
                if P <: Complex
                    ((BLAS.hemm!, ('L', 'U'), false), (BLAS.hemv!, ('U',), true))
                else
                    ()
                end
            )...,
        ),
        badarg in (1, 2, 3, 4)

        push!(
            cases,
            (
                f,
                flags...,
                badarg == 3 ? P(NaN) : one(P),
                badarg == 1 ? N : A,
                vector ? (badarg == 2 ? fill(P(NaN), 2) : x) : (badarg == 2 ? N : copy(A)),
                badarg == 4 ? P(NaN) : zero(P),
                vector ? zeros(P, 2) : C,
            ),
        )
    end
    for f in (BLAS.trmv!, BLAS.trsv!)
        push!(cases, (f, 'U', 'N', 'N', N, x))
    end
    for f in (BLAS.trmm!, BLAS.trsm!)
        push!(cases, (f, 'L', 'U', 'N', 'N', one(P), N, copy(A)))
    end
    for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...)
        R = f === BLAS.herk! ? real(P) : P
        push!(cases, (f, 'U', 'N', one(R), N, zero(R), C))
    end
    for f in (P <: Real ? (BLAS.dot,) : (BLAS.dotc, BLAS.dotu))
        push!(cases, (f, fill(P(NaN), 2), x))
    end
    @testset "$(first(args))" for args in cases
        ds = map(Mooncake.zero_fcodual, deepcopy(args))
        rule =
            first(args) in (BLAS.dot, BLAS.dotc, BLAS.dotu) ? build_rrule(args...) : rrule!!
        out, pb = rule(ds...)
        r = pb(primal(out) isa Number ? zero(primal(out)) : NoRData())
        @test all(
            i -> !(args[i] isa AbstractArray) || all(iszero, Mooncake.tangent(ds[i])),
            eachindex(args),
        ) && all(v -> !(v isa Number) || iszero(v), r)
    end
end

@testset "BLAS scalar cotangents" for P in (Float64, ComplexF64)
    for f in (BLAS.gemv!, BLAS.symv!, (P <: Complex ? (BLAS.hemv!,) : ())...),
        which in (:alpha, :beta)

        A = which === :alpha ? P[NaN 1; 1 2] : P[2 1; 1 2]
        y = which === :beta ? P[NaN, 3] : zeros(P, 2)
        ds = map(
            Mooncake.zero_fcodual,
            (f, f === BLAS.gemv! ? 'N' : 'U', P(2), A, ones(P, 2), P(2), y),
        )
        out, pb = rrule!!(ds...)
        Mooncake.tangent(out) .= P[0, 1]
        @test pb(NoRData())[which === :alpha ? 3 : 6] ≈ P(3)
    end
    for args in (
        (BLAS.scal!, 2, P(2), P[NaN, 3], 1),
        (BLAS.axpy!, 2, P(2), P[NaN, 3], 1, zeros(P, 2), 1),
    )
        ds = map(Mooncake.zero_fcodual, args)
        out, pb = rrule!!(ds...)
        Mooncake.tangent(out) .= P[0, 1]
        @test pb(NoRData())[3] == P(3)
    end
end

@testset "zero coefficients discard nonfinite array cotangents" begin
    for args in (
        (BLAS.syrk!, 'U', 'N', 0.0, fill(NaN, 1, 1), 1.0, zeros(1, 1)),
        (BLAS.trsm!, 'L', 'U', 'N', 'N', 0.0, fill(NaN, 1, 1), ones(1, 1)),
    )
        ds = map(Mooncake.zero_fcodual, args)
        out, pb = rrule!!(ds...)
        fill!(Mooncake.tangent(out), 1.0)
        pb(NoRData())
        @test iszero(only(Mooncake.tangent(ds[first(args) === BLAS.syrk! ? 5 : 7])))
    end
end

@testset "zero triangular coefficient keeps its direction" begin
    f = a -> only(BLAS.trsm!('L', 'U', 'N', 'N', a, fill(a + 2, 1, 1), ones(1, 1)))
    cache = Mooncake.prepare_hvp_cache(f, 0.0)
    @test Mooncake.value_and_hvp!!(cache, f, 1.0, 0.0) == (0.0, 0.5, -0.5)
end
