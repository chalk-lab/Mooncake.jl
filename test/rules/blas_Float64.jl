@testset "blas (basic)" begin
    @testset "axpby! validates before writes" for P in (
            Float64, Float32, ComplexF64, ComplexF32
        ),
        N in (1, 8)

        x = Mooncake.zero_lifted(Val(N), ones(P, 3))
        y = Mooncake.zero_lifted(Val(N), view(ones(P, 3), StepRangeLen(1, 0, 3)))
        foreach(dx -> fill!(dx, one(P)), Mooncake.arrayify(x)[2])
        @test_throws ArgumentError Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), BLAS.axpby!),
            Mooncake.zero_lifted(Val(N), P(2)),
            x,
            Mooncake.zero_lifted(Val(N), P(3)),
            y,
        )
        @test Mooncake.primal(y) == ones(P, 3)
        @test all(iszero, Mooncake.arrayify(y)[2])
    end

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
            @testset "forward view: $(typeof(x)), width $N" for x in xs, N in (1, 8)
                x isa SubArray || continue
                _, parts = Mooncake.arrayify(Mooncake.zero_lifted(Val(N), x))
                @test all(p -> p isa StridedArray, parts)
            end
        end

        # Check wrapper type and write-through aliasing, which numerical rule tests do not pin.
        @testset "forward _arrayify_lane: $P, $W" for P in (Float16, Float64),
            W in
            (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)

            A = randn(StableRNG(1), P, 3, 3)
            for x in (W(A), W(view(A, 1:2, 1:2))), N in (1, 2, 8)
                slot = Mooncake.zero_lifted(Val(N), x)
                _x, parts = Mooncake.arrayify(slot)
                @test _x === x
                @test length(parts) == N
                @test all(p -> p isa W, parts)  # lane partials reconstruct the same wrapper
                i, j = W in (UpperTriangular, UnitUpperTriangular) ? (1, 2) : (2, 1)
                parts[1][i, j] = 1
                _, parts2 = Mooncake.arrayify(slot)
                @test parts2[1][i, j] == 1
                @test all(p -> iszero(p[i, j]), parts2[2:end])
                fill!(parent(parts[1]), 7)
                expected = Matrix(parts[1])
                W === UnitUpperTriangular && (expected = triu(expected, 1))
                W === UnitLowerTriangular && (expected = tril(expected, -1))
                _, reads = Mooncake.arrayify(slot, Val(:read))
                @test reads[1] == expected
                @test selectdim(first(Mooncake._partials_block(slot)), 1, 1) == expected
                @test all(==(7), parent(parts[1]))
                # The mask must also survive wrappers around a unit triangular.
                for wrap in (
                    identity,
                    x -> view(x, :, :),
                    vec,
                    transpose,
                    adjoint,
                    Symmetric,
                    Hermitian,
                    UpperTriangular,
                    LowerTriangular,
                )
                    wrapped = Mooncake.zero_lifted(Val(N), wrap(x))
                    _, reads = Mooncake.arrayify(wrapped, Val(:read))
                    @test all(iszero, reads)
                    @test iszero(first(Mooncake._partials_block(wrapped)))
                end
            end
        end
    end

    @testset "BLAS tangent storage bounds" begin
        for P in (Float32, Float64, ComplexF32, ComplexF64)
            buffer = ones(P, 256)
            GC.@preserve buffer begin
                p = unsafe_wrap(Array, pointer(buffer), 24)
                walks = (
                    (view(p, 20:-1:3), 10, 1),
                    (transpose(reshape(view(p, 20:-1:3), 3, 6)), 10, 1),
                    (p, 4, 10),
                )
                for (x, n, inc) in walks,
                    f in (P <: Real ? (BLAS.dot,) : (BLAS.dotc, BLAS.dotu))

                    args = (f, n, x, inc, ones(P, n), 1)
                    for N in (1, 8)
                        slots = map(z -> Mooncake.zero_lifted(Val(N), z), args)
                        @test_throws "tangent storage" Mooncake.frule!!(slots...)
                    end
                    # Entry must refuse before the pullback can write out of bounds.
                    rule = Mooncake.build_rrule(args...)
                    @test_throws "tangent storage" rule(map(Mooncake.zero_fcodual, args)...)
                end
            end
        end
    end

    @testset "wrapped low-level walks" for P in (Float64, ComplexF64),
        wrap in (P <: Real ? (transpose, adjoint) : (transpose,))

        x = wrap(view(collect(P, 1:8), 1:2:8))
        @test Mooncake._blas_walk_step(x, 2, 4) === 1
        for mode in (ForwardMode, ReverseMode)
            TestUtils.test_rule(
                StableRNG(123),
                BLAS.scal!,
                4,
                P(2),
                x,
                2;
                is_primitive=true,
                perf_flag=:none,
                mode=mode,
            )
        end
    end

    @testset "dot lane cost follows walk" for N in (1, 8), inc in (1, 2000)
        args = (BLAS.dot, 3, view(ones(1000, 1000), 2, 1:2:7), inc, ones(3), 1)
        slots = map(x -> Mooncake.zero_lifted(Val(N), x), args)
        Mooncake.frule!!(slots...)
        TestUtils.count_allocs(Mooncake.frule!!, slots...)
        @test TestUtils.count_allocs(Mooncake.frule!!, slots...) == 0
    end

    @testset "dot walks into parent" for P in (Float64, ComplexF64),
        f in (P <: Real ? (BLAS.dot,) : (BLAS.dotc, BLAS.dotu)),
        wrap in (
            A -> view(A, 2, 1:2:7),
            A -> view(A, 2:5, 3),
            A -> transpose(view(A, 2, 1:2:7)),
            A -> SubArray(view(A, 2, :), (1:2:7,)),
        ),
        inc in (-2, 2)

        g = (A, y) -> real(f(3, wrap(A), inc, y, 1))
        for mode in (ForwardMode, ReverseMode)
            TestUtils.test_rule(
                StableRNG(123),
                g,
                reshape(collect(P, 1:48), 6, 8),
                ones(P, 3);
                is_primitive=false,
                perf_flag=:none,
                mode=mode,
            )
        end
    end

    @testset "asum strong zeros" for P in (Float32, Float64, ComplexF32, ComplexF64),
        N in (1, 8)

        x = P[0, NaN, Inf]
        out = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), BLAS.asum), Mooncake.zero_lifted(Val(N), x)
        )
        @test all(k -> iszero(tangent(out, k)), 1:N)
        dx = Mooncake.zero_fcodual(x)
        _, pb = Mooncake.rrule!!(Mooncake.zero_fcodual(BLAS.asum), dx)
        pb(zero(real(P)))
        @test all(iszero, tangent(dx))
        # A zero component's subgradient must also mask a nonfinite direction.
        z = Mooncake.zero_lifted(Val(N), zeros(P, 1))
        fill!(first(Mooncake._partials_block(z)), P(Inf))
        out = Mooncake.frule!!(Mooncake.zero_lifted(Val(N), BLAS.asum), z)
        @test all(k -> iszero(tangent(out, k)), 1:N)
    end

    @testset "asum mixed Hessian at zero weight" begin
        f(z) = z[1] * BLAS.asum(view(z, 2:3))
        x, v = [0.0, 2.0, -3.0], [1.0, 0.0, 0.0]
        cache = Mooncake.prepare_hvp_cache(f, x)
        _, _, hv = Mooncake.value_and_hvp!!(cache, f, v, x)
        @test hv ≈ [0.0, 1.0, -1.0]
    end

    @testset "complex dot strong zeros" for f in (BLAS.dotc, BLAS.dotu),
        P in (ComplexF32, ComplexF64), N in (1, 8), wrap in (identity, x -> view(x, :)),
        zero_input in (false, true)

        x = wrap(zero_input ? zeros(P, 2) : P[NaN, Inf])
        ds = map(z -> Mooncake.zero_lifted(Val(N), z), (f, x, copy(x)))
        if zero_input
            for d in ds[2:3], k in 1:N
                fill!(Mooncake._blas_lane_partial(d, k), P(Inf))
            end
        end
        out = Mooncake.frule!!(ds...)
        @test all(k -> iszero(tangent(out, k)), 1:N)
        cds = map(Mooncake.zero_fcodual, (f, x, copy(x)))
        _, pb = Mooncake.rrule!!(cds...)
        pb(zero_input ? P(Inf) : zero(P))
        @test all(iszero, last(Mooncake.arrayify(cds[2])))
        @test all(iszero, last(Mooncake.arrayify(cds[3])))
    end

    # Keep this canonical test_rule loop to preserve its consumed RNG stream.
    @testset "guarded accumulation" for P in (Float32, Float64, ComplexF32, ComplexF64),
        a in (0, 2), coefficient_first in (false, true),
        add in (false, true)

        rng = StableRNG(12)
        X = coefficient_first ? randn(rng, P, 3) : randn(rng, P, 3, 3)
        Y = coefficient_first ? randn(rng, P, 3) : randn(rng, P, 3, 3)
        TestUtils.test_rule(
            rng,
            Mooncake._rvs_muladd!,
            randn(rng, P, 3, 3),
            X,
            Y,
            P(a),
            'N',
            'C',
            add,
            coefficient_first;
            mode=Mooncake.ForwardMode,
            perf_flag=:stability,
        )
    end

    # Keep this canonical test_rule loop to preserve its StableRNG(12) probes.
    @testset "guarded unit-triangular reads" for W in
                                                 (UnitUpperTriangular, UnitLowerTriangular),
        (x, y) in (
            (view(W(zeros(2, 2)), :, 1:1), view(W(zeros(2, 2)), 1:1, :)),
            (W(zeros(1, 1)), W(zeros(1, 1))),
        )

        TestUtils.test_rule(
            StableRNG(12),
            Mooncake._rvs_muladd!,
            zeros(size(x, 1), size(y, 2)),
            x,
            y,
            1.0,
            'N',
            'N',
            false,
            true;
            mode=Mooncake.ForwardMode,
            # Before 1.12, type-only analysis reports dispatch in a generic BLAS branch these inputs never take.
            perf_flag=VERSION >= v"1.12-" ? :stability : :none,
        )
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))

    # Selected positions: finite differences cannot run; the full cotangent exceeds the documented BLAS range limit.
    @testset "finite alpha cotangents" for P in (Float64, ComplexF64),
        op in (BLAS.gemv!, BLAS.symv!, (P <: Complex ? (BLAS.hemv!,) : ())...),
        flag in (op === BLAS.gemv! ? ('N', 'T', 'C') : ('U', 'L')),
        (a, x, dy, expected) in
        ((1e308, 2, 1e-308, 1.9999999999999998), (1.0, NaN, 0.0, 0.0))

        args = (op, flag, P(1e-308), fill(P(a), 1, 1), P[x], zero(P), zeros(P, 1))
        out, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, args)...)
        fill!(Mooncake.tangent(out), P(dy))
        @test pb(NoRData())[3] == P(expected)
    end

    # Selected positions: finite differences cannot run; the full cotangent exceeds the documented BLAS range limit.
    @testset "finite extreme gemm cotangents" for P in (Float64, ComplexF64),
        (a, b) in ((1e200, 1e-200), (1e-200, 1e200)), tA in "NTC", tB in "NTC",
        n in (1, 2, 3, 16)

        lhs, rhs = fill(P(a), n, n), fill(P(b), n, n)
        args = (BLAS.gemm!, tA, tB, P(a), lhs, rhs, zero(P), zeros(P, n, n))
        ds = map(Mooncake.zero_fcodual, args)
        out, pb = Mooncake.rrule!!(ds...)
        fill!(Mooncake.tangent(out), P(b))
        pb(Mooncake.NoRData())
        expected = BLAS.gemm('N', 'N', P(a), lhs, fill(P(b), n, n))
        @test isequal(Mooncake.tangent(ds[6]) .+ zero(P), expected .+ zero(P))
    end

    # Primitive C coverage must match the matrix-only rules; vector C needs fallback.
    @testset "gemm! is_primitive C-slot lockstep" begin
        w = Base.get_world_counter()
        gemm = typeof(BLAS.gemm!)
        vecC = Tuple{
            gemm,Char,Char,Float64,Matrix{Float64},Vector{Float64},Float64,Vector{Float64}
        }
        matC = Tuple{
            gemm,Char,Char,Float64,Matrix{Float64},Vector{Float64},Float64,Matrix{Float64}
        }
        for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            @test !Mooncake.is_primitive(Mooncake.MinimalCtx, mode, vecC, w)  # vector C: not primitive
            @test Mooncake.is_primitive(Mooncake.MinimalCtx, mode, matC, w)   # matrix C: primitive
        end
    end

    # Strong zeros permit NaN in unreferenced operands. Finite differences cannot
    # express these inputs, so check the primal exactly against its BLAS semantics.
    @testset "BLAS strong zeros with a NaN operand" begin
        A = randn(StableRNG(3), 3, 3)
        B = randn(StableRNG(4), 3, 3)
        Asym = (A + A') / 2
        nan3 = fill(NaN, 3, 3)

        # α != 1 reaches the recomputation instead of the α==1 && β==0 fast path.
        for (f, flags, M) in ((BLAS.gemm!, ('N', 'N'), A), (BLAS.symm!, ('L', 'U'), Asym))
            args = (f, flags..., 2.0, copy(M), copy(B), 0.0, copy(nan3))
            o = Mooncake.rrule!!(map(Mooncake.zero_fcodual, args)...)[1]
            @test primal(o) ≈ 2.0 * M * B
        end

        # `α == 0`: A unreferenced, so a NaN there must not reach the result or the partials.
        args = (BLAS.symm!, 'L', 'U', 0.0, copy(nan3), copy(B), 1.0, zeros(3, 3))
        o = Mooncake.rrule!!(map(Mooncake.zero_fcodual, args)...)[1]
        @test all(iszero, primal(o))

        # Test cases cannot check only the primal over nonfinite inputs without finite differences.
        @testset "trsm! α=0, seeded dα: width $Nw" for Nw in (1, 2, 3)
            α = Mooncake.randn_lifted(Val(Nw), StableRNG(9), 0.0)
            r = Mooncake.frule!!(
                Mooncake.zero_lifted(Val(Nw), BLAS.trsm!),
                Mooncake.lift('L', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.lift('N', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                α,
                Mooncake.zero_lifted(Val(Nw), copy(nan3)),
                Mooncake.zero_lifted(Val(Nw), copy(B)),
            )
            @test all(iszero, primal(r))
        end
    end
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end

@testset "gemm! reproduces its own primal at the alpha/beta zeros" begin
    # Builds may multiply or skip A at α=0: compare with the running BLAS.
    # This exercises the prepared-cache API, beyond the rule registry's contract.
    Anan = [NaN 0.0; 0.0 0.0]
    I2 = [1.0 0.0; 0.0 1.0]
    C0 = [1.0 2.0; 3.0 4.0]
    alpha_zero(C, A, B) = (BLAS.gemm!('N', 'N', 0.0, A, B, 1.0, C); sum(C))
    got = Mooncake.value_and_gradient!!(
        Mooncake.prepare_gradient_cache(alpha_zero, copy(C0), Anan, I2),
        alpha_zero,
        copy(C0),
        Anan,
        I2,
    )[1]
    @test isequal(got, alpha_zero(copy(C0), Anan, I2))
    # β=0 must ignore NaN in C. Keep A/B distinct to isolate β semantics from
    # repeated-mutable-argument handling in the prepared cache.
    Cnan = [NaN 0.0; 0.0 0.0]
    Aone = [1.0 0.0; 0.0 1.0]
    Bone = [1.0 0.0; 0.0 1.0]
    beta_zero(C, A, B) = (BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, C); sum(C))
    got_b = Mooncake.value_and_gradient!!(
        Mooncake.prepare_gradient_cache(beta_zero, copy(Cnan), Aone, Bone),
        beta_zero,
        copy(Cnan),
        Aone,
        Bone,
    )[1]
    @test isequal(got_b, beta_zero(copy(Cnan), Aone, Bone))
end
