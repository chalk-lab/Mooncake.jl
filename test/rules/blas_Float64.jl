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
            @testset "forward view: $(typeof(x)), width $N" for x in xs, N in (1, 8)
                x isa SubArray || continue
                _, parts = Mooncake.arrayify(Mooncake.zero_lifted(Val(N), x))
                @test all(p -> p isa StridedArray, parts)
            end
        end

        # Check wrapper type and write-through aliasing, which numerical rule tests do not pin.
        @testset "forward _arrayify_lane: $W" for W in (
            UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular
        )
            A = randn(StableRNG(1), 3, 3)
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

    @testset "nrm2 large tangent" for P in (Float32, Float64, ComplexF32, ComplexF64)
        d = floatmax(real(P))
        out = Mooncake.frule!!(
            Mooncake.zero_dual(BLAS.nrm2),
            Mooncake.zero_dual(1),
            Mooncake.lift(ones(P, 1), P[d]),
            Mooncake.zero_dual(1),
        )
        @test Mooncake.tangent(out, 1) == d
    end

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

    @testset "vector accumulation follows scalar frules" begin
        function broadcast_product!(C, X, Y, a, add)
            if add
                C .+= a .* X .* Y'
            else
                C .= a .* X .* Y'
            end
            return C
        end
        for P in (Float32, Float64, ComplexF32, ComplexF64)
            big = P <: Union{Float32,ComplexF32} ? 1.0f30 : 1e200
            rule = Mooncake.build_frule(
                broadcast_product!, zeros(P, 1, 1), zeros(P, 1), zeros(P, 1), zero(P), true
            )
            for add in (false, true),
                (a, da, x, dx, y, dy) in (
                    (1, big, 1, -big, big, 0),
                    (0, inv(big), inv(big), 0, big, 0),
                    (0, big, big, 1, inv(big), 2),
                    (0, 1, NaN, 0, 1, 0),
                    (0, 0, Inf, 0, 1, 0),
                )

                ds = (
                    Mooncake.lift(zeros(P, 1, 1), ones(P, 1, 1)),
                    Mooncake.lift(P[x], P[dx]),
                    Mooncake.lift(P[y], P[dy]),
                    Mooncake.lift(P(a), P(da)),
                )
                expected = rule(
                    Mooncake.zero_dual(broadcast_product!),
                    deepcopy(ds)...,
                    Mooncake.zero_dual(add),
                )
                actual = Mooncake.frule!!(
                    Mooncake.zero_dual(Mooncake._rvs_muladd!),
                    ds...,
                    Mooncake.zero_dual('N'),
                    Mooncake.zero_dual('C'),
                    Mooncake.zero_dual(add),
                    Mooncake.zero_dual(true),
                )
                @test isequal(Mooncake.tangent(actual, 1), Mooncake.tangent(expected, 1))
                iszero(a) && @test iszero(Mooncake.primal(actual))
            end
        end
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))

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

    # Empty gemv skips β scaling; uninitialised dot lanes can be denormal garbage
    # that passes finite differences. Pin exact zero with bespoke assertions.
    @testset "empty dot gives exactly-zero partials: width $Nw" for Nw in (1, 2, 3)
        o = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(Nw), dot),
            Mooncake.zero_lifted(Val(Nw), Float64[]),
            Mooncake.zero_lifted(Val(Nw), Float64[]),
        )
        @test primal(o) === 0.0
        @test all(k -> tangent(o, k) === 0.0, 1:Nw)
    end

    grad(f, a) = Mooncake.value_and_gradient!!(
        Mooncake.prepare_gradient_cache(f, a), f, a
    )[2][2]
    # Strong zeros permit NaN in unreferenced operands. Finite differences cannot
    # express these inputs, so check the primal exactly against its BLAS semantics.
    @testset "BLAS strong zeros with a NaN operand" begin
        A = randn(StableRNG(3), 3, 3)
        B = randn(StableRNG(4), 3, 3)
        Asym = (A + A') / 2
        nan3 = fill(NaN, 3, 3)

        args = (BLAS.scal!, 1, 0.0, [1.0], 1)
        ds = map(Mooncake.zero_fcodual, args)
        fill!(Mooncake.tangent(ds[4]), NaN)
        _, pb = Mooncake.rrule!!(ds...)
        pb(NoRData())
        @test iszero(only(Mooncake.tangent(ds[4])))

        @testset "zero coefficients with NaN cotangents" for P in (
            Float32, Float64, ComplexF32, ComplexF64
        )
            local A, x, y, C = ones(P, 1, 1), ones(P, 1), zeros(P, 1), zeros(P, 1, 1)
            cases = [
                ((BLAS.axpy!, 1, zero(P), x, 1, y, 1), 4),
                ((BLAS.gemv!, 'N', one(P), A, x, zero(P), y), 7),
                ((BLAS.symv!, 'U', one(P), A, x, zero(P), y), 7),
                ((BLAS.gemm!, 'N', 'N', one(P), A, copy(A), zero(P), C), 8),
                ((BLAS.symm!, 'L', 'U', one(P), A, copy(A), zero(P), C), 8),
                ((BLAS.syrk!, 'U', 'N', one(P), A, zero(P), C), 7),
            ]
            if P <: Complex
                append!(
                    cases,
                    [
                        ((BLAS.hemv!, 'U', one(P), A, x, zero(P), y), 7),
                        ((BLAS.hemm!, 'L', 'U', one(P), A, copy(A), zero(P), C), 8),
                        ((BLAS.herk!, 'U', 'N', one(real(P)), A, zero(real(P)), C), 7),
                    ],
                )
            end
            for (args, i) in cases
                ds = map(Mooncake.zero_fcodual, deepcopy(args))
                out, pb = Mooncake.rrule!!(ds...)
                fill!(Mooncake.tangent(out), P(NaN))
                pb(NoRData())
                @test all(iszero, Mooncake.tangent(ds[i]))
            end
        end

        @testset "zero output cotangents with nonfinite operands" for P in (
                Float32, Float64, ComplexF32, ComplexF64
            ),
            bad in (NaN, Inf)

            local A, N = P[2 1; 0 3], fill(P(bad), 2, 2)
            local x, nx, y, C = ones(P, 2), fill(P(bad), 2), zeros(P, 2), zeros(P, 2, 2)
            cases = Any[
                (BLAS.scal!, 2, P(bad), x, 1),
                (BLAS.axpy!, 2, P(bad), x, 1, y, 1),
                (BLAS.nrm2, 2, nx, 1),
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
                bad_arg in 1:4

                push!(
                    cases,
                    (
                        f,
                        flags...,
                        bad_arg == 3 ? P(bad) : one(P),
                        bad_arg == 1 ? N : A,
                        vector ? (bad_arg == 2 ? nx : x) : (bad_arg == 2 ? N : copy(A)),
                        bad_arg == 4 ? P(bad) : zero(P),
                        vector ? y : C,
                    ),
                )
            end
            for f in (BLAS.trmv!, BLAS.trsv!), lhs in (false, true)
                push!(cases, (f, 'U', 'N', 'N', lhs ? N : A, lhs ? x : nx))
            end
            for f in (BLAS.trmm!, BLAS.trsm!), bad_arg in 1:3
                push!(
                    cases,
                    (
                        f,
                        'L',
                        'U',
                        'N',
                        'N',
                        bad_arg == 3 ? P(bad) : one(P),
                        bad_arg == 1 ? N : A,
                        bad_arg == 2 ? N : copy(A),
                    ),
                )
            end
            for f in (P <: Real ? (BLAS.dot, dot) : (BLAS.dotc, BLAS.dotu))
                push!(cases, (f, nx, x))
            end
            for f in (BLAS.syrk!, (P <: Complex ? (BLAS.herk!,) : ())...), bad_arg in 1:3
                R = f === BLAS.herk! ? real(P) : P
                push!(
                    cases,
                    (
                        f,
                        'U',
                        'N',
                        bad_arg == 2 ? R(bad) : one(R),
                        bad_arg == 1 ? N : A,
                        bad_arg == 3 ? R(bad) : zero(R),
                        C,
                    ),
                )
            end
            for args in cases
                ds = map(Mooncake.zero_fcodual, deepcopy(args))
                rule = if first(args) in (BLAS.dot, BLAS.dotc, BLAS.dotu)
                    build_rrule(args...)
                else
                    rrule!!
                end
                out, pb = rule(ds...)
                scalar = primal(out) isa Number
                scalar || fill!(Mooncake.tangent(out), zero(P))
                r = pb(scalar ? zero(primal(out)) : NoRData())
                @testset "$(first(args))" begin
                    @test all(
                        i ->
                            !(args[i] isa AbstractArray) ||
                            all(iszero, Mooncake.tangent(ds[i])),
                        eachindex(args),
                    ) && all(v -> !(v isa Number) || iszero(v), r)
                end
            end
        end

        @testset "zero alpha array cotangents" begin
            local A, N = ones(1, 1), fill(NaN, 1, 1)
            C, Z = ones(ComplexF64, 1, 1), fill(ComplexF64(NaN), 1, 1)
            M, bad = ones(2, 2), fill(NaN, 2, 2)
            for (args, i) in (
                ((BLAS.gemv!, 'N', 0.0, A, [NaN], 1.0, [0.0]), 4),
                ((BLAS.symv!, 'U', 0.0, A, [NaN], 1.0, [0.0]), 4),
                ((BLAS.hemv!, 'U', 0.0im, C, ComplexF64[NaN], 1.0 + 0im, ComplexF64[0]), 4),
                ((BLAS.gemm!, 'N', 'N', 0.0, M, bad, 1.0, zeros(2, 2)), 5),
                ((BLAS.gemm!, 'N', 'N', 0.0, bad, M, 1.0, zeros(2, 2)), 6),
                ((BLAS.syrk!, 'U', 'N', 0.0, N, 1.0, zeros(1, 1)), 5),
                ((BLAS.herk!, 'U', 'N', 0.0, Z, 1.0, zeros(ComplexF64, 1, 1)), 5),
                ((BLAS.trmm!, 'L', 'U', 'N', 'N', 0.0, A, N), 7),
                ((BLAS.trsm!, 'L', 'U', 'N', 'N', 0.0, N, A), 7),
            )
                ds = map(Mooncake.zero_fcodual, deepcopy(args))
                out, pb = Mooncake.rrule!!(ds...)
                fill!(Mooncake.tangent(out), 1)
                pb(Mooncake.NoRData())
                @testset "$(first(args)) operand $i" begin
                    @test all(iszero, Mooncake.tangent(ds[i]))
                end
            end
        end

        @testset "finite extreme array cotangents" begin
            for (a, lhs, rhs, seed) in
                ((1e-300, 1e-200, 1e200, 1e200), (1e300, 1e-100, 1e-200, 1e-200)),
                P in (Float64, ComplexF64),
                op in (
                    BLAS.gemv!,
                    BLAS.symv!,
                    BLAS.symm!,
                    (P <: Complex ? (BLAS.hemv!, BLAS.hemm!) : ())...,
                )

                args = if op in (BLAS.gemv!, BLAS.symv!, BLAS.hemv!)
                    (
                        op,
                        op === BLAS.gemv! ? 'N' : 'U',
                        P(a),
                        fill(P(lhs), 1, 1),
                        P[rhs],
                        zero(P),
                        zeros(P, 1),
                    )
                else
                    (
                        op,
                        'L',
                        'U',
                        P(a),
                        fill(P(lhs), 1, 1),
                        fill(P(rhs), 1, 1),
                        zero(P),
                        zeros(P, 1, 1),
                    )
                end
                ds = map(Mooncake.zero_fcodual, args)
                out, pb = Mooncake.rrule!!(ds...)
                fill!(Mooncake.tangent(out), P(seed))
                pb(Mooncake.NoRData())
                i = op in (BLAS.gemv!, BLAS.symv!, BLAS.hemv!) ? 4 : 5
                expected = if i == 4
                    (a * seed) * rhs
                else
                    tmp = only(P(a)' * fill(P(seed), 1, 1) * fill(P(rhs), 1, 1)')
                    op === BLAS.hemm! ? tmp + tmp' - real(tmp) : tmp + tmp - tmp
                end
                # Ignore backend-dependent zero signs, keeping all other values exact.
                @test isequal(only(Mooncake.tangent(ds[i])) + zero(P), expected + zero(P))
            end
        end

        # Nonfinite inputs and overflowing differences need direct cotangent checks.
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

        @testset "fast coefficient cotangent extremes" for P in (Float64, ComplexF64),
            op in (
                BLAS.gemm!,
                BLAS.symm!,
                BLAS.symv!,
                (P <: Complex ? (BLAS.hemm!, BLAS.hemv!) : ())...,
            ),
            (a, b, seed, expected) in
            ((1e200, 1e100, 1e100, Inf), (1e-200, 1e-100, 1e-100, 0.0))

            vector = op in (BLAS.symv!, BLAS.hemv!)
            flags = if vector
                ('U',)
            elseif op === BLAS.gemm!
                ('N', 'N')
            else
                ('L', 'U')
            end
            dims = vector ? (1,) : (1, 1)
            args = (
                op,
                flags...,
                one(P),
                fill(P(a), 1, 1),
                fill(P(b), dims),
                zero(P),
                zeros(P, dims),
            )
            out, pb = Mooncake.rrule!!(map(Mooncake.zero_fcodual, args)...)
            fill!(Mooncake.tangent(out), P(seed))
            @test isequal(pb(Mooncake.NoRData())[length(flags) + 2], P(expected))
        end

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

        @testset "matrix reference extremes" for (op, P) in (
                (BLAS.symm!, Float64), (BLAS.hemm!, ComplexF64)
            ),
            side in "LR", n in (1, 2, 3, 16),
            (a, b, c) in (
                (1e200, 1e-200, 1e200),
                (1e-200, 1e200, 1e-200),
                (1e200, 1e200, 1e-200),
                (1e-200, 1e-200, 1e200),
            )

            rhs, seed = fill(P(b), n, n), fill(P(c), n, n)
            args = (op, side, 'U', P(a), ones(P, n, n), rhs, zero(P), zeros(P, n, n))
            ds = map(Mooncake.zero_fcodual, args)
            out, pb = Mooncake.rrule!!(ds...)
            Mooncake.tangent(out) .= seed
            pb(Mooncake.NoRData())
            expected = side == 'L' ? P(a)' * seed * rhs' : P(a)' * rhs' * seed
            projected = Matrix(
                transpose(LowerTriangular(expected)) + UpperTriangular(expected)
            )
            projected[diagind(projected)] .-= diag(expected)
            # BLAS and the projection can differ in zero signs (including imaginary parts).
            @test isequal(Mooncake.tangent(ds[5]) .+ zero(P), projected .+ zero(P))
        end

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

        # Ignore NaN outside the selected output. The finite-difference check
        # perturbs every entry (including NaN), so these need bespoke checks.
        @testset "beta gradient ignores a NaN in an unused entry" begin
            xx = randn(StableRNG(5), 3)
            ynan = [NaN, 1.0, 2.0]
            Cnan = [NaN 1.0 2.0; 3.0 4.0 5.0; 6.0 7.0 8.0]
            @test grad(b -> (z=copy(ynan); BLAS.gemv!('N', 1.0, A, xx, b, z); z[2]), 2.0) ==
                ynan[2]
            @test grad(
                b -> (z=copy(ynan); BLAS.symv!('U', 1.0, Asym, xx, b, z); z[2]), 2.0
            ) == ynan[2]
            @test grad(
                b -> (Z=copy(Cnan); BLAS.gemm!('N', 'N', 1.0, A, B, b, Z); Z[2, 2]), 2.0
            ) == Cnan[2, 2]
            @test grad(
                b -> (Z=copy(Cnan); BLAS.syrk!('U', 'N', 1.0, A, b, Z); Z[2, 2]), 2.0
            ) == Cnan[2, 2]
        end

        # NaN outside the selected output must not poison the scalar α gradient.
        @testset "alpha gradient ignores a NaN outside the selected output" begin
            Mn = [NaN 0.0 0.0; 1.0 2.0 3.0; 4.0 5.0 6.0]
            v3 = randn(StableRNG(6), 3)
            @test grad(
                a -> (Z=zeros(3, 3); BLAS.gemm!('N', 'N', a, Mn, B, 1.0, Z); Z[2, 2]), 2.0
            ) ≈ (Mn * B)[2, 2]
            @test grad(a -> (z=zeros(3); BLAS.gemv!('N', a, Mn, v3, 1.0, z); z[2]), 2.0) ≈
                (Mn * v3)[2]
            @test grad(
                a -> (Z=zeros(3, 3); BLAS.syrk!('U', 'N', a, Mn, 1.0, Z); Z[2, 2]), 2.0
            ) ≈ (Mn * Mn')[2, 2]
            @test grad(a -> (z=[NaN, 2.0, 3.0]; BLAS.scal!(3, a, z, 1); z[2]), 2.0) == 2.0
            for P in (Float32, Float64, ComplexF32, ComplexF64)
                @test grad(
                    a -> (y=P[NaN, 2]; BLAS.axpy!(2, a, P[NaN, 3], 1, y, 1); real(y[2])),
                    P(2),
                ) == P(3)
            end
        end

        # Keep operands local to avoid sharing a captured global's tangent between differentiations.
        @testset "trmm!/trsm! alpha gradient ignores a NaN in an unused column" begin
            At = [2.0 1.0 1.0; 0.0 3.0 1.0; 0.0 0.0 4.0]
            Bt = [1.0 NaN 2.0; 3.0 NaN 4.0; 5.0 NaN 6.0]
            @test grad(
                a -> (C=copy(Bt); BLAS.trmm!('L', 'U', 'N', 'N', a, At, C); C[1, 1]), 2.0
            ) ≈ (At * Bt)[1, 1]
            @test grad(
                a -> (C=copy(Bt); BLAS.trsm!('L', 'U', 'N', 'N', a, At, C); C[1, 1]), 2.0
            ) ≈ (At \ Bt)[1, 1]
        end

        # Seeded dα needs the solve, but the α == 0 primal must still be zero
        # even when the solve reads NaN from A.
        @testset "trsm! α=0, seeded dα=$seeded: width $Nw" for seeded in (false, true),
            Nw in (1, 2, 3)

            α = if seeded
                Mooncake.randn_lifted(Val(Nw), StableRNG(9), 0.0)
            else
                Mooncake.zero_lifted(Val(Nw), 0.0)
            end
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
            if !seeded
                @test all(k -> all(iszero, tangent(r, k)), 1:Nw)
            end
        end
    end

    # At β=0, C may be uninitialised/NaN; the dβ*C term must mask NaN entries.
    @testset "syrk! dβ*C NaN-C guard at β=0" begin
        A = randn(StableRNG(1), 3, 2)
        for C in (fill(NaN, 3, 3), randn(StableRNG(2), 3, 3))
            r = Mooncake.frule!!(
                Mooncake.zero_lifted(Val(1), BLAS.syrk!),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.lift('N', Mooncake.NoTangent()),
                Mooncake.lift(1.0, 0.0),
                Mooncake.lift(A, zero(A)),
                Mooncake.lift(0.0, 1.0),
                Mooncake.lift(copy(C), zeros(3, 3)),
            )
            d = tangent(r)
            if isnan(C[1, 1])
                @test !any(isnan, [d[i, j].partials[1] for i in 1:3 for j in i:3])
            else
                @test d[1, 2].partials[1] ≈ C[1, 2]
            end
        end
    end
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end

@testset "gemm! reproduces its own primal at the alpha/beta zeros" begin
    # Builds may multiply or skip A at α=0: compare with the running BLAS.
    # The registry's finite-difference harness cannot handle NaN operands.
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

# The registry cannot replicate pinned array seeds or mix active and inactive lanes.
@testset "BLAS inactive lanes" begin
    for P in (Float64, Float32, ComplexF64, ComplexF32),
        bad in (P(NaN), P(Inf)), a in (zero(P), one(P)), which in 1:3,
        coefficients in (false, true)

        A = fill(which == 1 ? bad : P(2), 3, 3)
        B = fill(which == 2 ? bad : P(2), 3, 3)
        C = fill(which == 3 ? bad : P(2), 3, 3)
        cases = Any[
            (BLAS.gemm!, 'N', 'N', a, A, B, zero(P), C),
            (BLAS.gemv!, 'N', a, A, B[:, 1], zero(P), C[:, 1]),
            (BLAS.symm!, 'L', 'U', a, A, B, zero(P), C),
            (BLAS.symv!, 'U', a, A, B[:, 1], zero(P), C[:, 1]),
            (BLAS.syrk!, 'U', 'N', a, A, zero(P), C),
        ]
        if P <: Complex
            append!(
                cases,
                [
                    (BLAS.hemm!, 'L', 'U', a, A, B, zero(P), C),
                    (BLAS.hemv!, 'U', a, A, B[:, 1], zero(P), C[:, 1]),
                    (BLAS.herk!, 'U', 'N', real(a), A, zero(real(P)), C),
                ],
            )
        end
        for f in (BLAS.trmm!, BLAS.trsm!), side in ('L', 'R'), diag in ('N', 'U')
            push!(cases, (f, side, 'U', 'N', diag, a, A, B))
        end
        for f in (BLAS.trmv!, BLAS.trsv!), diag in ('N', 'U')
            push!(cases, (f, 'U', 'N', diag, A, B[:, 1]))
        end
        for args in cases
            expected = first(args)(deepcopy(args[2:end])...)
            slots = map(deepcopy(args)) do x
                d = Mooncake.zero_lifted(Val(8), x)
                if x isa AbstractArray
                    if !coefficients || which == 3
                        lane = coefficients ? 2 : 1
                        fill!(Mooncake.tangent_view(d, lane), one(eltype(x)))
                    end
                elseif x isa Union{AbstractFloat,Complex}
                    ds = ntuple(k -> k == 1 && coefficients ? one(x) : zero(x), 8)
                    d = Mooncake.Lifted{typeof(x),8}(x, Mooncake._scalar_ndual(x, ds))
                end
                d
            end
            reference = if coefficients && which == 3
                ds = map(d -> Mooncake.lift(deepcopy(primal(d)), tangent(d, 2)), slots)
                tangent(Mooncake.frule!!(ds...), 1)
            end
            out = Mooncake.frule!!(slots...)
            @test isequal(primal(out), expected)
            first_inactive = reference === nothing ? 2 : 3
            @test all(
                k -> isequal(Mooncake.tangent(out, k), zero(expected)), first_inactive:8
            )
            reference === nothing || @test isequal(tangent(out, 2), reference)
        end
    end
end
