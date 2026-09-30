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

    @testset "nrm2 large tangent" for P in (Float32, Float64, ComplexF32, ComplexF64)
        d = floatmax(real(P))
        out = Mooncake.frule!!(
            Mooncake.zero_dual(BLAS.nrm2),
            Mooncake.zero_dual(1),
            Mooncake.Dual(ones(P, 1), P[d]),
            Mooncake.zero_dual(1),
        )
        @test Mooncake.tangent(out) == d
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
                    Mooncake.Dual(zeros(P, 1, 1), ones(P, 1, 1)),
                    Mooncake.Dual(P[x], P[dx]),
                    Mooncake.Dual(P[y], P[dy]),
                    Mooncake.Dual(P(a), P(da)),
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
                @test isequal(Mooncake.tangent(actual), Mooncake.tangent(expected))
                iszero(a) && @test iszero(Mooncake.primal(actual))
            end
        end
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end

@testset "reverse strong zeros" begin
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

        for P in (Float32, Float64, ComplexF32, ComplexF64)
            dX = P[NaN]
            _, pb = Mooncake.rrule!!(
                Mooncake.zero_fcodual(BLAS.scal!),
                Mooncake.zero_fcodual(1),
                Mooncake.zero_fcodual(zero(P)),
                CoDual(ones(P, 1), dX),
                Mooncake.zero_fcodual(1),
            )
            pb(NoRData())
            @test iszero(only(dX))
        end

        @testset "zero alpha array cotangents" begin
            for P in (Float32, Float64, ComplexF32, ComplexF64), bad in (NaN, Inf)
                ops=(
                    BLAS.gemv!,
                    BLAS.symv!,
                    BLAS.gemm!,
                    BLAS.symm!,
                    BLAS.syrk!,
                    BLAS.trmm!,
                    BLAS.trsm!,
                    (P<:Complex ? (BLAS.hemv!, BLAS.hemm!, BLAS.herk!) : ())...,
                )
                for op in ops,
                    badarg in (1, 2),
                    flags in
                    (op === BLAS.gemm! ? Iterators.product("NTC", "NTC") : (('L', 'U'),))

                    op in (BLAS.syrk!, BLAS.herk!) && badarg==2 && continue
                    lhs=fill(P(badarg==1 ? bad : 1), 3, 3);
                    rhs=fill(P(badarg==2 ? bad : 1), 3, 3)
                    args = if op in (BLAS.gemv!, BLAS.symv!, BLAS.hemv!)
                        (
                            op,
                            op===BLAS.gemv! ? 'N' : 'U',
                            zero(P),
                            lhs,
                            rhs[:, 1],
                            one(P),
                            zeros(P, 3),
                        )
                    elseif op in (BLAS.gemm!, BLAS.symm!, BLAS.hemm!)
                        (op, flags..., zero(P), lhs, rhs, one(P), zeros(P, 3, 3))
                    elseif op in (BLAS.syrk!, BLAS.herk!)
                        Q=op===BLAS.herk! ? real(P) : P
                        (op, 'U', 'N', zero(Q), lhs, one(Q), zeros(P, 3, 3))
                    else
                        (op, 'L', 'U', 'N', 'N', zero(P), lhs, rhs)
                    end
                    ds=map(Mooncake.zero_fcodual, args);
                    out, pb=Mooncake.rrule!!(ds...)
                    fill!(Mooncake.tangent(out), one(P));
                    pb(Mooncake.NoRData())
                    inds = if op in (BLAS.gemv!, BLAS.symv!, BLAS.hemv!)
                        (4, 5)
                    elseif op in (BLAS.trmm!, BLAS.trsm!)
                        (7, 8)
                    elseif op in (BLAS.syrk!, BLAS.herk!)
                        (5,)
                    else
                        (5, 6)
                    end
                    @testset "$P $op $flags $bad $badarg" for i in inds
                        @test all(iszero, Mooncake.tangent(ds[i]))
                    end
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
                @test isequal(only(Mooncake.tangent(ds[i])), expected)
            end
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
            @test isequal(Mooncake.tangent(ds[6]), expected)
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
            @test isequal(Mooncake.tangent(ds[5]), projected)
        end

        # α != 1 reaches the recomputation instead of the α==1 && β==0 fast path.
        for (f, flags, M) in ((BLAS.gemm!, ('N', 'N'), A), (BLAS.symm!, ('L', 'U'), Asym))
            args = (f, flags..., 2.0, copy(M), copy(B), 0.0, copy(nan3))
            o = Mooncake.rrule!!(map(Mooncake.zero_fcodual, args)...)[1]
            @test primal(o) ≈ 2.0 * M * B
        end

        # `α == 0`: A unreferenced, so a NaN there must not reach the result or the partials.
        o = Mooncake.rrule!!(
            Mooncake.zero_fcodual(BLAS.symm!),
            Mooncake.zero_fcodual('L'),
            Mooncake.zero_fcodual('U'),
            Mooncake.zero_fcodual(0.0),
            Mooncake.zero_fcodual(copy(nan3)),
            Mooncake.zero_fcodual(copy(B)),
            Mooncake.zero_fcodual(1.0),
            Mooncake.zero_fcodual(zeros(3, 3)),
        )[1]
        @test all(iszero, primal(o))

        # Ignore NaN outside the selected output. The finite-difference oracle
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
    end
end
