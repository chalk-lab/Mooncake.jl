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

    # Strong zeros permit NaN in unreferenced operands. Finite differences cannot
    # express these inputs, so check the primal exactly against its BLAS semantics.
    @testset "BLAS strong zeros with a NaN operand" begin
        A = randn(StableRNG(3), 3, 3)
        B = randn(StableRNG(4), 3, 3)
        Asym = (A + A') / 2
        nan3 = fill(NaN, 3, 3)

        # `α != 1` skips the (α==1 && β==0) fast path, reaching the recomputation.
        o = Mooncake.rrule!!(
            Mooncake.zero_fcodual(BLAS.gemm!),
            Mooncake.zero_fcodual('N'),
            Mooncake.zero_fcodual('N'),
            Mooncake.zero_fcodual(2.0),
            Mooncake.zero_fcodual(copy(A)),
            Mooncake.zero_fcodual(copy(B)),
            Mooncake.zero_fcodual(0.0),
            Mooncake.zero_fcodual(copy(nan3)),
        )[1]
        @test primal(o) ≈ 2.0 * A * B

        o = Mooncake.rrule!!(
            Mooncake.zero_fcodual(BLAS.symm!),
            Mooncake.zero_fcodual('L'),
            Mooncake.zero_fcodual('U'),
            Mooncake.zero_fcodual(2.0),
            Mooncake.zero_fcodual(copy(Asym)),
            Mooncake.zero_fcodual(copy(B)),
            Mooncake.zero_fcodual(0.0),
            Mooncake.zero_fcodual(copy(nan3)),
        )[1]
        @test primal(o) ≈ 2.0 * Asym * B

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
            grad(f, b) = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(f, b), f, b
            )[2][2]
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
            grad(f, a) = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(f, a), f, a
            )[2][2]
            @test grad(
                a -> (Z=zeros(3, 3); BLAS.gemm!('N', 'N', a, Mn, B, 1.0, Z); Z[2, 2]), 2.0
            ) ≈ (Mn * B)[2, 2]
            @test grad(a -> (z=zeros(3); BLAS.gemv!('N', a, Mn, v3, 1.0, z); z[2]), 2.0) ≈
                (Mn * v3)[2]
            @test grad(
                a -> (Z=zeros(3, 3); BLAS.syrk!('U', 'N', a, Mn, 1.0, Z); Z[2, 2]), 2.0
            ) ≈ (Mn * Mn')[2, 2]
            @test grad(a -> (z=[NaN, 2.0, 3.0]; BLAS.scal!(3, a, z, 1); z[2]), 2.0) == 2.0
        end

        # Cover NaN in unused columns for triangular α gradients. Keep operands
        # local to avoid sharing a captured global's tangent between differentiations.
        @testset "trmm!/trsm! alpha gradient ignores a NaN in an unused column" begin
            At = [2.0 1.0 1.0; 0.0 3.0 1.0; 0.0 0.0 4.0]
            Bt = [1.0 NaN 2.0; 3.0 NaN 4.0; 5.0 NaN 6.0]
            gr(f, a) = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(f, a), f, a
            )[2][2]
            @test gr(
                a -> (C=copy(Bt); BLAS.trmm!('L', 'U', 'N', 'N', a, At, C); C[1, 1]), 2.0
            ) ≈ (At * Bt)[1, 1]
            @test isfinite(
                gr(
                    a -> (C=copy(Bt); BLAS.trsm!('L', 'U', 'N', 'N', a, At, C); C[1, 1]),
                    2.0,
                ),
            )
        end

        @testset "trsm! α=0 ignores a NaN A: width $Nw" for Nw in (1, 2, 3)
            r = Mooncake.frule!!(
                Mooncake.zero_lifted(Val(Nw), BLAS.trsm!),
                Mooncake.lift('L', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.lift('N', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.zero_lifted(Val(Nw), 0.0),
                Mooncake.zero_lifted(Val(Nw), copy(nan3)),
                Mooncake.zero_lifted(Val(Nw), copy(B)),
            )
            @test all(iszero, primal(r))
            @test all(k -> all(iszero, tangent(r, k)), 1:Nw)
        end

        # Seeded dα needs the solve, but the α == 0 primal must still be zero
        # even when the solve reads NaN from A.
        @testset "trsm! α=0 with a seeded dα: width $Nw" for Nw in (1, 2, 3)
            r = Mooncake.frule!!(
                Mooncake.zero_lifted(Val(Nw), BLAS.trsm!),
                Mooncake.lift('L', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.lift('N', Mooncake.NoTangent()),
                Mooncake.lift('U', Mooncake.NoTangent()),
                Mooncake.randn_lifted(Val(Nw), StableRNG(9), 0.0),
                Mooncake.zero_lifted(Val(Nw), copy(nan3)),
                Mooncake.zero_lifted(Val(Nw), copy(B)),
            )
            @test all(iszero, primal(r))
        end
    end

    # At β=0, C may be uninitialised/NaN; the dβ*C term must mask NaN entries.
    @testset "syrk! dβ*C NaN-C guard at β=0" begin
        A = randn(StableRNG(1), 3, 2)
        # NaN input C, β=0, dβ=1: the output tangent's upper triangle must be NaN-free.
        rN = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(1), BLAS.syrk!),
            Mooncake.lift('U', Mooncake.NoTangent()),
            Mooncake.lift('N', Mooncake.NoTangent()),
            Mooncake.lift(1.0, 0.0),
            Mooncake.lift(A, zero(A)),
            Mooncake.lift(0.0, 1.0),
            Mooncake.lift(fill(NaN, 3, 3), zeros(3, 3)),
        )
        dN = tangent(rN)
        @test !any(isnan, [dN[i, j].partials[1] for i in 1:3 for j in i:3])
        # Finite C: the dβ=1 term contributes exactly C on the upper triangle.
        C = randn(StableRNG(2), 3, 3)
        rF = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(1), BLAS.syrk!),
            Mooncake.lift('U', Mooncake.NoTangent()),
            Mooncake.lift('N', Mooncake.NoTangent()),
            Mooncake.lift(1.0, 0.0),
            Mooncake.lift(A, zero(A)),
            Mooncake.lift(0.0, 1.0),
            Mooncake.lift(copy(C), zeros(3, 3)),
        )
        dF = tangent(rF)
        @test dF[1, 2].partials[1] ≈ C[1, 2]
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
