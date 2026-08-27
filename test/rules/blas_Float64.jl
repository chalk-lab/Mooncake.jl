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

        # Forward per-lane `_arrayify_lane` must cover the same triangular surface as the reverse
        # `arrayify(::AbstractTriangular)` above. Regression: the unit-triangular variants had no
        # forward method and `MethodError`d (the four share `.data` + a `Tx(data)` constructor, so
        # one `AbstractTriangular` method covers them, mirroring reverse).
        @testset "forward _arrayify_lane: $W" for W in (
            UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular
        )
            x = W(randn(StableRNG(1), 3, 3))
            for N in (1, 2)
                _x, parts = Mooncake.arrayify(Mooncake.zero_lifted(Val(N), x))
                @test _x === x
                @test length(parts) == N
                @test all(p -> p isa W, parts)  # lane partials reconstruct the same wrapper
            end
        end
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))

    # Regression: gemm!'s frule!!/rrule!! only cover a matrix C, so the @is_primitive C slot
    # must be AbstractMatrix (not AbstractVecOrMat) to stay in lockstep with the rule methods — a
    # vector-C gemm! must NOT be declared primitive (else a MethodError instead of a clean fallback).
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

    # Regression: an empty `dot` must give EXACTLY zero lane partials. `gemv` returns early on an
    # empty operand without applying `beta`, so the frule's output buffer used to keep whatever the
    # allocator handed back. The registered `n = 0` case cannot pin this: the garbage is typically
    # denormal (~1e-310), so it passes a finite-difference comparison against zero. Only an exact
    # check catches it, hence a bespoke assertion rather than a registry entry.
    @testset "empty dot gives exactly-zero partials: width $Nw" for Nw in (1, 2, 3)
        o = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(Nw), dot),
            Mooncake.zero_lifted(Val(Nw), Float64[]),
            Mooncake.zero_lifted(Val(Nw), Float64[]),
        )
        @test primal(o) === 0.0
        @test all(k -> tangent(o, k) === 0.0, 1:Nw)
    end

    # Regression: BLAS's strong zeros. `β == 0` leaves the destination unreferenced and `α == 0`
    # leaves `A`/`B` unreferenced, so either may legally hold NaN. Recomputing the primal as
    # `α*tmp + β*C` (reverse) or solving unconditionally then scaling by `α` (forward `trsm!`) made
    # `0*NaN` a NaN RESULT, not merely a NaN derivative. A finite-difference comparison cannot
    # express a NaN operand, so these are bespoke exact checks.
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

        # The beta GRADIENT, not just the primal: `dβ = Σ conj(Cᵢ)·dCᵢ` contracted the whole of
        # `C`, so one `NaN` in an entry the selected output does not depend on poisoned it, while
        # forward returned the right number. Not a registry case: `test_rule`'s finite-difference
        # oracle perturbs every entry, the NaN one included, so the oracle itself returns NaN.
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

        # The alpha gradient has the same shape: a `NaN` confined to row 1 of a factor leaves the
        # selected output finite, but `dα` contracted the whole product and returned `NaN`.
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

        # The early return above requires EVERY dα lane to be zero. A seeded dα takes the solve
        # path, which legitimately reads `A` for the derivative — but the primal at `α == 0` is
        # still BLAS's zero fill, and rebuilding it as `α*X` turned the NaN in `A` into a NaN
        # PRIMAL. Same reason as above for it being a bespoke check.
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

    # Regression: the syrk!/herk! frule's `dβ*C` term must mask NaN input-C elements (the β==0
    # convention lets the caller pass an uninitialised/NaN C, overwritten by the primal), matching the
    # sibling level-3 frules. Unguarded `dβ .* triu(C)` leaked NaN into the tangent.
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
    # OpenBLAS multiplies even at `alpha == 0`, though the reference spec permits skipping `A`,
    # so a rule emulating the skip returns a different value than the routine it differentiates.
    # Compared against `BLAS.gemm!` on the RUNNING build rather than a hardcoded NaN, since a
    # build that does skip is equally valid. Not a `test_rule` case: the harness is not NaN-safe
    # -- four of its checks fail on NaN operands whatever the rule does.
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
    # `beta == 0` keeps BLAS's strong zero the other way: a NaN in a `C` BLAS never reads must
    # not leak in through `0 * NaN`.
    # DISTINCT `A` and `B`: one object at both positions is a repeated mutable argument, which a
    # prepared cache refuses, so the assertion would measure that refusal instead of the `beta`
    # semantics it is here for.
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
