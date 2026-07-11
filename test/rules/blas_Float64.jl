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

    # Regression: nrm2 at the zero vector has a removable singularity (`s / (2y)` is `0/0`);
    # every lane's partial must be zero, not NaN.
    @testset "nrm2 zero-vector lanes (width $N)" for N in (1, 2, 3)
        Xz = Mooncake.randn_lifted(Val(N), Xoshiro(1), zeros(3))
        r = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), BLAS.nrm2),
            Mooncake.zero_lifted(Val(N), 3),
            Xz,
            Mooncake.zero_lifted(Val(N), 1),
        )
        @test all(iszero, tangent(r).partials)
    end

    # Regression (#200): the syrk!/herk! frule's `dβ*C` term must mask NaN input-C elements (the β==0
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

    # Regression (#201/#202): the trmm!/trsm! reverse rules computed ∇α = dot(B,dB)/α' after the primal
    # overwrote B; at α==0 that is 0/0 = NaN, though the true gradient is finite. ∇α must be finite and
    # correct at α==0 (and unchanged for α≠0). f is linear in α, so the gradient is constant.
    @testset "trmm!/trsm! ∇α finite at α=0" begin
        A = [1.0 2.0; 0.0 3.0]
        B = [1.0 4.0; 2.0 5.0]
        ftrmm(a, A, B) = sum(BLAS.trmm!('L', 'U', 'N', 'N', a, A, copy(B)))
        ftrsm(a, A, B) = sum(BLAS.trsm!('L', 'U', 'N', 'N', a, A, copy(B)))
        for f in (ftrmm, ftrsm)
            _, g0 = value_and_gradient!!(prepare_gradient_cache(f, 0.0, A, B), f, 0.0, A, B)
            _, g2 = value_and_gradient!!(prepare_gradient_cache(f, 2.0, A, B), f, 2.0, A, B)
            @test isfinite(g0[2])            # ∇α at α=0 is finite (was NaN)
            @test g0[2] ≈ g2[2]              # linear in α → constant gradient, matches α≠0
        end
    end
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end
