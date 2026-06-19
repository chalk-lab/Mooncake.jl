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
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end
