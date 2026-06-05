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

    # Regression: the `gemv!` frule must run the in-place primal `BLAS.gemv!` exactly ONCE,
    # after the per-lane loop — not once per lane. A per-lane primal nests the update and lets
    # lanes >1 read an already-overwritten `y` in the `dβ * y[n]` term, silently corrupting both
    # the primal and the β-derivative for chunk width Nw>1. Width-1 `test_rule` (all the rule
    # test cases use) cannot see this, so it is checked directly here at widths 1/2/3.
    @testset "gemv! chunked primal (Nw>1)" begin
        gv! = BLAS.gemv!
        A0, x0, y0 = randn(sr(1), 3, 2), randn(sr(2), 2), randn(sr(3), 3)
        α, β = 0.7, 1.3
        fL(N) = Mooncake.Lifted{typeof(gv!),N}(gv!, Mooncake.NoDual())
        tL(N) = Mooncake.Lifted{Char,N}('N', Mooncake.NoDual())
        sL(N, v, parts) = Mooncake.Lifted{Float64,N}(
            v, Mooncake.Nfwd.NDual{Float64,N}(v, parts)
        )
        aL(N, M) = Mooncake.Lifted{typeof(M),N}(M, Mooncake.zero_dual(Val(N), M))
        gemv_lanes(N, dα, dβ) = begin
            A, x, y = copy(A0), copy(x0), copy(y0)
            r = Mooncake.frule!!(
                fL(N), tL(N), sL(N, α, dα), aL(N, A), aL(N, x), sL(N, β, dβ), aL(N, y)
            )
            (y, [collect(Mooncake.tangent(r, k)) for k in 1:N])
        end
        # Primal must equal a single update at every width.
        @testset "primal width $N" for N in (1, 2, 3)
            y, _ = gemv_lanes(N, ntuple(_ -> 0.0, N), ntuple(_ -> 0.0, N))
            @test y ≈ α * A0 * x0 + β * y0
        end
        # Per-lane derivatives at width 2 must match independent width-1 runs (nonzero dβ
        # exercises the previously-corrupted `dβ * y[n]` lane).
        _, w2 = gemv_lanes(2, (0.5, -0.3), (0.9, 0.2))
        @test w2[1] ≈ gemv_lanes(1, (0.5,), (0.9,))[2][1]
        @test w2[2] ≈ gemv_lanes(1, (-0.3,), (0.2,))[2][1]
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_basic))
end

@testset "blas (Float64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_Float64))
end
