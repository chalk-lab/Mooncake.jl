@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))

    # Regression (#169): a Float16 WRAPPED `_kron!` input has a struct-lift V and would route to the
    # `arrayify`-based wrapper-fallback frule, but `arrayify` supports only `BlasFloat` → a raw
    # MethodError. The `@is_primitive` now covers dense Float16 and wrapped `BlasFloat` only, so a
    # Float16 wrapped input is left non-primitive and handled by derived forward mode. Float16 finite
    # differences are imprecise, hence the loose tolerance — the point is that it runs (no crash) and
    # is roughly correct, not high-precision agreement.
    @testset "Float16 wrapped kron! is derived, not an arrayify crash" begin
        fk(A, B) = (
            C=Matrix{Float16}(undef, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2));
            LinearAlgebra.kron!(C, A, B);
            sum(C)
        )
        srng = StableRNG(169)
        A = LinearAlgebra.UpperTriangular(rand(srng, Float16, 2, 2))
        B = Matrix(rand(srng, Float16, 2, 2))
        TestUtils.test_rule(
            StableRNG(1),
            fk,
            A,
            B;
            is_primitive=false,
            mode=Mooncake.ForwardMode,
            atol=5e-2,
            rtol=5e-2,
        )
    end
end
