@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))

    # Regression: a Float16 WRAPPED `_kron!` input has a struct-lift V and would route to the
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

    # Regression: the `_kron!` `@is_primitive` declarations must be per-mode. The
    # `BlasFloat` widening exists only for the forward wrapper frule; if it leaks into reverse (as it
    # did when declared with the two-arg, both-modes `@is_primitive`) then complex `_kron!` becomes a
    # reverse primitive while the reverse rrule is real-only → complex reverse `MethodError`. Likewise
    # narrowing reverse to `BlasFloat` dropped wrapped-Float16's reverse primitive status that `main`
    # had. Reverse is now `IEEEFloat` (no complex, with Float16); forward is dense-`IEEEFloat` +
    # wrapped-`BlasFloat`.
    @testset "_kron! is_primitive is per-mode (complex derived, Float16 reverse kept)" begin
        W = Base.get_world_counter()
        ksig(T) = Tuple{typeof(LinearAlgebra._kron!),Matrix{T},Matrix{T},Matrix{T}}
        # Complex: forward primitive (wrapped-BlasFloat frule), reverse derived (no complex rrule).
        @test Mooncake.is_primitive(DefaultCtx, Mooncake.ForwardMode, ksig(ComplexF64), W)
        @test !Mooncake.is_primitive(DefaultCtx, Mooncake.ReverseMode, ksig(ComplexF64), W)
        # Float16: primitive in both modes (dense forward frule; real reverse rrule via arrayify).
        @test Mooncake.is_primitive(DefaultCtx, Mooncake.ForwardMode, ksig(Float16), W)
        @test Mooncake.is_primitive(DefaultCtx, Mooncake.ReverseMode, ksig(Float16), W)

        # Complex reverse-mode kron must run (via derived mode), not `MethodError`.
        fc(A, B) = sum(abs2, kron(A, B))
        Ac = ComplexF64[1 2; 3 4]
        Bc = ComplexF64[0.5 0; 0 2]
        cache = Mooncake.prepare_gradient_cache(fc, Ac, Bc)
        v, g = Mooncake.value_and_gradient!!(cache, fc, Ac, Bc)
        @test v ≈ sum(abs2, kron(Ac, Bc))
        @test any(!iszero, g[2])
    end
end
