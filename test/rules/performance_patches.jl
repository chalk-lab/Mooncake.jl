@testset "performance_patches" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))

    # Wrapped Float16 must stay derived because arrayify supports only BlasFloat.
    # Loose tolerances accommodate Float16 finite differences.
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

    # Primitive coverage differs by mode: complex is forward-only; real Float16
    # remains primitive in reverse, including wrapped inputs.
    @testset "_kron! is_primitive is per-mode (complex derived, Float16 reverse kept)" begin
        W = Base.get_world_counter()
        ksig(T) = Tuple{typeof(LinearAlgebra._kron!),Matrix{T},Matrix{T},Matrix{T}}
        for T in (ComplexF64, Float16), mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            @test Mooncake.is_primitive(DefaultCtx, mode, ksig(T), W) ==
                (T === Float16 || mode === Mooncake.ForwardMode)
        end

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
