@testset "performance_patches" begin
    @testset "direct primitive refuses union storage" begin
        T = Union{Float32,Float64}
        a = T[2.0f0 3.0]
        for args in
            ((permutedims, a), (kron, a, a), (LinearAlgebra._kron!, kron(a, a), a, a)),
            w in (1, 8)

            slots = map(x -> Mooncake.zero_lifted(Val(w), x), args)
            @test_throws "This forward primitive requires" Mooncake.frule!!(slots...)
        end
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:performance_patches))

    @testset "real-union arrays constructed inside a function" begin
        for op in (
            sum,
            x -> sum(abs2, x),
            maximum,
            minimum,
            x -> sum(view(x, :)),
            x -> sum(sort(x)),
            x -> sum(permutedims(reshape(x, 1, :))),
            x -> sum(kron(x, x)),
        )
            f = x -> op(Union{Float32,Float64}[x, 2.0f0])
            TestUtils.test_rule(
                StableRNG(123), f, 3.0; is_primitive=false, mode=Mooncake.ForwardMode
            )
        end
    end

    @testset "real-union matrix kron" begin
        f =
            x -> begin
                a = kron(Union{Float32,Float64}[x 2.0f0], Union{Float32,Float64}[3.0 x])
                a[1] + a[2] + a[3] + a[4]
            end
        TestUtils.test_rule(
            StableRNG(123), f, 3.0; is_primitive=false, mode=Mooncake.ForwardMode
        )
    end

    @testset "mixed-union kron" begin
        T = Union{Float32,ComplexF64}
        for wrap in (identity, transpose)
            a = wrap(T[2.0f0 3.0 + 1.0im])
            b = wrap(T[1.0f0 2.0 + 1.0im])
            TestUtils.test_rule(
                StableRNG(123), kron, a, b; is_primitive=false, mode=Mooncake.ForwardMode
            )
            TestUtils.test_rule(
                StableRNG(123),
                LinearAlgebra._kron!,
                kron(a, b),
                a,
                b;
                is_primitive=false,
                mode=Mooncake.ForwardMode,
            )
        end
    end

    @testset "nested unit-triangular reads" begin
        for P in (Float64, Float32),
            W in (UnitUpperTriangular, UnitLowerTriangular),
            wrap in (x -> view(x, :, :), x -> reshape(x, 1, 4), Symmetric)

            x, y = wrap(W(zeros(P, 2, 2))), ones(P, 2, 2)
            for args in ((kron, x, y), (LinearAlgebra._kron!, zero(kron(x, y)), x, y))
                TestUtils.test_rule(
                    StableRNG(123),
                    args...;
                    mode=wrap === Symmetric ? Mooncake.ForwardMode : nothing,
                    is_primitive=true,
                )
            end
        end
    end

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
