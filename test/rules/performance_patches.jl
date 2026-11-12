@testset "mul! lane independence and rejected dimensions" for N in (1, 2, 8)
    for bad in (Inf, NaN), op in (identity, transpose, adjoint)
        C = Mooncake.zero_lifted(Val(N), zeros(2, 2))
        A = Mooncake.zero_lifted(Val(N), ones(2, 2))
        B = Mooncake.zero_lifted(Val(N), op([bad 1.0; 1.0 1.0]))
        N > 1 && fill!(Mooncake.Nfwd.tangent_view(A, 1), 1.0)
        Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), mul!),
            C,
            A,
            B,
            Mooncake.zero_lifted(Val(N), true),
            Mooncake.zero_lifted(Val(N), false),
        )
        @test iszero(Mooncake.Nfwd.tangent_view(C, N))
    end
    C = Mooncake.zero_lifted(Val(N), fill(NaN, 2))
    A = Mooncake.zero_lifted(Val(N), fill(Inf, 2, 2))
    B = Mooncake.zero_lifted(Val(N), ones(2))
    for k in 1:N
        fill!(Mooncake.Nfwd.tangent_view(C, k), NaN)
    end
    N > 1 && fill!(Mooncake.Nfwd.tangent_view(B, 1), 1.0)
    Mooncake.frule!!(
        Mooncake.zero_lifted(Val(N), mul!),
        C,
        A,
        B,
        Mooncake.zero_lifted(Val(N), 1.0),
        Mooncake.zero_lifted(Val(N), 0.0),
    )
    @test iszero(Mooncake.Nfwd.tangent_view(C, N))
    for bad in (Inf, NaN)
        C = Mooncake.zero_lifted(Val(N), ones(2, 2))
        A = Mooncake.zero_lifted(Val(N), fill(bad, 2, 2))
        B = Mooncake.zero_lifted(Val(N), ones(2, 2))
        Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), mul!),
            C,
            A,
            B,
            Mooncake.zero_lifted(Val(N), 0.0),
            Mooncake.zero_lifted(Val(N), 3.0),
        )
        @test primal(C) == fill(3.0, 2, 2)
        @test iszero(Mooncake.Nfwd.tangent_view(C, N))
    end
    C = Mooncake.zero_lifted(Val(N), ones(1, 2))
    A = Mooncake.zero_lifted(Val(N), ones(1, 2))
    B = Mooncake.zero_lifted(Val(N), ones(1, 2))
    for k in 1:N
        fill!(Mooncake.Nfwd.tangent_view(C, k), 1.0)
        fill!(Mooncake.Nfwd.tangent_view(B, k), 1.0)
    end
    @test_throws DimensionMismatch Mooncake.frule!!(
        Mooncake.zero_lifted(Val(N), mul!),
        C,
        A,
        B,
        Mooncake.zero_lifted(Val(N), 2.0),
        Mooncake.zero_lifted(Val(N), 3.0),
    )
    @test primal(C) == ones(1, 2)
    @test all(k -> Mooncake.Nfwd.tangent_view(C, k) == ones(1, 2), 1:N)
end

@testset "mul! forward block allocations" for N in (1, 8), wrap in (identity, transpose)
    C, A, B = [ones(64, 64) for _ in 1:3]
    f = (C, A, B) -> mul!(C, wrap(A), B)
    rule = build_frule(f, C, A, B; chunk_size=N)
    slots = (
        Mooncake.zero_lifted(Val(N), f),
        map((C, A, B)) do x
            slot = Mooncake.zero_lifted(Val(N), x)
            for k in 1:N
                fill!(Mooncake.Nfwd.tangent_view(slot, k), 1.0)
            end
            slot
        end...,
    )
    sink = Ref(0.0)
    consume() =
        (y=rule(slots...); sink[]=primal(y)[1] + Mooncake.Nfwd.tangent_view(y, 1)[1])
    consume()
    @test (@allocated consume()) <= sizeof(C) + 1024
    @test sink[] == 192.0
end

@testset "performance_patches" begin
    @test !Mooncake.may_match_primitive(
        MinimalCtx, ForwardMode, Tuple{typeof(mul!),Any,Any,Any,Int,Int}
    )
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
        for k in (kron, (a, b) -> LinearAlgebra._kron!(similar(a, 1, 4), a, b))
            f =
                x -> begin
                    a = k(Union{Float32,Float64}[x 2.0f0], Union{Float32,Float64}[3.0 x])
                    a[1] + a[2] + a[3] + a[4]
                end
            TestUtils.test_rule(StableRNG(123), f, 3.0; is_primitive=false)
        end
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

    # Wrapped Float16 `_kron!` is not a forward primitive, so it must take the derived rule.
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
