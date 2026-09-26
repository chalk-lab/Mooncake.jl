@testset "foreigncall" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:foreigncall))

    # Zero derivatives alone cannot distinguish canonical V from an incorrect NoDual
    # for matching-methods' Vector{Any}. Check shape explicitly alongside the registry.
    @testset "world-counter / matching-methods canonical V (width $N)" for N in (1, 2)
        w = Base.get_world_counter()
        yw = w                                                   # jl_get_world_counter → UInt
        ym = Base._methods_by_ftype(Tuple{typeof(sin),Float64}, -1, w)  # jl_matching_methods → Vector{Any}
        # world counter: genuinely non-differentiable → NoDual is the canonical V.
        @test Mooncake.dual_type(Val(N), typeof(yw)) === Mooncake.NoDual
        @test tangent(Mooncake.zero_lifted(Val(N), yw)) isa Mooncake.NoDual
        # matching methods: canonical V is Vector{Any}, NOT NoDual (the bug hardcoded NoDual).
        @test Mooncake.dual_type(Val(N), typeof(ym)) !== Mooncake.NoDual
        rm = Mooncake.zero_lifted(Val(N), ym)  # exactly what the fixed frule returns
        @test primal(rm) === ym
        @test tangent(rm) isa Mooncake.dual_type(Val(N), typeof(ym))
        @test !(tangent(rm) isa Mooncake.NoDual)
    end

    @testset "llvm powi via fastmath lowering" begin
        fn(x) = @fastmath x^2
        cache = prepare_gradient_cache(fn, 3.0)
        val, grad = value_and_gradient!!(cache, fn, 3.0)
        @test val == 9.0
        @test grad[2] == 6.0

        g(x) = Base.FastMath.pow_fast(x, Int32(3))
        cache_g = prepare_gradient_cache(g, 2.0)
        val_g, grad_g = value_and_gradient!!(cache_g, g, 2.0)
        @test val_g == 8.0
        @test grad_g[2] == 12.0
    end

    # Check inner value coherence and inactive lanes at infinite gradients explicitly.
    @testset "llvm.powi forward (x=$x, n=$n, width $N)" for (N, x, n) in (
        (1, 2.0, Int32(3)), (2, 2.0, Int32(3)), (3, 2.0, Int32(3)), (2, 0.0, Int32(-2))
    )
        fc = Mooncake._foreigncall_
        nm = Symbol("llvm.powi.f64.i32")
        L(T, N, v) = Lifted{T,N}(v, Mooncake.NoDual())
        xL(N, x, parts) = Lifted{Float64,N}(x, Mooncake.Nfwd.NDual{Float64,N}(x, parts))
        parts = n == 3 ? ntuple(k -> Float64(k), N) : (1.0, 0.0)
        r = Mooncake.frule!!(
            L(typeof(fc), N, fc),
            L(Val{nm}, N, Val(nm)),
            L(Val{Float64}, N, Val(Float64)),
            L(Tuple{Val{Float64},Val{Int32}}, N, (Val(Float64), Val(Int32))),
            L(Val{0}, N, Val(0)),
            L(Val{:llvmcall}, N, Val(:llvmcall)),
            xL(N, x, parts),
            L(Int32, N, n),
            L(Int32, N, n),
            xL(N, x, parts),
        )
        iv = tangent(r)
        if n == 3
            @test iv.value == 2.0^3
            @test all(iv.partials .≈ ntuple(k -> 12.0 * parts[k], N))
        else
            @test iv.partials[2] == 0.0
        end
    end

    # Zero cotangents must stay zero at infinite gradients.
    # The synthetic llvmcall argument list is rejected by the registry's interface check;
    # ordinary derived x^n cases cannot reach this pole.
    @testset "llvm.powi reverse zero-cotangent guard at x=0 negative exponent" begin
        fc = Mooncake._foreigncall_
        nm = Symbol("llvm.powi.f64.i32")
        zc = Mooncake.zero_codual
        args = (
            zc(fc),
            zc(Val(nm)),
            zc(Val(Float64)),
            zc((Val(Float64), Val(Int32))),
            zc(Val(0)),
            zc(Val(:llvmcall)),
        )
        # x=0, exponent=-2 → grad = ±Inf; a zero cotangent must give dx = 0.0, not NaN.
        _, pb = Mooncake.rrule!!(args..., zc(0.0), zc(Int32(-2)), zc(Int32(-2)), zc(0.0))
        @test pb(0.0)[6] === 0.0
        # A nonzero cotangent still propagates the analytic gradient (x=2, exp=3 → 3x² = 12).
        _, pb2 = Mooncake.rrule!!(args..., zc(2.0), zc(Int32(3)), zc(Int32(3)), zc(2.0))
        @test pb2(1.0)[6] ≈ 12.0
    end

    # A shared deepcopy walk must preserve slot-internal aliasing after primal mutation.
    @testset "deepcopy preserves slot-internal aliasing (width $N)" for N in (1, 2, 3)
        x = Mooncake.randn_lifted(Val(N), Xoshiro(123), [1.0, 2.0])
        y = Mooncake.frule!!(Mooncake.zero_lifted(Val(N), deepcopy), x)
        @test tangent(y).primal === primal(y)
        primal(y)[1] = 99.0
        @test tangent(y)[1].value == 99.0
    end
end
