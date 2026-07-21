struct Reshape166Pair
    a::Float64
    b::Float64
end

@testset "foreigncall" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:foreigncall))

    # Regression: reshape of an array whose elements are differentiable NON-numeric types has
    # an element-wise `Array{VE}` forward V (`Array{ImmutableDual}` for a struct, `Array{Tuple{NDual,
    # …}}` for a tuple) — matching neither the numeric `NDualArray` frule nor the old `Array{NoDual}`
    # one, so forward `reshape` errored while reverse (element-type-generic) worked. Covered now by
    # the general `Array{VE}` reshape frule.
    @testset "reshape of differentiable non-numeric element arrays" begin
        g_struct(x) = (
            v=[Reshape166Pair(x, 2x), Reshape166Pair(3x, 4x)];
            r=reshape(v, 2, 1);
            r[1, 1].a + r[2, 1].b
        )
        g_tuple(x) = (v=[(x, 2x), (3x, 4x)]; r=reshape(v, 2, 1); r[1, 1][1] + r[2, 1][2])
        TestUtils.test_rule(
            StableRNG(1), g_struct, 1.0; is_primitive=false, mode=Mooncake.ForwardMode
        )
        TestUtils.test_rule(
            StableRNG(1), g_tuple, 1.0; is_primitive=false, mode=Mooncake.ForwardMode
        )
    end

    # Regression: the jl_get_world_counter/jl_matching_methods frule returns
    # `zero_lifted(Val(Nw), y)` so the forward V is CANONICAL. jl_matching_methods returns a
    # `Vector{Any}` (tangent_type is `Vector{Any}`, NOT `NoTangent`), so hardcoding `NoDual` was a
    # non-canonical V; the world counter (`UInt`) legitimately duals to `NoDual`. This asserts the
    # shape the rule's `zero_lifted` guarantees for both (the registered `Base._methods_by_ftype`
    # case above exercises the rule end-to-end but can't distinguish `NoDual` from the zero
    # `Vector{Any}` dual, since both yield a zero derivative).
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

    # Regression: the llvm.powi frule must set the inner NDual's `.value` to the primal result
    # `y` and scale only the partials. A naive `grad * tangent(x)` scaled `.value` to `grad*x`,
    # silently breaking the V.value === primal invariant — latent, since width-1 `test_rule`
    # checks only the outer primal and the partials, never the inner NDual value.
    @testset "llvm.powi forward NDual.value coherence" begin
        fc = Mooncake._foreigncall_
        nm = Symbol("llvm.powi.f64.i32")
        L(T, N, v) = Lifted{T,N}(v, Mooncake.NoDual())
        xL(N, parts) = Lifted{Float64,N}(2.0, Mooncake.Nfwd.NDual{Float64,N}(2.0, parts))
        @testset "width $N" for N in (1, 2, 3)
            parts = ntuple(k -> Float64(k), N)
            r = Mooncake.frule!!(
                L(typeof(fc), N, fc),
                L(Val{nm}, N, Val(nm)),
                L(Val{Float64}, N, Val(Float64)),
                L(Tuple{Val{Float64},Val{Int32}}, N, (Val(Float64), Val(Int32))),
                L(Val{0}, N, Val(0)),
                L(Val{:llvmcall}, N, Val(:llvmcall)),
                xL(N, parts),
                L(Int32, N, Int32(3)),
                L(Int32, N, Int32(3)),
                xL(N, parts),
            )
            iv = tangent(r)
            @test iv.value == 2.0^3                                    # V.value === primal result
            @test all(iv.partials .≈ ntuple(k -> 12.0 * parts[k], N))  # d/dx x^3 = 3x^2 = 12
        end
    end

    # Regression: the chunked llvm.powi frule must scale partials with `_fwd_guarded_scale`, so
    # an inactive (zero-seed) lane stays exactly 0.0 even where `grad` is ±Inf (x=0, negative
    # exponent). Unguarded `_fwd_scale` gave `0 * Inf = NaN`. Mirrors the `pow_fast` guard.
    @testset "llvm.powi inactive-lane guard at x=0 negative exponent" begin
        fc = Mooncake._foreigncall_
        nm = Symbol("llvm.powi.f64.i32")
        L(T, N, v) = Lifted{T,N}(v, Mooncake.NoDual())
        xL(N, x, parts) = Lifted{Float64,N}(x, Mooncake.Nfwd.NDual{Float64,N}(x, parts))
        N = 2
        r = Mooncake.frule!!(
            L(typeof(fc), N, fc),
            L(Val{nm}, N, Val(nm)),
            L(Val{Float64}, N, Val(Float64)),
            L(Tuple{Val{Float64},Val{Int32}}, N, (Val(Float64), Val(Int32))),
            L(Val{0}, N, Val(0)),
            L(Val{:llvmcall}, N, Val(:llvmcall)),
            xL(N, 0.0, (1.0, 0.0)),
            L(Int32, N, Int32(-2)),
            L(Int32, N, Int32(-2)),
            xL(N, 0.0, (1.0, 0.0)),
        )
        # Lane 2 (zero seed) must be exactly 0.0, not NaN, despite grad = ±Inf at the pole.
        @test tangent(r).partials[2] == 0.0
    end

    # Regression: the REVERSE llvm.powi pullback must apply the same zero-cotangent guard, so a
    # zero incoming cotangent yields an exact 0 even where grad is ±Inf (x=0, negative exponent).
    # Unguarded `grad * dy` gave `Inf * 0 = NaN`.
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

    # Regression: the deepcopy frule must copy the whole slot in ONE deepcopy walk. Copying
    # primal and V separately severs `NDualArray.primal === primal(slot)`, so the copy's inner
    # `.value` reads a stale third array after the copied primal is mutated.
    @testset "deepcopy preserves slot-internal aliasing (width $N)" for N in (1, 2, 3)
        x = Mooncake.randn_lifted(Val(N), Xoshiro(123), [1.0, 2.0])
        y = Mooncake.frule!!(Mooncake.zero_lifted(Val(N), deepcopy), x)
        @test tangent(y).primal === primal(y)
        primal(y)[1] = 99.0
        @test tangent(y)[1].value == 99.0
    end

    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d,
        :jl_alloc_array_2d,
        :jl_alloc_array_3d,
        :jl_new_array,
        :jl_array_copy,
        :jl_type_intersection,
        :memset,
        :jl_get_tls_world_age,
        :memmove,
        :jl_object_id,
        :jl_array_sizehint,
        :jl_array_grow_beg,
        :jl_array_grow_end,
        :jl_array_grow_at,
        :jl_array_del_beg,
        :jl_array_del_end,
        :jl_array_del_at,
        :jl_value_ptr,
        :jl_threadid,
        :memhash_seed,
        :memhash32_seed,
        :jl_get_field_offset,
    ]
        @test_throws(
            ErrorException,
            Mooncake.frule!!(zero_dual(Mooncake._foreigncall_), zero_dual(Val(name))),
        )
        @test_throws(
            ErrorException,
            Mooncake.rrule!!(zero_codual(Mooncake._foreigncall_), zero_codual(Val(name))),
        )
    end
end
