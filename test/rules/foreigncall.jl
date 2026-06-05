@testset "foreigncall" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:foreigncall))

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
        L(T, N, v) = Mooncake.Lifted{T,N}(v, Mooncake.NoDual())
        xL(N, parts) = Mooncake.Lifted{Float64,N}(
            2.0, Mooncake.Nfwd.NDual{Float64,N}(2.0, parts)
        )
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
            iv = Mooncake.tangent(r)
            @test iv.value == 2.0^3                                    # V.value === primal result
            @test all(iv.partials .≈ ntuple(k -> 12.0 * parts[k], N))  # d/dx x^3 = 3x^2 = 12
        end
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
