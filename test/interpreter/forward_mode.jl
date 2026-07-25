function foo(x)
    y = 0.0
    try
        if x > 0
            error("")
        end
        y = x
    catch
        y = 2x
    end
    return y
end

# Helpers for the world-advance rule-staleness regression test (issue #1218;
# see `_build_rule!` in forward_mode.jl for the scope caveat). Defining `stale_fwd_inner(::Float64)`
# later tightens `stale_fwd_callee`'s return type Float32->Float64 via a mid-pass world advance.
# `stale_fwd_lazy` reaches the callee statically (LazyFRule), `stale_fwd_dyn` dynamically (DynamicFRule).
stale_fwd_inner(x) = Float32(x) * 2.0f0
@noinline stale_fwd_callee(x) = stale_fwd_inner(x)
stale_fwd_lazy(x) = stale_fwd_callee(x)
const STALE_FWD_FNS = Function[stale_fwd_callee]
stale_fwd_dyn(x) = (STALE_FWD_FNS[1])(x)

# Dynamic dispatch (`inferencebarrier` hides the callee) so the derived rule captures a
# `DynamicFRule` with a mutable `cache` Dict — used by the cache-hit `_copy` regression below.
fwd_cache_dyn(x) = Base.inferencebarrier(sin)(x)::Float64 + x

@testset "s2s_forward_mode_ad" begin
    test_cases = collect(enumerate(TestResources.generate_test_functions()))
    @testset "$n - $(_typeof((fx)))" for (n, (int_only, pf, opts, fx...)) in test_cases
        @info "$n: $(_typeof(fx))"
        rng = Xoshiro(123546)
        mode = ForwardMode
        skip_chunked = TestUtils._case_skip_chunked(opts)
        fwd_allocs_broken = TestUtils._case_fwd_allocs_broken(opts)
        TestUtils.test_rule(
            rng,
            fx...;
            perf_flag=pf,
            interface_only=int_only,
            is_primitive=false,
            mode,
            skip_chunked,
            fwd_allocs_broken,
        )
    end

    @testset "integration testing for invalid global ref errors" begin
        @static if VERSION > v"1.12-"
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                Mooncake.build_frule(Mooncake.TestResources.non_const_global_ref, 5.0)
            )
        end
    end

    # Try try-catch statements.
    @testset "try-catch" begin
        rng = StableRNG(123)
        perf_flag = :none
        interface_only = false
        is_primitive = false
        mode = ForwardMode
        TestUtils.test_rule(rng, foo, 5.0; perf_flag, interface_only, is_primitive, mode)
    end

    @testset "capture in ReturnNode regression test" begin
        struct RegTestStruct
            x::Vector{Float64}
            RegTestStruct() = new()
        end
        f(x) = RegTestStruct()
        TestUtils.test_rule(
            StableRNG(123), f, 1.0; perf_flag=:none, is_primitive=false, mode=ForwardMode
        )
    end

    # Without the fix the lazy path throws a `convert` MethodError in _build_rule! after the
    # world advance; both lazy and dynamic must return the build-world result (Float32), not
    # the post-advance world's (Float64).
    @testset "stale rule build-world after world advance (issue #1218)" begin
        lazy = Mooncake.build_frule(stale_fwd_lazy, 1.5)
        dyn = Mooncake.build_frule(stale_fwd_dyn, 1.5)
        @eval stale_fwd_inner(x::Float64) = x * 2.0  # advance world; tightens callee's type
        lazy_out = Base.invokelatest(
            lazy, Mooncake.zero_lifted(Val(1), stale_fwd_lazy), Mooncake.lift(1.5, 1.0)
        )
        dyn_out = Base.invokelatest(
            dyn, Mooncake.zero_lifted(Val(1), stale_fwd_dyn), Mooncake.lift(1.5, 1.0)
        )
        @test Mooncake.primal(lazy_out) === 3.0f0
        @test Mooncake.primal(dyn_out) === 3.0f0
    end

    # A cache hit must return an independent copy (as reverse `build_derived_rrule` does),
    # not the shared cached object: otherwise two builds share one `DynamicFRule.cache`
    # Dict and race under threads / nested AD.
    @testset "cache-hit returns an independent rule copy" begin
        interp = Mooncake.MooncakeInterpreter(ForwardMode)
        sig = Tuple{typeof(fwd_cache_dyn),Float64}
        r1 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)
        r2 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)  # cache HIT
        dyns1 = filter(c -> c isa Mooncake.DynamicFRule, collect(r1.fwd_oc.oc.captures))
        dyns2 = filter(c -> c isa Mooncake.DynamicFRule, collect(r2.fwd_oc.oc.captures))
        # The `Base.inferencebarrier` in `fwd_cache_dyn` only forces a captured `DynamicFRule` on
        # Julia ≥ 1.11; 1.10 resolves it with no top-level dynamic-rule capture (empty
        # `oc.captures`), so the shared-`cache` scenario cannot arise there (the frule still runs
        # correctly). Check the independent-copy invariant only on ≥ 1.11, where the capture exists;
        # `only(dyns1)` then fails loudly if a future regression drops it.
        @static if VERSION >= v"1.11-"
            dyn1 = only(dyns1)
            dyn2 = only(dyns2)
            @test dyn1 !== dyn2
            @test dyn1.cache !== dyn2.cache
        end
    end
end;

@testset "nfwd primitive coverage" begin
    # The nfwd classifier trusts one set of dual-transparent ops: structural `Core.Builtin`s
    # in `_NFWD_SAFE_BUILTINS`. Every other builtin — and all intrinsics / foreigncalls — that touches a
    # dual routes the function to the frule transform. If Julia gains a builtin it must be
    # classified deliberately (safe vs opaque indirection) rather than silently trusted, so assert
    # every current builtin is in exactly one of the two sets; a new one fails here loudly.
    builtins = Set(
        n for n in names(Core; all=true) if
        isdefined(Core, n) && getfield(Core, n) isa Core.Builtin
    )
    accounted = Mooncake._NFWD_SAFE_BUILTINS ∪ Mooncake._NFWD_OPAQUE_BUILTINS
    @test isempty(setdiff(builtins, accounted))
    @test isempty(intersect(Mooncake._NFWD_SAFE_BUILTINS, Mooncake._NFWD_OPAQUE_BUILTINS))
end
