# Helpers for the world-advance staleness test below (issue #1218).
# `stale_fwd_lazy` uses LazyFRule; `stale_fwd_dyn` uses DynamicFRule.
stale_fwd_inner(x) = Float32(x) * 2.0f0
@noinline stale_fwd_callee(x) = stale_fwd_inner(x)
# Two `:invoke` levels: one for the pinned rebuild, one for the rules that rebuild itself
# constructs, which predicted at the current world until they were given the pinned one.
@noinline stale_fwd_mid(x) = stale_fwd_callee(x)
stale_fwd_lazy(x) = stale_fwd_mid(x)
const STALE_FWD_FNS = Function[stale_fwd_mid]
stale_fwd_dyn(x) = (STALE_FWD_FNS[1])(x)

@noinline refined_fwd_inner(x) =
    sizeof(x) == sizeof(typeof(x)) ? x : Base.inferencebarrier(x)
@noinline refined_fwd_outer(x) = refined_fwd_inner(x)
refined_fwd(x) = refined_fwd_outer(x)

# Dynamic dispatch (`inferencebarrier` hides the callee) so the derived rule captures a
# `DynamicFRule` with a mutable `cache` Dict — used by the cache-hit `_copy` regression below.
fwd_cache_dyn(x) = Base.inferencebarrier(sin)(x)::Float64 + x

module FwdAliasGlobals
# Read while also passed as the argument, to exercise the refusal of an argument that aliases a
# differentiable global.
const alias_vector = [1.0, 2.0]
alias_read_only(x) = sum(x .* alias_vector)
alias_scalar(x) = x[1] * alias_vector[1]
end

# GC preservation: a forward rule must keep BOTH the primal and the partials storage alive for
# the duration of a `GC.@preserve` scope. `gc_preserve_storage` reaches the backing `Memory` so
# collection can be observed without dereferencing a possibly stale pointer.
const GC_PRESERVE_OWNERS = Ref((WeakRef(nothing), WeakRef(nothing)))
gc_preserve_storage(x) = @static VERSION >= v"1.11-" ? x.ref.mem : x
# The partials of an `NDualArray` live in one block whose parent vector owns the storage.
function gc_preserve_partials(v)
    gc_preserve_storage(getfield(getfield(v, :partials_block), :parent))
end
function gc_preserve_observe(x)
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(x)), WeakRef(gc_preserve_storage(x))
    )
    return x
end
Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gc_preserve_observe),Vector{Float64}
}
function Mooncake.frule!!(::Lifted{typeof(gc_preserve_observe),Nw}, x::Lifted) where {Nw}
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(primal(x))), WeakRef(gc_preserve_partials(tangent(x)))
    )
    return x
end
@noinline function gc_preserve_alive(::Ptr{Float64})
    GC.gc(true)
    return map(w -> w.value !== nothing, GC_PRESERVE_OWNERS[])
end
Mooncake.@zero_derivative DefaultCtx Tuple{typeof(gc_preserve_alive),Ptr{Float64}}
function gc_preserve_probe(x)
    a = gc_preserve_observe(copy(x))
    return GC.@preserve a gc_preserve_alive(pointer(a))
end

@testset "s2s_forward_mode_ad" begin
    @testset "rule return type after normalisation (#1327)" begin
        for x in (1.0f0, 1.0), debug_mode in (false, true)
            rule = Mooncake.build_frule(refined_fwd, x; debug_mode)
            result = rule(
                Mooncake.zero_lifted(Val(1), refined_fwd), Mooncake.lift(x, one(x))
            )
            @test primal(result) === x
            @test tangent(result, 1) === one(x)
        end
    end

    @testset "GC preservation of primal and tangent storage (issue #1303)" begin
        # Observe collection without dereferencing a potentially stale pointer.
        @test gc_preserve_probe([1.0, 2.0]) == (true, true)
        rule = Mooncake.build_frule(gc_preserve_probe, [1.0, 2.0])
        for _ in 1:2
            result = rule(
                Mooncake.zero_lifted(Val(1), gc_preserve_probe),
                Mooncake.lift([1.0, 2.0], [3.0, 4.0]),
            )
            @test primal(result) == (true, true)
            GC.gc(true)
            @test all(w -> w.value === nothing, GC_PRESERVE_OWNERS[])
        end
    end

    test_cases = collect(enumerate(TestResources.generate_test_functions()))
    @testset "$n - $(_typeof((fx)))" for (n, (int_only, pf, opts, fx...)) in test_cases
        @info "$n: $(_typeof(fx))"
        TestUtils._case_skip_forward(opts) && continue
        rng = Xoshiro(123546)
        mode = ForwardMode
        skip_chunked = TestUtils._case_skip_chunked(opts)
        TestUtils.test_rule(
            rng,
            fx...;
            perf_flag=pf,
            interface_only=int_only,
            is_primitive=false,
            mode,
            skip_chunked,
            throws=TestUtils._case_throws(opts),
            primal_throws=TestUtils._case_primal_throws(opts),
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

    @testset "argument aliasing a differentiable global is refused" begin
        G = FwdAliasGlobals.alias_vector
        @testset "$f" for f in
                          (FwdAliasGlobals.alias_read_only, FwdAliasGlobals.alias_scalar)
            cache = Mooncake.prepare_derivative_cache(f, G)
            @test_throws ArgumentError Mooncake.value_and_derivative!!(
                cache, (f, Mooncake.zero_tangent(f)), (G, [1.0, 1.0])
            )
            # The same global read with an unaliased argument is supported and unaffected.
            y = [3.0, 4.0]
            c2 = Mooncake.prepare_derivative_cache(f, y)
            @test Mooncake.value_and_derivative!!(
                c2, (f, Mooncake.zero_tangent(f)), (y, [1.0, 1.0])
            )[2] ≈ (f === FwdAliasGlobals.alias_read_only ? sum(G) : G[1])
        end
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

    # Without the fix the lazy path throws a `convert` MethodError in _build_rule!; both
    # paths must return the build-world result (Float32), not the post-advance (Float64).
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

    # Separate builds must not share mutable DynamicFRule caches (threads / nested AD).
    @testset "cache-hit returns an independent rule copy" begin
        interp = Mooncake.MooncakeInterpreter(ForwardMode)
        sig = Tuple{typeof(fwd_cache_dyn),Float64}
        r1 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)
        r2 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)  # cache HIT
        dyns1 = filter(c -> c isa Mooncake.DynamicFRule, collect(r1.fwd_oc.oc.captures))
        dyns2 = filter(c -> c isa Mooncake.DynamicFRule, collect(r2.fwd_oc.oc.captures))
        # Julia 1.10 resolves the callee without a DynamicFRule capture. On 1.11+,
        # only() also detects regressions that lose the expected capture.
        @static if VERSION >= v"1.11-"
            dyn1 = only(dyns1)
            dyn2 = only(dyns2)
            @test dyn1 !== dyn2
            @test dyn1.cache !== dyn2.cache
        end
    end
end;

@testset "a type-observing branch takes the primal's side" begin
    # Type queries must observe the primal: e.g. sizeof(NDual{Float64,1}) == 16,
    # not Float64's 8, which would change both the branch's value and derivative.
    sz(x) = sizeof(x) == 8 ? x * x : x * x * x
    ty(x) = typeof(x) === Float64 ? x * x : x * x * x
    nf(x) = nfields(x) == 0 ? x * x : x * x * x
    szt(t) = sizeof(t[1]) == 8 ? t[1] * t[1] : t[1] * t[1] * t[1]
    for (f, x, grad) in
        ((sz, 2.0, 4.0), (ty, 2.0, 4.0), (nf, 2.0, 4.0), (szt, (2.0, 5.0), (4.0, 0.0)))
        v, g = Mooncake.value_and_gradient!!(Mooncake.prepare_derivative_cache(f, x), f, x)
        @test (v, g[2]) == (4.0, grad)
    end
end
