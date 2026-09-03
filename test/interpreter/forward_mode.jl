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

# Helpers for the world-advance staleness test below (issue #1218; scope caveat at
# `_build_rule!`). `stale_fwd_lazy` reaches the callee via LazyFRule, `stale_fwd_dyn` via DynamicFRule.
stale_fwd_inner(x) = Float32(x) * 2.0f0
@noinline stale_fwd_callee(x) = stale_fwd_inner(x)
# Two `:invoke` levels: one for the pinned rebuild, one for the rules that rebuild itself
# constructs, which predicted at the current world until they were given the pinned one.
@noinline stale_fwd_mid(x) = stale_fwd_callee(x)
stale_fwd_lazy(x) = stale_fwd_mid(x)
const STALE_FWD_FNS = Function[stale_fwd_mid]
stale_fwd_dyn(x) = (STALE_FWD_FNS[1])(x)

# Dynamic dispatch (`inferencebarrier` hides the callee) so the derived rule captures a
# `DynamicFRule` with a mutable `cache` Dict — used by the cache-hit `_copy` regression below.
fwd_cache_dyn(x) = Base.inferencebarrier(sin)(x)::Float64 + x

module FwdAliasGlobals
# Read while also passed as the argument, to exercise the refusal of an argument that aliases a
# differentiable global. `alias_read_only` routes to the transform and `alias_scalar` to the
# nfwd-native path, so the pair covers both forward rule kinds.
const alias_vector = [1.0, 2.0]
alias_read_only(x) = sum(x .* alias_vector)
alias_scalar(x) = x[1] * alias_vector[1]
end

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

    # Forward counterpart of the reverse-mode testset of the same name, and bespoke for the same
    # reason: a `generate_test_functions` row's third slot is allocation bounds, which the driver
    # drops, so it cannot carry a `throws` expectation.
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

# `===` on two isbits duals compares `partials`, not just `.value`, so a branch on it flips.
_nfwd_egal_branch(x, y) = (x === y) ? x * x * x : x * y
# The same builtin on a non-differentiable operand stays admissible: the gate is the operand, not
# the name. `n` lifts to `NoDual`, so egal answers for the dual exactly what it answers for the
# primal.
_nfwd_egal_nondual(x, n::Int) = (n === 2) ? x * x : x * x * x

@testset "representation-observing builtins are refused a dual operand" begin
    # nfwd ran `_nfwd_egal_branch(2.0, 2.0)` natively and returned 4.0 against a primal of 8.0 —
    # a wrong VALUE, not merely a wrong derivative — because egal saw the differing seeds.
    @test !Mooncake._nfwd_safe(Any[typeof(_nfwd_egal_branch), Float64, Float64], 1)
    @test Mooncake._nfwd_safe(Any[typeof(_nfwd_egal_nondual), Float64, Int], 1)
end

# `sizeof` folds to 16 for `NDual{Float64,1}` against 8 for `Float64`, so the optimiser leaves no
# call behind for an operand test to see; the divergence shows only in the folded constants.
_nfwd_folded_size(x) = x * sizeof(x)
# A dual carrying the primal's value folds to a constant too, and must NOT read as divergence:
# `sum(abs2, x)` folds `NDual{Float64,1}(0.0, (0.0,))` against `0.0`.
_nfwd_folded_lift(x) = sum(abs2, x)
# Only the DUAL side folds here: `sizeof(::NDualArray)` is the wrapper's fixed struct size, while
# `sizeof(::Vector{Float64})` scales with length and stays unknown. Ran natively and returned 192.0
# against a truth of 144.0.
_nfwd_folded_one_sided(x) = sum(x) * sizeof(x)

@testset "a folded constant must be the lift of the primal's" begin
    # Ran natively and returned 32.0 against a primal of 16.0.
    @test !Mooncake._nfwd_safe(Any[typeof(_nfwd_folded_size), Float64], 1)
    @test Mooncake._nfwd_safe(Any[typeof(_nfwd_folded_lift), Vector{Float64}], 1)
    @test !Mooncake._nfwd_safe(Any[typeof(_nfwd_folded_one_sided), Vector{Float64}], 1)
end

# `objectid` is content-addressed for an isbits value, so an isbits dual hashes its own partials.
# Ran natively and returned a primal of 8.0 at seed 1.0 against a truth of 4.0.
_nfwd_objectid_scalar(x) = iseven(objectid(x)) ? x * x : x * x * x
# A dual that is not isbits hashes the wrapper's address instead, which the derivative does not
# affect, so it stays admissible — that is what lets a logdensity's dict bookkeeping run natively.
# Admissible only where the surrounding body is: `sum(sin.(x))` is in-protocol on 1.12 alone.
_nfwd_objectid_array(x) = (objectid(x); sum(sin.(x)))

@testset "objectid is trusted only for a dual that is not isbits" begin
    @test !Mooncake._nfwd_safe(Any[typeof(_nfwd_objectid_scalar), Float64], 1)
    @test Mooncake._nfwd_safe(Any[typeof(_nfwd_objectid_array), Vector{Float64}], 1) ==
        (VERSION >= v"1.12-")
end

@testset "nfwd primitive coverage" begin
    # The nfwd classifier sorts builtins three ways: `_NFWD_SAFE_BUILTINS` pass a dual through
    # unchanged and are trusted unconditionally; `_NFWD_REPR_BUILTINS` answer differently for a dual
    # than for its primal and are trusted only when no operand is dual-typed; `_NFWD_OPAQUE_BUILTINS`
    # are indirections the scan cannot see through. Everything else — intrinsics, foreigncalls —
    # routes to the frule transform on contact with a dual. If Julia gains a builtin it must be
    # classified deliberately rather than silently trusted, so assert every current builtin is in
    # exactly one of the three sets; a new one fails here loudly.
    builtins = Set(
        n for n in names(Core; all=true) if
        isdefined(Core, n) && getfield(Core, n) isa Core.Builtin
    )
    sets = (
        Mooncake._NFWD_SAFE_BUILTINS,
        Mooncake._NFWD_REPR_BUILTINS,
        Mooncake._NFWD_OPAQUE_BUILTINS,
    )
    @test isempty(setdiff(builtins, ∪(sets...)))
    for (i, a) in enumerate(sets), b in sets[(i + 1):end]
        @test isempty(intersect(a, b))
    end
end

@testset "nfwd rejects array-length mutation" begin
    # A length-changing op on a dual array cannot run on the fixed-shape `NDualArray`, so the
    # classifier must reject it and fall back to the transform (which handles growth). Regression
    # for `append!` lowering to `invoke push!(::NDualArray, …)` → `resize!(::NDualArray)` MethodError.
    @test !Mooncake._nfwd_safe(Any[typeof(append!), Vector{Float64}, Vector{Float64}], 1)
    @test !Mooncake._nfwd_safe(Any[typeof(push!), Vector{Float64}, Float64], 1)
    # A non-mutating array reduction fires natively only on 1.12+. On 1.10/1.11 the broadcast
    # materialises an out-of-protocol `Array{NDual}` intermediate whose reduction takes a
    # `jl_array_ptr` foreigncall (not whitelisted), so the classifier conservatively routes it to the
    # transform — which handles it correctly (verified). 1.12's broadcast/reduction lowering keeps
    # it in-protocol, so nfwd fires. Assert the actual per-version behaviour, not a false positive.
    reduce_fn(x) = sum(sin.(x))
    @test Mooncake._nfwd_safe(Any[typeof(reduce_fn), Vector{Float64}], 1) ==
        (VERSION >= v"1.12-")
end

@testset "nfwd rejects type-observing branches" begin
    # `sizeof`/`typeof`/`nfields` answer for the DUAL, not the primal — `sizeof(NDual{Float64,1})`
    # is 16 against `Float64`'s 8 — so a branch on one takes the wrong side and returns a wrong
    # VALUE, not merely a wrong derivative. Tightening the whitelist cannot catch it: inference
    # folds the query and bakes the wrong branch in before the classifier sees the body, so the
    # classifier infers the body at primal and at dual types and rejects when they diverge.
    sz(x) = sizeof(x) == 8 ? x * x : x * x * x
    ty(x) = typeof(x) === Float64 ? x * x : x * x * x
    nf(x) = nfields(x) == 0 ? x * x : x * x * x
    for f in (sz, ty, nf)
        @test !Mooncake._nfwd_safe(Any[typeof(f), Float64], 1)
        v, g = Mooncake.value_and_gradient!!(
            Mooncake.prepare_derivative_cache(f, 2.0), f, 2.0
        )
        @test (v, g[2]) == (4.0, 4.0)
    end
    # The same branch reached through a TUPLE argument. `_nfwd_undual` must invert every shape
    # `_nfwd_projectable` admits into a dual signature; while it lacked the aggregate cases the
    # primal and dual signatures compared equal, so the check returned early and never ran.
    @test Mooncake._nfwd_undual(Tuple{Mooncake.Nfwd.NDual{Float64,1},Int}) ===
        Tuple{Float64,Int}
    @test Mooncake._nfwd_undual(
        NamedTuple{(:p, :q),Tuple{Mooncake.Nfwd.NDual{Float64,1},Int}}
    ) === NamedTuple{(:p, :q),Tuple{Float64,Int}}
    szt(t) = sizeof(t[1]) == 8 ? t[1] * t[1] : t[1] * t[1] * t[1]
    @test !Mooncake._nfwd_safe(Any[typeof(szt), Tuple{Float64,Float64}], 1)
    v, g = Mooncake.value_and_gradient!!(
        Mooncake.prepare_derivative_cache(szt, (2.0, 5.0)), szt, (2.0, 5.0)
    )
    @test (v, g[2]) == (4.0, (4.0, 0.0))
end
