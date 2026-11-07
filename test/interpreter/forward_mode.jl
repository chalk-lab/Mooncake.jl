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

const GC_PRESERVE_OWNERS = Ref((WeakRef(nothing), WeakRef(nothing)))
gc_preserve_storage(x) = @static VERSION >= v"1.11-" ? x.ref.mem : x
function gc_preserve_observe(x)
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(x)), WeakRef(gc_preserve_storage(x))
    )
    return x
end
Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gc_preserve_observe),Vector{Float64}
}
function Mooncake.frule!!(::Dual{typeof(gc_preserve_observe)}, x::Dual)
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(primal(x))), WeakRef(gc_preserve_storage(tangent(x)))
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

# Helpers for the shared-rule tests. `shared_fwd_lazy` reaches its callee through a
# LazyFRule and `shared_fwd_dyn` through a DynamicFRule; neither holds per-call state.
@noinline shared_fwd_inner(x) = sin(x) * x
shared_fwd_lazy(x) = shared_fwd_inner(x)
const SHARED_FWD_FNS = Function[shared_fwd_inner]
shared_fwd_dyn(x) = (SHARED_FWD_FNS[1])(x)

# `stateful_fwd_prim`'s primitive rule counts its calls, so a rule capturing it is unshared.
stateful_fwd_prim(x) = 2x
struct StatefulFRule
    calls::Vector{Int}
end
function (rule::StatefulFRule)(::Dual{typeof(stateful_fwd_prim)}, x::Dual{Float64})
    rule.calls[1] += 1
    return Dual(2 * primal(x), 2 * tangent(x))
end
Mooncake._copy(::StatefulFRule) = StatefulFRule([0])
Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{typeof(stateful_fwd_prim),Float64}
function Mooncake.build_primitive_frule(::Type{<:Tuple{typeof(stateful_fwd_prim),Float64}})
    return StatefulFRule([0])
end
@noinline stateful_fwd_mid(x) = stateful_fwd_prim(x) + 1.0
stateful_fwd_outer(x) = stateful_fwd_mid(x)

@testset "s2s_forward_mode_ad" begin
    @testset "rule return type after normalisation (#1327)" begin
        for x in (1.0f0, 1.0), debug_mode in (false, true)
            rule = build_frule(refined_fwd, x; debug_mode)
            result = rule(zero_dual(refined_fwd), Dual(x, one(x)))
            @test primal(result) === x
            @test tangent(result) === one(x)
        end
    end

    @testset "GC preservation of primal and tangent storage (issue #1303)" begin
        # Observe collection without dereferencing a potentially stale pointer.
        @test gc_preserve_probe([1.0, 2.0]) == (true, true)
        rule = build_frule(gc_preserve_probe, [1.0, 2.0])
        for _ in 1:2
            result = rule(zero_dual(gc_preserve_probe), Dual([1.0, 2.0], [3.0, 4.0]))
            @test primal(result) == (true, true)
            GC.gc(true)
            @test all(w -> w.value === nothing, GC_PRESERVE_OWNERS[])
        end
    end

    test_cases = collect(enumerate(TestResources.generate_test_functions()))
    @testset "$n - $(_typeof((fx)))" for (n, (int_only, pf, _, fx...)) in test_cases
        @info "$n: $(_typeof(fx))"
        rng = Xoshiro(123546)
        mode = ForwardMode
        TestUtils.test_rule(
            rng, fx...; perf_flag=pf, interface_only=int_only, is_primitive=false, mode
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

    # Both paths must return the build-world result (Float32), not the post-advance
    # (Float64); a lazy rule built at a later world fails to `convert` to its predicted type.
    @testset "stale rule build-world after world advance (issue #1218)" begin
        lazy = Mooncake.build_frule(stale_fwd_lazy, 1.5)
        dyn = Mooncake.build_frule(stale_fwd_dyn, 1.5)
        @eval stale_fwd_inner(x::Float64) = x * 2.0  # advance world; tightens callee's type
        lazy_out = Base.invokelatest(
            lazy, Mooncake.zero_dual(stale_fwd_lazy), Mooncake.Dual(1.5, 1.0)
        )
        dyn_out = Base.invokelatest(
            dyn, Mooncake.zero_dual(stale_fwd_dyn), Mooncake.Dual(1.5, 1.0)
        )
        @test Mooncake.primal(lazy_out) === 3.0f0
        @test Mooncake.primal(dyn_out) === 3.0f0
    end

    @testset "a rule without per-call state is shared" begin
        for f in (shared_fwd_lazy, shared_fwd_dyn), debug_mode in (false, true)
            rule = build_frule(f, 1.5; debug_mode)
            @test Mooncake._shared(rule)
            @test Mooncake._copy(rule) === rule
            @test build_frule(f, 1.5; debug_mode) === rule
            xs = collect(range(-2.0, 2.0; length=64))
            serial = map(x -> rule(zero_dual(f), Dual(x, 1.0)), xs)
            tasks = map(x -> Threads.@spawn(rule(zero_dual(f), Dual(x, 1.0))), xs)
            @test map(fetch, tasks) == serial
            @test map(primal, serial) == map(f, xs)
            @test map(tangent, serial) ≈ map(x -> cos(x) * x + sin(x), xs)
        end
    end

    @testset "an unshared rule is reused by single-threaded calls and copied under contention" begin
        rule = build_frule(stateful_fwd_outer, 1.5)
        @test Mooncake._shared(rule)
        lazy = only(filter(c -> c isa Mooncake.LazyFRule, collect(rule.fwd_oc.oc.captures)))
        rule(zero_dual(stateful_fwd_outer), Dual(1.5, 1.0))
        inner = @atomic lazy.rule
        @test !Mooncake._shared(inner)
        @test Mooncake._copy(inner) !== inner
        rule(zero_dual(stateful_fwd_outer), Dual(1.5, 1.0))
        @test (@atomic lazy.rule) === inner
        stateful = only(filter(c -> c isa StatefulFRule, collect(inner.fwd_oc.oc.captures)))
        @test stateful.calls == [2]

        # Take `inner` out, as a concurrent call would: the next call runs on its own copy.
        @test (@atomicreplace lazy.rule inner => nothing).success
        output = rule(zero_dual(stateful_fwd_outer), Dual(1.5, 1.0))
        @test output == Dual(4.0, 2.0)
        @test stateful.calls == [2]
        @test (@atomic lazy.rule) !== inner
        xs = collect(range(-2.0, 2.0; length=64))
        tasks = map(
            x -> Threads.@spawn(rule(zero_dual(stateful_fwd_outer), Dual(x, 1.0))), xs
        )
        @test map(fetch, tasks) == map(x -> Dual(2x + 1.0, 2.0), xs)
    end
end;
