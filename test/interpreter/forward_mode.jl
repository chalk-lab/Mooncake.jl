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

# Helpers for the world-advance rule-staleness regression test (one trigger from issue
# #1209). Defining `issue1209_inner(::Float64)` later tightens `issue1209_callee`'s return
# type Float32->Float64 via a mid-pass world advance. `issue1209_lazy` reaches the callee
# statically (LazyFRule), `issue1209_dyn` dynamically (DynamicFRule). NOTE: this covers only
# the world-advance trigger; the issue's headline MWE, where `frule_type` widens the
# predicted return type by inference type-complexity (not a world advance), is a distinct
# failure mode that this fix does not address.
issue1209_inner(x) = Float32(x) * 2.0f0
@noinline issue1209_callee(x) = issue1209_inner(x)
issue1209_lazy(x) = issue1209_callee(x)
const ISSUE1209_FNS = Function[issue1209_callee]
issue1209_dyn(x) = (ISSUE1209_FNS[1])(x)

@testset "s2s_forward_mode_ad" begin
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

    # Without the fix the lazy path throws a `convert` MethodError in _build_rule! after the
    # world advance; both lazy and dynamic must return the build-world result (Float32), not
    # the post-advance world's (Float64).
    @testset "stale rule build-world after world advance (issue #1209 trigger)" begin
        lazy = Mooncake.build_frule(issue1209_lazy, 1.5)
        dyn = Mooncake.build_frule(issue1209_dyn, 1.5)
        @eval issue1209_inner(x::Float64) = x * 2.0  # advance world; tightens callee's type
        lazy_out = Base.invokelatest(
            lazy, Mooncake.zero_dual(issue1209_lazy), Mooncake.Dual(1.5, 1.0)
        )
        dyn_out = Base.invokelatest(
            dyn, Mooncake.zero_dual(issue1209_dyn), Mooncake.Dual(1.5, 1.0)
        )
        @test Mooncake.primal(lazy_out) === 3.0f0
        @test Mooncake.primal(dyn_out) === 3.0f0
    end
end;
