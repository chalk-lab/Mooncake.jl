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

    # Try try-catch statements.
    rng = StableRNG(123)
    perf_flag = :none
    interface_only = false
    is_primitive = false
    mode = ForwardMode
    TestUtils.test_rule(rng, foo, 5.0; perf_flag, interface_only, is_primitive, mode)
end;
