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
end;
