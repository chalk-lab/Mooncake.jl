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

@testset "build_frule maybeinline_primitive" begin
    # Test with a primitive function
    args = (sin, 5.0)

    # Test default is maybeinline_primitive=true
    frule_default = Mooncake.build_frule(Mooncake.zero_dual(sin), 5.0)
    @test frule_default isa Mooncake.PrimitiveFRule{<:Any,true}

    # Test maybeinline_primitive=true (explicit) returns PrimitiveFRule{_, true}
    frule_inline = Mooncake.build_frule(
        Mooncake.zero_dual(sin), 5.0; maybeinline_primitive=true
    )
    @test frule_inline isa Mooncake.PrimitiveFRule{<:Any,true}

    # Test maybeinline_primitive=false returns PrimitiveFRule{_, false}
    frule_noinline = Mooncake.build_frule(
        Mooncake.zero_dual(sin), 5.0; maybeinline_primitive=false
    )
    @test frule_noinline isa Mooncake.PrimitiveFRule{<:Any,false}

    # Both should produce correct results when called
    x_dual = Mooncake.Dual(5.0, 1.0)
    result_inline = frule_inline(Mooncake.zero_dual(sin), x_dual)
    result_noinline = frule_noinline(Mooncake.zero_dual(sin), x_dual)
    @test Mooncake.primal(result_inline) == Mooncake.primal(result_noinline)
    @test Mooncake.tangent(result_inline) ≈ Mooncake.tangent(result_noinline)

    # Verify correctness of computed derivatives
    @test Mooncake.primal(result_inline) ≈ sin(5.0)
    @test Mooncake.tangent(result_inline) ≈ cos(5.0)  # d/dx sin(x) = cos(x)

    # Test with debug_mode=true wraps in DebugFRule
    frule_debug_noinline = Mooncake.build_frule(
        Mooncake.zero_dual(sin), 5.0; debug_mode=true, maybeinline_primitive=false
    )
    @test frule_debug_noinline isa Mooncake.DebugFRule

    # Test varargs form also accepts maybeinline_primitive
    frule_args_noinline = Mooncake.build_frule(
        Mooncake.zero_dual(sin), 5.0; maybeinline_primitive=false
    )
    @test frule_args_noinline isa Mooncake.PrimitiveFRule{<:Any,false}
end;
