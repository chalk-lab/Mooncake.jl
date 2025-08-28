@testset "function_wrappers" begin
    rng = Xoshiro(123)
    _data = Ref{Float64}(5.0)
    @testset "$p" for p in Any[
        FunctionWrapper{Float64,Tuple{Float64}}(sin),
        FunctionWrapper{Float64,Tuple{Float64}}(x -> x * _data[]),
    ]
        TestUtils.test_tangent_interface(rng, p)
        TestUtils.test_tangent_splitting(rng, p)

        # Check that we can run `to_cr_tangent` on tangents for FunctionWrappers.
        t = zero_tangent(p)
        @test Mooncake.to_cr_tangent(t) === t
        
        # Will's comment test: comprehensive validation of to_cr_tangent for FunctionWrapperTangent
        # (lines 152-155 in src/rrules/function_wrappers.jl)
        # Currently returns tangent as-is, but this may need revision if FunctionWrappers
        # become more widely used in ChainRules ecosystem
        @testset "Will's comment: to_cr_tangent compatibility with ChainRules" begin
            # Test with different tangent states (zero and random)
            t_zero = zero_tangent(p)
            t_random = randn_tangent(rng, p)
            
            # Validate current behavior: to_cr_tangent returns the tangent unchanged
            @test Mooncake.to_cr_tangent(t_zero) === t_zero
            @test Mooncake.to_cr_tangent(t_random) === t_random
            
            # Ensure the tangent type is preserved exactly
            @test typeof(Mooncake.to_cr_tangent(t_zero)) === typeof(t_zero)
            @test typeof(Mooncake.to_cr_tangent(t_random)) === typeof(t_random)
            
            # Test idempotency: applying to_cr_tangent multiple times should be safe
            cr_tangent_1 = Mooncake.to_cr_tangent(t_zero)
            cr_tangent_2 = Mooncake.to_cr_tangent(cr_tangent_1)
            @test cr_tangent_1 === cr_tangent_2
            
            # Regression test: if the implementation changes to return a different type,
            # this test will catch it and signal that ChainRules compatibility needs review
            @test Mooncake.to_cr_tangent(t_zero) isa Mooncake.FunctionWrapperTangent
        end
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:function_wrappers), ReverseMode)
end
