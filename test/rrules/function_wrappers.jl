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
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:function_wrappers), ReverseMode)
end
