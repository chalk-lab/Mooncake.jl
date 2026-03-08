@testset "bfloat16" begin
    @static if isdefined(Core, :BFloat16)
        rng = sr(123)
        TestUtils.test_tangent_interface(rng, Core.BFloat16(1.5))
        TestUtils.test_tangent_splitting(rng, Core.BFloat16(1.5))
        TestUtils.run_rule_test_cases(StableRNG, Val(:bfloat16))
    else
        @info "Skipping BFloat16 tests: Core.BFloat16 not available on this Julia version"
    end
end
