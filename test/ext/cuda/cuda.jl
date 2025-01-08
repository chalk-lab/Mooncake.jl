using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_consistency, test_fwds_rvs_data, test_rule

@testset "cuda" begin

    # Check we can operate on CuArrays.
    p = CuArray{Float32,2,CUDA.DeviceMemory}(undef, 8, 8)
    test_tangent_consistency(StableRNG(123456), p; interface_only=false)
    test_fwds_rvs_data(StableRNG(123456), p)

    # Check we can instantiate a CuArray.
    test_rule(
        StableRNG(123456),
        CuArray{Float32,1,CUDA.DeviceMemory},
        undef,
        256;
        interface_only=true,
        is_primitive=true,
        debug_mode=true,
    )
end
