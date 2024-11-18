using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent, test_rule

@testset "cuda" begin

    # Check we can operate on CuArrays.
    test_tangent(
        StableRNG(123456), CuArray{Float32, 2, CUDA.DeviceMemory}(undef, 8, 8);
        interface_only=false,
    )

    # Check we can instantiate a CuArray.
    test_rule(
        StableRNG(123456), CuArray{Float32, 1, CUDA.DeviceMemory}, undef, 256;
        interface_only=true, is_primitive=true, debug_mode=true,
    )
end
