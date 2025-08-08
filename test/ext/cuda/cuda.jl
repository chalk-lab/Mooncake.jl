using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule

@testset "cuda" begin
    if CUDA.functional()
        # Check we can operate on CuArrays.
        p = CuArray{Float32,2,CUDA.DeviceMemory}(undef, 8, 8)
        test_tangent_interface(StableRNG(123456), p; interface_only=false)
        test_tangent_splitting(StableRNG(123456), p)

        # Check we can instantiate a CuArray.
        test_rule(
            StableRNG(123456),
            CuArray{Float32,1,CUDA.DeviceMemory},
            undef,
            256;
            interface_only=true,
            is_primitive=true,
            debug_mode=true,
            mode=Mooncake.ReverseMode,
        )
    else
        println("Tests are skipped since no CUDA device was found. ")
    end
end
