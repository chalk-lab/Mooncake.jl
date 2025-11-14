using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, AMDGPU, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule

@testset "amdgpu" begin
    if AMDGPU.functional()
        # Check we can operate on CuArrays of various element types.
        @testset for ET in (Float32, Float64, ComplexF32, ComplexF64)
            p = ROCArray{ET,2,AMDGPU.Mem.HIPBuffer}(undef, 8, 8)
            test_tangent_interface(StableRNG(123456), p; interface_only=false)
            test_tangent_splitting(StableRNG(123456), p)

            # Check we can instantiate a CuArray.
            test_rule(
                StableRNG(123456),
                ROCArray{ET,1,AMDGPU.Mem.HIPBuffer},
                undef,
                256;
                interface_only=true,
                is_primitive=true,
                debug_mode=true,
                mode=Mooncake.ReverseMode,
            )
        end
    else
        println("Tests are skipped since no AMDGPU device was found. ")
    end
end
