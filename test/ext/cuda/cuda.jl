using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule

@testset "cuda" begin
    if CUDA.functional()
        # Check we can operate on CuArrays of various element types.
        @testset for ET in (Float32, Float64, ComplexF32, ComplexF64)
            p = CuArray{ET,2,CUDA.DeviceMemory}(undef, 8, 8)
            test_tangent_interface(StableRNG(123456), p; interface_only=false)
            test_tangent_splitting(StableRNG(123456), p)

            # Check we can instantiate a CuArray.
            test_rule(
                StableRNG(123456),
                CuArray{ET,1,CUDA.DeviceMemory},
                undef,
                256;
                interface_only=true,
                is_primitive=true,
                debug_mode=true,
                mode=Mooncake.ReverseMode,
            )
            dp = Mooncake.zero_codual(p)
            if ET <: Real
                @test Mooncake.arrayify(dp) == (p, Mooncake.zero_tangent(p))
            elseif ET <: Complex
                primal_p, tangent_ p = Mooncake.arrayify(dp)
                @test (primal_p, tangent_p) isa Tuple{CuArray{ET, 2, CUDA.DeviceMemory}, CuArray{ET, 2, CUDA.DeviceMemory}}
                @test all(iszero, primal_p)
                @test all(iszero, tangent_p)
            end
        end
    else
        println("Tests are skipped since no CUDA device was found. ")
    end
end
