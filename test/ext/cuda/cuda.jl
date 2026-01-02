using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule
using LinearAlgebra

@testset "cuda" begin
    cuda = CUDA.functional()
    if cuda 
        # Check we can operate on CuArrays of various element types.
        @testset for ET in (Float32, Float64, ComplexF32, ComplexF64)
            # Use `undef` to test against garbage memory (NaNs, Infs, subnormals).
            # `randn` generates well-behaved values and can miss these edge cases.
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
            test_rule(
                StableRNG(123456),
                CuArray{ET,2,CUDA.DeviceMemory},
                undef,
                (16, 32);
                interface_only=true,
                is_primitive=true,
                debug_mode=true,
                mode=Mooncake.ReverseMode,
            )
            dp = Mooncake.zero_codual(p)
            if ET <: Real
                @test Mooncake.arrayify(dp) == (p, Mooncake.zero_tangent(p))
            elseif ET <: Complex
                primal_p, tangent_p = Mooncake.arrayify(dp)
                @test (primal_p, tangent_p) isa
                    Tuple{CuArray{ET,2,CUDA.DeviceMemory},CuArray{ET,2,CUDA.DeviceMemory}}
                @test all(iszero, tangent_p)
            end
        end
        Trng = CUDA.RNG
        rng  = StableRNG(123)
        _rand = (rng, size...) -> CuArray(randn(rng, size...))
        test_cases = Any[
            # sum
            (false, :none, false, sum, _rand(rng, 64, 32)),
            # similar 
            (true, :none, false, similar, _rand(rng, 64, 32)),
            # adjoint
            (false, :none, false, adjoint, _rand(rng, ComplexF64, 64, 32)),
        ]
        @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in
                                        test_cases

            @info "$(typeof(fargs))"
            perf_flag = cuda ? :none : perf_flag
            mode = Mooncake.ReverseMode
            test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only, mode)
        end
    else
        println("Tests are skipped since no CUDA device was found. ")
    end
end
