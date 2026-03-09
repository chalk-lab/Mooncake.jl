using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake: lgetfield
using Mooncake.TestUtils: test_tangent_interface, test_tangent_splitting, test_rule
using LinearAlgebra

@testset "cuda" begin
    cuda = CUDA.functional()
    if cuda
        # TODO: move test case definitions to `src/ext/MooncakeCUDAExt.jl`, in line
        # with other rules.
        #
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
                is_primitive=false,
            )
            test_rule(
                StableRNG(123456),
                CuArray{ET,2,CUDA.DeviceMemory},
                undef,
                (16, 32);
                interface_only=true,
                is_primitive=true,
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
        rng = StableRNG(123)
        _rand = (rng, size...) -> CuArray(randn(rng, size...))
        _sin_bcast(x) = sin.(x)
        test_cases = Any[
            # sum
            (false, :none, false, sum, _rand(rng, 64, 32)),
            # similar
            (true, :none, false, similar, _rand(rng, 64, 32)),
            # adjoint
            (false, :none, false, adjoint, _rand(rng, 64, 32)),
            (false, :none, false, adjoint, _rand(rng, ComplexF64, 64, 32)),
            # transpose 
            (false, :none, false, transpose, _rand(rng, 64, 32)),
            (false, :none, false, transpose, _rand(rng, ComplexF64, 64, 32)),
            # reshape — exercises the DataRef-based _new_ rule
            (false, :none, false, x -> reshape(x, 32, 64), _rand(rng, 64, 32)),
            (false, :none, false, x -> reshape(x, 32, 64), _rand(rng, ComplexF64, 64, 32)),
            # _new_ — direct test of the DataRef-based rule
            (
                false,
                :none,
                true,
                Mooncake._new_,
                CuArray{Float64,2,CUDA.DeviceMemory},
                getfield(_rand(rng, 64, 32), :data),
                2048,
                0,
                (64, 32),
            ),
            (
                false,
                :none,
                true,
                Mooncake._new_,
                CuArray{ComplexF64,2,CUDA.DeviceMemory},
                getfield(_rand(rng, ComplexF64, 64, 32), :data),
                2048,
                0,
                (64, 32),
            ),
            # lgetfield
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(1)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(2)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(3)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(4)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:data)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:maxsize)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:offset)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:dims)),
        ]
        @testset "$(typeof(fargs))" for (
            interface_only, perf_flag, is_primitive, fargs...
        ) in test_cases

            @info "$(typeof(fargs))"
            perf_flag = cuda ? :none : perf_flag
            test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only)
        end
    else
        println("Tests are skipped because no CUDA device was found.")
    end
end
