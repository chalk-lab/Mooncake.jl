using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, FunctionWrappers, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule, test_tangent_interface, test_tangent_splitting
using FunctionWrappers: FunctionWrapper

@testset "function_wrappers" begin

    # Tangent interface tests.
    rng = StableRNG(123)
    _data = Ref{Float64}(5.0)
    @testset "$p" for p in Any[
        FunctionWrapper{Float64,Tuple{Float64}}(sin),
        FunctionWrapper{Float64,Tuple{Float64}}(x -> x * _data[]),
    ]
        test_tangent_interface(rng, p)
        test_tangent_splitting(rng, p)

        # Check that we can run `to_cr_tangent` on tangents for FunctionWrappers.
        t = Mooncake.zero_tangent(p)
        @test Mooncake.to_cr_tangent(t) === t
    end

    # Rule testing.
    @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in [
        (false, :none, true, FunctionWrapper{Float64,Tuple{Float64}}, sin),
        (false, :none, true, FunctionWrapper{Float64,Tuple{Float64}}(sin), 5.0),
        (
            false,
            :none,
            false,
            function (x, y)
                p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
                out = 0.0
                for _ in 1:1_000
                    out += p(x)
                end
                return out
            end,
            5.0,
            4.0,
        ),
        (
            false,
            :none,
            false,
            function (x::Vector{Float64}, y::Float64)
                p = FunctionWrapper{Float64,Tuple{Float64}}(x -> x * y)
                out = 0.0
                for _x in x
                    out += p(_x)
                end
                return out
            end,
            randn(100),
            randn(),
        ),
    ]
        mode = Mooncake.ReverseMode
        test_rule(rng, fargs...; perf_flag, is_primitive, interface_only, mode)
    end
end
