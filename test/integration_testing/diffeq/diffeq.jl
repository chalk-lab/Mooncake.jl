using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using OrdinaryDiffEq, SciMLSensitivity, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule
using Mooncake: ReverseMode

# Helper function for Mooncake gradient computation  
function mooncake_gradient(f, x)
    Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, x), f, x)[2][2]
end

# Define the ODE function from the original issue
odef(du, u, p, t) = du .= u .* p

# Define the sensitivity loss function from the original issue
struct senseloss0{T}
    sense::T
end

function (f::senseloss0)(u0p)
    prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
    sum(solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12, saveat=0.1))
end

@testset "diffeq" begin
    rng = StableRNG(123456)

    # Test parameters from the original issue
    u0p = [2.0, 3.0]

    @testset "senseloss0 with InterpolatingAdjoint" begin
        sense_func = senseloss0(InterpolatingAdjoint())

        # First test that the function works
        @testset "Function evaluation" begin
            result = sense_func(u0p)
            @test result isa Real
            @test isfinite(result)
        end

        # Test Mooncake gradient computation 
        @testset "mooncake_gradient computation" begin
            dup_mc = mooncake_gradient(sense_func, u0p)
            @test dup_mc isa Vector
            @test length(dup_mc) == 2
            @test all(isfinite, dup_mc)
        end

        # Test with Mooncake's test_rule
        @testset "test_rule evaluation" begin
            test_rule(
                rng,
                sense_func,
                u0p;
                is_primitive=false,
                unsafe_perturb=true,
                mode=ReverseMode,
            )
        end
    end
end
