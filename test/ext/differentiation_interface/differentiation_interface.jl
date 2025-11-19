using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Mooncake: Mooncake
using Test

test_differentiation(
    [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())];
    excluded=SECOND_ORDER,
    logging=true,
)

# Test Hessian computation using forward-over-reverse (it hangs)
# test_differentiation(
#     [SecondOrder(AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing))];
#     excluded=vcat(FIRST_ORDER, [:hvp, :second_derivative]),  # Only test hessian
#     logging=true,
# )

@testset "Mooncake Hessian tests" begin
    backend = SecondOrder(
        AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing)
    )

    # Sum: Hessian is zero
    @testset "sum" begin
        @test DI.hessian(sum, backend, [2.0]) == [0.0]
    end

    # Rosenbrock 2D at [1.2, 1.2]
    @testset "Rosenbrock" begin
        rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
        H = DI.hessian(rosen, backend, [1.2, 1.2])
        @test isapprox(H, [1250.0 -480.0; -480.0 200.0]; rtol=1e-10, atol=1e-12)
    end
end
