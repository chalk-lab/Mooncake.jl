using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

test_differentiation(
    [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())];
    excluded=[:hvp, :second_derivative],  # Enable hessian tests only
    logging=true,
)

@testset "Mooncake second-order examples" begin
    backend = SecondOrder(AutoMooncake(; config=nothing), AutoMooncake(; config=nothing))

    # Sum: Hessian is zero
    @test DI.hessian(sum, backend, [2.0]) == [0.0;;]

    # Rosenbrock 2D at [1.2, 1.2]
    rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
    H = DI.hessian(rosen, backend, [1.2, 1.2])
    @test isapprox(H, [1250.0 -480.0; -480.0 200.0]; rtol=1e-10, atol=1e-12)
end
