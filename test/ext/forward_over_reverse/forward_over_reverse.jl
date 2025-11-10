using Pkg, Test
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface
import DifferentiationInterface as DI
using Mooncake: Mooncake

@testset "forward-over-reverse via DifferentiationInterface" begin
    backend = SecondOrder(AutoMooncakeForward(), AutoMooncake())

    @test DI.hessian(sum, backend, [2.0]) == [0.0]

    rosen(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
    H = DI.hessian(rosen, backend, [1.2, 1.2])
    @test isapprox(H, [1250.0 -480.0; -480.0 200.0]; rtol=1e-10, atol=1e-12)
end
