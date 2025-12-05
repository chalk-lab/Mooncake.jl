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

# Test Hessian computation using forward-over-reverse with DITest scenarios.
test_differentiation(
    [SecondOrder(AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing))];
    excluded=vcat(FIRST_ORDER, [:hvp, :second_derivative]),
    logging=true,
)

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

    # Test higher integer powers (fixed by adding frule for ^(Float, Int))
    @testset "higher powers" begin
        @test DI.hessian(x -> x[1]^4, backend, [2.0]) ≈ [48.0]
        @test DI.hessian(x -> x[1]^6, backend, [2.0]) ≈ [480.0]
    end

    @testset "https://github.com/chalk-lab/Mooncake.jl/issues/632" begin
        function gams_objective(x)
            return (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    (
                                                                                        (
                                                                                            (
                                                                                                (
                                                                                                    (
                                                                                                        (
                                                                                                            (
                                                                                                                (
                                                                                                                    (
                                                                                                                        x[1] *
                                                                                                                        x[1] +
                                                                                                                        x[10] *
                                                                                                                        x[10]
                                                                                                                    ) *
                                                                                                                    (
                                                                                                                        x[1] *
                                                                                                                        x[1] +
                                                                                                                        x[10] *
                                                                                                                        x[10]
                                                                                                                    ) -
                                                                                                                    4 *
                                                                                                                    x[1]
                                                                                                                ) +
                                                                                                                3
                                                                                                            ) +
                                                                                                            (
                                                                                                                x[2] *
                                                                                                                x[2] +
                                                                                                                x[10] *
                                                                                                                x[10]
                                                                                                            ) *
                                                                                                            (
                                                                                                                x[2] *
                                                                                                                x[2] +
                                                                                                                x[10] *
                                                                                                                x[10]
                                                                                                            )
                                                                                                        ) -
                                                                                                        4 *
                                                                                                        x[2]
                                                                                                    ) +
                                                                                                    3
                                                                                                ) +
                                                                                                (
                                                                                                    x[3] *
                                                                                                    x[3] +
                                                                                                    x[10] *
                                                                                                    x[10]
                                                                                                ) *
                                                                                                (
                                                                                                    x[3] *
                                                                                                    x[3] +
                                                                                                    x[10] *
                                                                                                    x[10]
                                                                                                )
                                                                                            ) -
                                                                                            4 *
                                                                                            x[3]
                                                                                        ) +
                                                                                        3
                                                                                    ) +
                                                                                    (
                                                                                        x[4] *
                                                                                        x[4] +
                                                                                        x[10] *
                                                                                        x[10]
                                                                                    ) * (
                                                                                        x[4] *
                                                                                        x[4] +
                                                                                        x[10] *
                                                                                        x[10]
                                                                                    )
                                                                                ) -
                                                                                4 * x[4]
                                                                            ) + 3
                                                                        ) +
                                                                        (
                                                                            x[5] * x[5] +
                                                                            x[10] * x[10]
                                                                        ) * (
                                                                            x[5] * x[5] +
                                                                            x[10] * x[10]
                                                                        )
                                                                    ) - 4 * x[5]
                                                                ) + 3
                                                            ) +
                                                            (x[6] * x[6] + x[10] * x[10]) *
                                                            (x[6] * x[6] + x[10] * x[10])
                                                        ) - 4 * x[6]
                                                    ) + 3
                                                ) +
                                                (x[7] * x[7] + x[10] * x[10]) *
                                                (x[7] * x[7] + x[10] * x[10])
                                            ) - 4 * x[7]
                                        ) + 3
                                    ) +
                                    (x[8] * x[8] + x[10] * x[10]) *
                                    (x[8] * x[8] + x[10] * x[10])
                                ) - 4 * x[8]
                            ) + 3
                        ) + (x[9] * x[9] + x[10] * x[10]) * (x[9] * x[9] + x[10] * x[10])
                    ) - 4 * x[9]
                ) + 3
            )
        end
        x0 = [0.0; fill(1.0, 9)]
        H = DI.hessian(gams_objective, backend, x0)

        # Expected Hessian at x0:
        # - H[1,1] = 4 (since x₁=0)
        # - H[i,i] = 16 for i ∈ 2:9 (since xᵢ=1, x₁₀=1)
        # - H[10,10] = 140 (sum of contributions from all 9 terms)
        # - H[i,10] = H[10,i] = 8xᵢx₁₀ = 0 for i=1, 8 for i∈2:9
        H_expected = zeros(10, 10)
        H_expected[1, 1] = 4.0
        for i in 2:9
            H_expected[i, i] = 16.0
            H_expected[i, 10] = 8.0
            H_expected[10, i] = 8.0
        end
        H_expected[10, 10] = 140.0

        @test H ≈ H_expected
    end
end
