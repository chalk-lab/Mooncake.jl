using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake

# Test first-order differentiation (reverse mode)
test_differentiation(
    [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())];
    excluded=SECOND_ORDER,
    logging=true,
)

# Test second-order differentiation (forward-over-reverse for Hessian)
test_differentiation(
    [SecondOrder(AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing))],
    DifferentiationInterfaceTest.default_scenarios(; linalg=false);
    excluded=vcat(FIRST_ORDER, [:hvp, :second_derivative]),
    logging=true,
)
