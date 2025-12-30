using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake

function DifferentiationInterface.inner_preparation_behavior(::AutoMooncakeForward)
    DifferentiationInterface.PrepareInnerSimple()
end

# Test second-order differentiation (forward-over-reverse)
test_differentiation(
    [SecondOrder(AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing))];
    excluded=[FIRST_ORDER..., :hvp, :hessian], # testing only :second_derivative
    logging=true,
)
