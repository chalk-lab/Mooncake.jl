using Pkg, Test
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Mooncake: Mooncake

test_differentiation(
    [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())];
    excluded=SECOND_ORDER,
    logging=true,
)
