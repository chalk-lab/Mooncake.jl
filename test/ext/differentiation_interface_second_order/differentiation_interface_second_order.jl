using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

EXCLUDED = @static if VERSION ≥ v"1.11-" && VERSION ≤ v"1.12-"
    # testing only :hessian on 1.11 due to opaque closure bug.
    # This is potentially the same issue as discussed in
    # https://github.com/chalk-lab/MistyClosures.jl/pull/12#issue-3278662295
    [FIRST_ORDER..., :hvp, :second_derivative]
else
    [FIRST_ORDER...]
end

# Test second-order differentiation (forward-over-reverse)
test_differentiation(
    [
        SecondOrder(
            AutoMooncakeForward(),
            AutoMooncake(; config=Mooncake.Config(; maybeinline_primitive=false)),
        ),
    ];
    excluded=EXCLUDED,
    logging=true,
)

# Test for world-age fix when using closures (#916, #632)
# The bug occurs when:
# 1. prepare_hessian creates cached rules with MistyClosures
# 2. A closure is defined that captures the prep (advances world age)
# 3. The closure is called, triggering _dual_mc with stale world age
# See also: test/rules/misty_closures.jl for a unit test of the fix.
module TestWorldAge
function gams_objective(x)
    #! format: off
    objvar = (((((((((((((((((((((((((((x[1] * x[1] + x[10] * x[10]) * (x[1] * x[1] + x[10] * x[10]) - 4 * x[1]) + 3) + (x[2] * x[2] + x[10] * x[10]) * (x[2] * x[2] + x[10] * x[10])) - 4 * x[2]) + 3) + (x[3] * x[3] + x[10] * x[10]) * (x[3] * x[3] + x[10] * x[10])) - 4 * x[3]) + 3) + (x[4] * x[4] + x[10] * x[10]) * (x[4] * x[4] + x[10] * x[10])) - 4 * x[4]) + 3) + (x[5] * x[5] + x[10] * x[10]) * (x[5] * x[5] + x[10] * x[10])) - 4 * x[5]) + 3) + (x[6] * x[6] + x[10] * x[10]) * (x[6] * x[6] + x[10] * x[10])) - 4 * x[6]) + 3) + (x[7] * x[7] + x[10] * x[10]) * (x[7] * x[7] + x[10] * x[10])) - 4 * x[7]) + 3) + (x[8] * x[8] + x[10] * x[10]) * (x[8] * x[8] + x[10] * x[10])) - 4 * x[8]) + 3) + (x[9] * x[9] + x[10] * x[10]) * (x[9] * x[9] + x[10] * x[10])) - 4 * x[9]) + 3) - 0
    #! format: on
    return objvar
end
end

@static if VERSION > v"1.12-"
    const DI = DifferentiationInterface
    # Ensure MistyClosures are created at execution time to trigger the world-age issue.
    DI.inner_preparation_behavior(::AutoMooncakeForward) = DI.DontPrepareInner()
    @testset "world age fix with closure (#916)" begin
        x0 = [0.0; fill(1.0, 9)]
        f = TestWorldAge.gams_objective

        backend = SecondOrder(
            AutoMooncakeForward(),
            AutoMooncake(; config=Mooncake.Config(; maybeinline_primitive=false)),
        )
        preph = prepare_hessian(f, backend, x0)

        # Wrapping in a closure triggers the world-age bug without the fix
        ∇²f(x) = hessian(f, preph, backend, x)

        # This should not throw a world-age error
        @test ∇²f(x0) isa Matrix
    end
end
