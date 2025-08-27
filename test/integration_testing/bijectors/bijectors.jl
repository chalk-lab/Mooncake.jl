using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Bijectors, LinearAlgebra, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

"""
Type for specifying a test case for `test_rule`.
"""
struct TestCase
    func::Function
    arg::Any
    name::Union{String,Nothing}
    broken::Bool
end

TestCase(f, arg; name=nothing, broken=false) = TestCase(f, arg, name, broken)

"""
A helper function that returns a TestCase that evaluates bijector(inverse(bijector)(x))
"""
function b_binv_test_case(bijector, dim; name=nothing, rng=StableRNG(23))
    if name === nothing
        name = string(bijector)
    end
    return TestCase(x -> bijector(inverse(bijector)(x)), randn(rng, dim); name)
end

@testset "Bijectors integration tests" begin
    test_cases = TestCase[
        b_binv_test_case(Bijectors.VecCorrBijector(), 3),
        b_binv_test_case(Bijectors.VecCorrBijector(), 0),
        b_binv_test_case(Bijectors.CorrBijector(), (3, 3)),
        b_binv_test_case(Bijectors.CorrBijector(), (0, 0)),
        b_binv_test_case(Bijectors.VecCholeskyBijector(:L), 3),
        b_binv_test_case(Bijectors.VecCholeskyBijector(:L), 0),
        b_binv_test_case(Bijectors.VecCholeskyBijector(:U), 3),
        b_binv_test_case(Bijectors.VecCholeskyBijector(:U), 0),
        b_binv_test_case(
            Bijectors.Coupling(Bijectors.Shift, Bijectors.PartitionMask(3, [1], [2])), 3
        ),
        b_binv_test_case(Bijectors.InvertibleBatchNorm(3; eps=1e-5, mtm=1e-1), (3, 3)),
        b_binv_test_case(Bijectors.LeakyReLU(0.2), 3),
        b_binv_test_case(Bijectors.Logit(0.1, 0.3), 3),
        b_binv_test_case(Bijectors.PDBijector(), (3, 3)),
        b_binv_test_case(Bijectors.PDVecBijector(), 3),
        b_binv_test_case(Bijectors.Permute([
            0 1 0
            1 0 0
            0 0 1
        ]), (3, 3)),
        # b_binv_test_case(Bijectors.PlanarLayer(3), (3, 3)),
        b_binv_test_case(Bijectors.RadialLayer(3), 3),
        b_binv_test_case(Bijectors.Reshape((2, 3), (3, 2)), (2, 3)),
        b_binv_test_case(Bijectors.Scale(0.2), 3),
        b_binv_test_case(Bijectors.Shift(-0.4), 3),
        b_binv_test_case(Bijectors.SignFlip(), 3),
        b_binv_test_case(Bijectors.SimplexBijector(), 3),
        b_binv_test_case(Bijectors.TruncatedBijector(-0.2, 0.5), 3),

        # Below, some test cases that don't fit the b_binv_test_case mold.

        TestCase(
            function (x)
                b = Bijectors.RationalQuadraticSpline(
                    [-0.2, 0.1, 0.5], [-0.3, 0.3, 0.9], [1.0, 0.2, 1.0]
                )
                binv = Bijectors.inverse(b)
                return binv(b(x))
            end,
            randn(StableRNG(23));
            name="RationalQuadraticSpline on scalar",
        ),
        TestCase(
            function (x)
                b = Bijectors.OrderedBijector()
                binv = Bijectors.inverse(b)
                return binv(b(x))
            end,
            randn(StableRNG(23), 7);
            name="OrderedBijector",
        ),
        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(
                    Bijectors.MvNormal(zeros(2), LinearAlgebra.I), layer
                )
                x = x[6:7]
                return Bijectors.logpdf(flow.dist, x) -
                       Bijectors.logabsdetjac(flow.transform, x)
            end,
            randn(StableRNG(23), 7);
            name="PlanarLayer7",
        ),
        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(
                    Bijectors.MvNormal(zeros(2), LinearAlgebra.I), layer
                )
                x = reshape(x[6:end], 2, :)
                return sum(
                    Bijectors.logpdf(flow.dist, x) -
                    Bijectors.logabsdetjac(flow.transform, x),
                )
            end,
            randn(StableRNG(23), 11);
            name="PlanarLayer11",
        ),
    ]

    @testset "$(c.name)" for c in test_cases
        if c.broken
            @test_broken begin
                test_rule(StableRNG(123456), c.func, c.arg; is_primitive=false)
                true
            end
        else
            rng = StableRNG(123456)
            test_rule(rng, c.func, c.arg; is_primitive=false, unsafe_perturb=true)
        end
    end
end
