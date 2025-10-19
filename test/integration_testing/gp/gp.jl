using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AbstractGPs, KernelFunctions, LinearAlgebra, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@testset "gp" begin
    rng = StableRNG(123456)
    ks = Any[
        ZeroKernel(),
        ConstantKernel(; c=1.0),
        SEKernel(),
        Matern12Kernel(),
        Matern32Kernel(),
        Matern52Kernel(),
        LinearKernel(),
        PolynomialKernel(; degree=2, c=0.5),
    ]
    xs = Any[
        (randn(rng, 10), randn(rng, 10)),
        (randn(rng, 1), randn(rng, 1)),
        (ColVecs(randn(rng, 2, 11)), ColVecs(randn(rng, 2, 11))),
        (RowVecs(randn(rng, 3, 4)), RowVecs(randn(rng, 3, 4))),
    ]
    d_2_xs = Any[
        (ColVecs(randn(rng, 2, 11)), ColVecs(randn(rng, 2, 11))),
        (RowVecs(randn(rng, 9, 2)), RowVecs(randn(rng, 9, 2))),
    ]
    @testset "$k, $(typeof(x1))" for (k, x1, x2) in vcat(
        Any[(k, x1, x2) for k in ks for (x1, x2) in xs],
        Any[(with_lengthscale(k, 1.1), x1, x2) for k in ks for (x1, x2) in xs],
        Any[(with_lengthscale(k, rand(rng, 2)), x1, x2) for k in ks for (x1, x2) in d_2_xs],
        Any[
            (k ∘ LinearTransform(randn(rng, 2, 2)), x1, x2) for k in ks for
            (x1, x2) in d_2_xs
        ],
        Any[
            (k ∘ LinearTransform(Diagonal(randn(rng, 2))), x1, x2) for k in ks for
            (x1, x2) in d_2_xs
        ],
    )
        fx = GP(k)(x1, 1.1)
        @testset "$(typeof(x))" for x in Any[
            (kernelmatrix, k, x1, x2),
            (kernelmatrix_diag, k, x1, x2),
            (kernelmatrix, k, x1),
            (kernelmatrix_diag, k, x1),
            (fx -> rand(StableRNG(123456), fx), fx),
            (logpdf, fx, rand(rng, fx)),
        ]
            @info typeof(x)
            test_rule(rng, x...; is_primitive=false, unsafe_perturb=true)
        end
    end
end
