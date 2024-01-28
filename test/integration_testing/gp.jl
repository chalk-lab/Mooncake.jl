using AbstractGPs, KernelFunctions

@testset "gp" begin
    interp = Taped.TInterp()
    base_kernels = Any[
        ZeroKernel(),
        ConstantKernel(; c=1.0),
        SEKernel(),
        Matern12Kernel(),
        Matern32Kernel(),
        Matern52Kernel(),
        LinearKernel(),
        PolynomialKernel(; degree=2, c=0.5),
    ]
    simple_xs = Any[
        randn(10),
        randn(1),
        range(0.0; step=0.1, length=11),
        ColVecs(randn(2, 11)),
        RowVecs(randn(9, 4)),
    ]
    d_2_xs = Any[ColVecs(randn(2, 11)), RowVecs(randn(9, 2))]
    @testset "kernelmatrix_diag $k, $(typeof(x1))" for (k, x1) in vcat(
        Any[(k, x) for k in base_kernels for x in simple_xs],
        Any[(with_lengthscale(k, 1.1), x) for k in base_kernels for x in simple_xs],
        Any[(with_lengthscale(k, rand(2)), x) for k in base_kernels for x in d_2_xs],
        Any[(k ∘ LinearTransform(randn(2, 2)), x) for k in base_kernels for x in d_2_xs],
        Any[
            (k ∘ LinearTransform(Diagonal(randn(2))), x) for
                k in base_kernels for x in d_2_xs
        ],
    )
        @info typeof(k), typeof(x1)
        @testset "ternary kernelmatrix" begin
            f = kernelmatrix
            x = (k, x1, x1)
            sig = Tuple{typeof(f), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, f, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
        @testset "ternary kernelmatrix_diag" begin
            f = kernelmatrix_diag
            x = (k, x1, x1)
            sig = Tuple{typeof(f), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, f, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
        @testset "binary kernelmatrix" begin
            f = kernelmatrix
            x = (k, x1)
            sig = Tuple{typeof(f), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, f, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
        @testset "binary kernelmatrix_diag" begin
            f = kernelmatrix_diag
            x = (k, x1)
            sig = Tuple{typeof(f), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, f, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
        @testset "rand" begin
            x = (Xoshiro(123546), GP(k)(x1, 1.1))
            sig = Tuple{typeof(rand), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, rand, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
        @testset "logpdf" begin
            fx = GP(k)(x1, 1.1)
            x = (fx, rand(fx))
            sig = Tuple{typeof(logpdf), map(typeof, x)...}
            in_f = Taped.InterpretedFunction(DefaultCtx(), sig, interp)
            TestUtils.test_rrule!!(
                sr(123456), in_f, logpdf, x...;
                perf_flag=:none, interface_only=true, is_primitive=false,
            )
        end
    end
end
