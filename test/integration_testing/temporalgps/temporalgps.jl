using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AbstractGPs, KernelFunctions, Mooncake, StableRNGs, TemporalGPs, Test

build_gp(k) = to_sde(GP(k), SArrayStorage(Float64))

temporalgps_logpdf_tester(k, x, y, s) = logpdf(build_gp(k)(x, s), y)

@testset "temporalgps" begin
    xs = Any[
        collect(range(-5.0; step=0.1, length=1_000)),
        RegularSpacing(0.0, 0.1, 1_000),
        range(-5.0; step=0.1, length=1_000),
    ]
    base_kernels = Any[Matern12Kernel(), Matern32Kernel()]
    kernels = vcat(base_kernels, [with_lengthscale(k, 1.1) for k in base_kernels])
    @testset "$k, $(typeof(x))" for (k, x) in vec(collect(Iterators.product(kernels, xs)))
        s = 1.0
        y = rand(build_gp(k)(x, s))
        f = temporalgps_logpdf_tester
        sig = typeof((temporalgps_logpdf_tester, k, x, y, s))
        @info "$sig"
        Mooncake.TestUtils.test_rule(StableRNG(123456), f, k, x, y, s; is_primitive=false)
    end
end
