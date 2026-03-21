using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck,
    JET, Distributions, FillArrays, Mooncake, LinearAlgebra, PDMats, StableRNGs, Test

using Mooncake.TestUtils: test_rule

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)
sr(n::Int) = StableRNG(n)

@testset "distributions" begin
    logpdf_test_cases = Any[

        #
        # Univariate
        #

        (:allocs, Arcsine(), 0.5),
        (:allocs, Arcsine(-0.3, 0.9), 0.5),
        (:allocs, Arcsine(0.5, 1.1), 1.0),
        (:allocs, Beta(1.1, 1.1), 0.5),
        (:allocs, Beta(1.1, 1.5), 0.9),
        (:allocs, Beta(1.6, 1.5), 0.5),
        (:allocs, BetaPrime(1.1, 1.1), 0.5),
        (:allocs, BetaPrime(1.1, 1.6), 0.5),
        (:allocs, BetaPrime(1.6, 1.3), 0.9),
        (:allocs, Biweight(1.0, 2.0), 0.5),
        (:allocs, Biweight(-0.5, 2.5), -0.45),
        (:allocs, Biweight(0.0, 1.0), 0.3),
        (:allocs, Cauchy(), -0.5),
        (:allocs, Cauchy(1.0), 0.99),
        (:allocs, Cauchy(1.0, 0.1), 1.01),
        (:allocs, Chi(2.5), 0.5),
        (:allocs, Chi(5.5), 1.1),
        (:allocs, Chi(0.1), 0.7),
        (:allocs, Chisq(2.5), 0.5),
        (:allocs, Chisq(5.5), 1.1),
        (:allocs, Chisq(0.1), 0.7),
        (:allocs, Cosine(0.0, 1.0), 0.5),
        (:allocs, Cosine(-0.5, 2.0), -0.1),
        (:allocs, Cosine(0.4, 0.5), 0.0),
        (:allocs, Epanechnikov(0.0, 1.0), 0.5),
        (:allocs, Epanechnikov(-0.5, 1.2), -0.9),
        (:allocs, Epanechnikov(-0.4, 1.6), 0.1),
        (:allocs, Erlang(), 0.5),
        (:allocs, Erlang(), 0.1),
        (:allocs, Erlang(), 0.9),
        (:allocs, Exponential(), 0.1),
        (:allocs, Exponential(0.5), 0.9),
        (:allocs, Exponential(1.4), 0.05),
        (:allocs, FDist(2.1, 3.5), 0.7),
        (:allocs, FDist(1.4, 5.4), 3.5),
        (:allocs, FDist(5.5, 3.3), 7.2),
        (:allocs, Frechet(), 0.1),
        (:allocs, Frechet(), 1.1),
        (:allocs, Frechet(1.5, 2.4), 0.1),
        (:allocs, Gamma(0.9, 1.2), 4.5),
        (:allocs, Gamma(0.5, 1.9), 1.5),
        (:allocs, Gamma(1.8, 3.2), 1.0),
        (:allocs, GeneralizedExtremeValue(0.3, 1.3, 0.1), 2.4),
        (:allocs, GeneralizedExtremeValue(-0.7, 2.2, 0.4), 1.1),
        (:allocs, GeneralizedExtremeValue(0.5, 0.9, -0.5), -7.0),
        (:allocs, GeneralizedPareto(0.3, 1.1, 1.1), 5.0),
        (:allocs, GeneralizedPareto(-0.25, 0.9, 0.1), 0.8),
        (:allocs, GeneralizedPareto(0.3, 1.1, -5.1), 0.31),
        (:allocs, Gumbel(0.1, 0.5), 0.1),
        (:allocs, Gumbel(-0.5, 1.1), -0.1),
        (:allocs, Gumbel(0.3, 0.1), 0.3),
        (:allocs, InverseGaussian(0.1, 0.5), 1.1),
        (:allocs, InverseGaussian(0.2, 1.1), 3.2),
        (:allocs, InverseGaussian(0.1, 1.2), 0.5),
        (:allocs, JohnsonSU(0.1, 0.95, 0.1, 1.1), 0.1),
        (:allocs, JohnsonSU(0.15, 0.9, 0.12, 0.94), 0.5),
        (:allocs, JohnsonSU(0.1, 0.95, 0.1, 1.1), -0.3),
        (:allocs, Kolmogorov(), 1.1),
        (:allocs, Kolmogorov(), 0.9),
        (:allocs, Kolmogorov(), 1.5),
        (:allocs, Kumaraswamy(2.0, 5.0), 0.71),
        (:allocs, Kumaraswamy(0.1, 5.0), 0.2),
        (:allocs, Kumaraswamy(0.5, 4.5), 0.1),
        (:allocs, Laplace(0.1, 1.0), 0.2),
        (:allocs, Laplace(-0.5, 2.1), 0.5),
        (:allocs, Laplace(-0.35, 0.4), -0.3),
        (:allocs, Levy(0.1, 0.9), 4.1),
        (:allocs, Levy(0.5, 0.9), 0.6),
        (:allocs, Levy(1.1, 0.5), 2.2),
        (:allocs, Lindley(0.5), 2.1),
        (:allocs, Lindley(1.1), 3.1),
        (:allocs, Lindley(1.9), 3.5),
        (:allocs, Logistic(0.1, 1.2), 1.1),
        (:allocs, Logistic(0.5, 0.7), 0.6),
        (:allocs, Logistic(-0.5, 0.1), -0.4),
        (:allocs, LogitNormal(0.1, 1.1), 0.5),
        (:allocs, LogitNormal(0.5, 0.7), 0.6),
        (:allocs, LogitNormal(-0.12, 1.1), 0.1),
        (:allocs, LogNormal(0.0, 1.0), 0.5),
        (:allocs, LogNormal(0.5, 1.0), 0.5),
        (:allocs, LogNormal(-0.1, 1.3), 0.75),
        (:allocs, LogUniform(0.1, 0.9), 0.75),
        (:allocs, LogUniform(0.15, 7.8), 7.1),
        (:allocs, LogUniform(2.0, 3.0), 2.1),
        # (:none, NoncentralBeta(1.1, 1.1, 1.2), 0.8), # foreigncall (Rmath.dnbeta). Not implemented anywhere.
        # (:none, NoncentralChisq(2, 3.0), 10.0), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        # (:none, NoncentralF(2, 3, 1.1), 4.1), # foreigncall (Rmath.dnf). Not implemented anywhere.
        # (:none, NoncentralT(1.3, 1.1), 0.1), # foreigncall (Rmath.dnt). Not implemented anywhere.
        (:allocs, Normal(), 0.1),
        (:allocs, Normal(0.0, 1.0), 1.0),
        (:allocs, Normal(0.5, 1.0), 0.05),
        (:allocs, Normal(0.0, 1.5), -0.1),
        (:allocs, Normal(-0.1, 0.9), -0.3),
        # (:none NormalInverseGaussian(0.0, 1.0, 0.2, 0.1), 0.1), # foreigncall -- https://github.com/JuliaMath/SpecialFunctions.jl/blob/be1fa06fee58ec019a28fb0cd2b847ca83a5af9a/src/bessel.jl#L265
        (:allocs, Pareto(1.0, 1.0), 3.5),
        (:allocs, Pareto(1.1, 0.9), 3.1),
        (:allocs, Pareto(1.0, 1.0), 1.4),
        (:allocs, PGeneralizedGaussian(0.2), 5.0),
        (:allocs, PGeneralizedGaussian(0.5, 1.0, 0.3), 5.0),
        (:allocs, PGeneralizedGaussian(-0.1, 11.1, 6.5), -0.3),
        (:allocs, Rayleigh(0.5), 0.6),
        (:allocs, Rayleigh(0.9), 1.1),
        (:allocs, Rayleigh(0.55), 0.63),
        # (:none, Rician(0.5, 1.0), 2.1), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        (:allocs, Semicircle(1.0), 0.9),
        (:allocs, Semicircle(5.1), 5.05),
        (:allocs, Semicircle(0.5), -0.1),
        (:allocs, SkewedExponentialPower(0.1, 1.0, 0.97, 0.7), -2.0),
        (:allocs, SkewedExponentialPower(0.15, 1.0, 0.97, 0.7), -2.0),
        (:allocs, SkewedExponentialPower(0.1, 1.1, 0.99, 0.7), 0.5),
        (:allocs, SkewNormal(0.0, 1.0, -1.0), 0.1),
        (:allocs, SkewNormal(0.5, 2.0, 1.1), 0.1),
        (:allocs, SkewNormal(-0.5, 1.0, 0.0), 0.1),
        (:allocs, SymTriangularDist(0.0, 1.0), 0.5),
        (:allocs, SymTriangularDist(-0.5, 2.1), -2.0),
        (:allocs, SymTriangularDist(1.7, 0.3), 1.75),
        (:allocs, TDist(1.1), 99.1),
        (:allocs, TDist(10.1), 25.0),
        (:allocs, TDist(2.1), -89.5),
        (:allocs, TriangularDist(0.0, 1.5, 0.5), 0.45),
        (:allocs, TriangularDist(0.1, 1.4, 0.45), 0.12),
        (:allocs, TriangularDist(0.0, 1.5, 0.5), 0.2),
        (:allocs, Triweight(1.0, 1.0), 1.0),
        (:allocs, Triweight(1.1, 2.1), 1.0),
        (:allocs, Triweight(1.9, 10.0), -0.1),
        (:allocs, Uniform(0.0, 1.0), 0.2),
        (:allocs, Uniform(-0.1, 1.1), 1.0),
        (:allocs, Uniform(99.5, 100.5), 100.0),
        (:allocs, VonMises(0.5), 0.1),
        (:allocs, VonMises(0.3), -0.1),
        (:allocs, VonMises(0.2), -0.5),
        (:allocs, Weibull(0.5, 1.0), 0.45),
        (:allocs, Weibull(0.3, 1.1), 0.66),
        (:allocs, Weibull(0.75, 1.3), 0.99),

        #
        # Multivariate
        #

        (:allocs, MvNormal(Diagonal(Fill(1.5, 1))), [-0.3]),
        (:allocs, MvNormal(Diagonal(Fill(0.5, 2))), [0.2, -0.3]),
        (:none, MvNormal([0.0], 0.9), [0.1]),
        (:none, MvNormal([0.0, 0.1], 0.9), [0.1, -0.05]),
        (:allocs, MvNormal(Diagonal([0.1])), [0.1]),
        (:allocs, MvNormal(Diagonal([0.1, 0.2])), [0.1, 0.15]),
        (:none, MvNormal([0.1, -0.3], Diagonal(Fill(0.9, 2))), [0.1, -0.1]),
        (:none, MvNormal([0.1, -0.1], 0.4I), [-0.1, 0.15]),
        (:none, MvNormal([0.2, 0.3], Hermitian(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (:none, MvNormal([0.2, 0.3], Symmetric(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (:none, MvNormal([0.2, 0.3], Diagonal([0.5, 0.4])), [-0.1, 0.05]),
        (:none, MvNormal([-0.15], _pdmat([1.1]')), [-0.05]),
        (:none, MvNormal([0.2, -0.15], _pdmat([1.0 0.9; 0.7 1.1])), [0.05, -0.05]),
        (:none, MvNormal([0.2, -0.3], [0.5, 0.6]), [0.4, -0.3]),
        (:none, MvNormalCanon([0.1, -0.1], _pdmat([0.5 0.4; 0.45 1.0])), [0.2, -0.25]),
        (:none, MvLogNormal(MvNormal([0.2, -0.1], _pdmat([1.0 0.9; 0.7 1.1]))), [0.5, 0.1]),
        (:none, product_distribution([Normal()]), [0.3]),
        (:none, product_distribution([Normal(), Uniform()]), [-0.4, 0.3]),

        #
        # Matrix-variate
        #

        (
            :none,
            MatrixNormal(
                randn(sr(0), 2, 3), _pdmat(randn(sr(1), 2, 2)), _pdmat(randn(sr(2), 3, 3))
            ),
            randn(sr(4), 2, 3),
        ),
        (
            :none,
            Wishart(5, _pdmat(randn(sr(5), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(6), 3, 3)))),
        ),
        (
            :none,
            InverseWishart(5, _pdmat(randn(sr(7), 3, 3))),
            Symmetric(collect(_pdmat(randn(sr(8), 3, 3)))),
        ),
        (
            :none,
            MatrixTDist(
                3.1,
                randn(sr(9), 2, 3),
                _pdmat(randn(sr(0), 2, 2)),
                _pdmat(randn(sr(1), 3, 3)),
            ),
            randn(sr(2), 2, 3),
        ),
        (:none, MatrixBeta(5, 9.0, 10.0), rand(sr(123456), MatrixBeta(5, 9.0, 10.0))),
        (
            :none,
            MatrixFDist(6.0, 7.0, _pdmat(randn(sr(1234), 5, 5))),
            rand(sr(13), MatrixFDist(6.0, 7.0, _pdmat(randn(StableRNG(11), 5, 5)))),
        ),
        (:none, LKJ(5, 1.1), rand(sr(123456), LKJ(5, 1.1))),
    ]
    work_around_test_cases = Any[
        (
            :allocs,
            "InverseGamma",
            (a, b, x) -> logpdf(InverseGamma(a, b), x),
            (1.5, 1.4, 0.4),
        ),
        (
            :allocs,
            "NormalCanon",
            (m, s, x) -> logpdf(NormalCanon(m, s), x),
            (0.1, 1.0, -0.5),
        ),
        (:none, "Categorical", x -> logpdf(Categorical(x, 1 - x), 1), 0.3),
        (
            :none,
            "MvLogitNormal",
            (m, S, x) -> logpdf(MvLogitNormal(m, S), vcat(x, 1 - sum(x))),
            ([0.4, 0.6], Symmetric(_pdmat([0.9 0.4; 0.5 1.1])), [0.27, 0.24]),
        ),
        (
            :allocs,
            "truncated Beta",
            (a, b, α, β, x) -> logpdf(truncated(Beta(α, β), a, b), x),
            (0.1, 0.9, 1.1, 1.3, 0.4),
        ),
        (
            :none,
            "truncated Normal",
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1),
        ),
        (
            :none,
            "truncated Uniform",
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4),
        ),
        (
            :none,
            "left-truncated Beta",
            (a, α, β, x) -> logpdf(truncated(Beta(α, β); lower=a), x),
            (0.1, 1.1, 1.3, 0.4),
        ),
        (:none, "Dirichlet", (a, x) -> logpdf(Dirichlet(a), [x, 1 - x]), ([1.5, 1.1], 0.6)),
        (
            :none,
            "reshape",
            x -> logpdf(reshape(product_distribution([Normal(), Uniform()]), 1, 2), x),
            ([2.1 0.7],),
        ),
        (:none, "vec", x -> logpdf(vec(LKJ(2, 1.1)), x), ([1.0, 0.489, 0.489, 1.0],)),
        (
            :none,
            "LKJCholesky",
            function (X, v)
                # LKJCholesky distributes over the Cholesky factorisation of correlation
                # matrices, so the argument to `logpdf` must be such a matrix.
                S = X'X
                Λ = Diagonal(map(inv ∘ sqrt, diag(S)))
                C = cholesky(Symmetric(Λ * S * Λ))
                return logpdf(LKJCholesky(2, v), C)
            end,
            (randn(2, 2), 1.1),
        ),
    ]

    @testset "$(typeof(d))" for (perf_flag, d, x) in logpdf_test_cases
        @info "$(map(typeof, (d, x)))"
        test_rule(StableRNG(123546), logpdf, d, x; perf_flag, is_primitive=false)
    end

    @testset "$name" for (perf_flag, name, f, x) in work_around_test_cases
        @info "$name"
        test_rule(StableRNG(123456), f, x...; perf_flag, is_primitive=false)
    end
end
