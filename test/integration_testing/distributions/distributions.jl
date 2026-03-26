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
            :none,
            "NormalInverseGaussian",
            (μ, α, β, δ, x) -> logpdf(NormalInverseGaussian(μ, α, β, δ), x),
            (0.0, 1.0, 0.2, 0.1, 0.1),
        ),
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

    # ── nforward tests ────────────────────────────────────────────────────────────────────────
    # These test logpdf derivatives via nforward (chunked NDual forward sweeps).
    # Each case differentiates w.r.t. scalar constructor parameters and/or the observation,
    # which is a different code path from the standard Mooncake AD tests above.
    # See src/interpreter/nforward.jl for the nforward API.
    #
    # Limitations / workarounds are documented inline:
    #   • Erlang: integer shape k is non-differentiable; x-only differentiation.
    #   • PDMat-based covariances: use Symmetric instead of PDMat (PDMat doesn't support NDual).
    #   • product_distribution components: Distribution objects are not NDual-parameterised.
    #   • Matrix-variate distributions: PDMat scale params don't accept NDual; obs-only.
    #   • truncated Beta shape params: ∂I_x/∂a, ∂I_x/∂b not implemented; bounds+x only.
    #   • LKJCholesky observation: pass lower-triangular factor L as plain Matrix, reconstruct
    #     Cholesky inside the lambda (Cholesky constructor requires numeric matrix, not NDual).

    _NForwardMode(f, args, C) = Mooncake.nforward_build_rrule(f, args...; chunk_size=C)

    nforward_test_cases = Any[

        # ── Univariate ────────────────────────────────────────────────────────────

        ("Arcsine() 1", x -> logpdf(Arcsine(), x), (0.5,), 1),
        ("Arcsine(a,b) 1", (a, b, x) -> logpdf(Arcsine(a, b), x), (-0.3, 0.9, 0.5), 3),
        ("Arcsine(a,b) 2", (a, b, x) -> logpdf(Arcsine(a, b), x), (0.5, 1.1, 1.0), 3),
        ("Beta 1", (α, β, x) -> logpdf(Beta(α, β), x), (1.1, 1.1, 0.5), 3),
        ("Beta 2", (α, β, x) -> logpdf(Beta(α, β), x), (1.1, 1.5, 0.9), 3),
        ("Beta 3", (α, β, x) -> logpdf(Beta(α, β), x), (1.6, 1.5, 0.5), 3),
        ("BetaPrime 1", (α, β, x) -> logpdf(BetaPrime(α, β), x), (1.1, 1.1, 0.5), 3),
        ("BetaPrime 2", (α, β, x) -> logpdf(BetaPrime(α, β), x), (1.1, 1.6, 0.5), 3),
        ("BetaPrime 3", (α, β, x) -> logpdf(BetaPrime(α, β), x), (1.6, 1.3, 0.9), 3),
        ("Biweight 1", (μ, σ, x) -> logpdf(Biweight(μ, σ), x), (1.0, 2.0, 0.5), 3),
        ("Biweight 2", (μ, σ, x) -> logpdf(Biweight(μ, σ), x), (-0.5, 2.5, -0.45), 3),
        ("Biweight 3", (μ, σ, x) -> logpdf(Biweight(μ, σ), x), (0.0, 1.0, 0.3), 3),
        ("Cauchy() 1", x -> logpdf(Cauchy(), x), (-0.5,), 1),
        ("Cauchy(μ) 1", (μ, x) -> logpdf(Cauchy(μ), x), (1.0, 0.99), 2),
        ("Cauchy(μ,σ) 1", (μ, σ, x) -> logpdf(Cauchy(μ, σ), x), (1.0, 0.1, 1.01), 3),
        ("Chi 1", (ν, x) -> logpdf(Chi(ν), x), (2.5, 0.5), 2),
        ("Chi 2", (ν, x) -> logpdf(Chi(ν), x), (5.5, 1.1), 2),
        ("Chi 3", (ν, x) -> logpdf(Chi(ν), x), (0.1, 0.7), 2),
        ("Chisq 1", (ν, x) -> logpdf(Chisq(ν), x), (2.5, 0.5), 2),
        ("Chisq 2", (ν, x) -> logpdf(Chisq(ν), x), (5.5, 1.1), 2),
        ("Chisq 3", (ν, x) -> logpdf(Chisq(ν), x), (0.1, 0.7), 2),
        ("Cosine 1", (μ, σ, x) -> logpdf(Cosine(μ, σ), x), (0.0, 1.0, 0.5), 3),
        ("Cosine 2", (μ, σ, x) -> logpdf(Cosine(μ, σ), x), (-0.5, 2.0, -0.1), 3),
        ("Cosine 3", (μ, σ, x) -> logpdf(Cosine(μ, σ), x), (0.4, 0.5, 0.0), 3),
        ("Epanechnikov 1", (μ, σ, x) -> logpdf(Epanechnikov(μ, σ), x), (0.0, 1.0, 0.5), 3),
        (
            "Epanechnikov 2",
            (μ, σ, x) -> logpdf(Epanechnikov(μ, σ), x),
            (-0.5, 1.2, -0.9),
            3,
        ),
        ("Epanechnikov 3", (μ, σ, x) -> logpdf(Epanechnikov(μ, σ), x), (-0.4, 1.6, 0.1), 3),

        # Erlang — x-only differentiation; integer shape k is non-differentiable.
        # Erlang(k, θ) requires k ∈ ℤ₊, so NDual cannot be passed as the shape argument.
        ("Erlang() 1", x -> logpdf(Erlang(), x), (0.5,), 1),
        ("Erlang() 2", x -> logpdf(Erlang(), x), (0.1,), 1),
        ("Erlang() 3", x -> logpdf(Erlang(), x), (0.9,), 1),
        ("Exponential() 1", x -> logpdf(Exponential(), x), (0.1,), 1),
        ("Exponential(θ) 1", (θ, x) -> logpdf(Exponential(θ), x), (0.5, 0.9), 2),
        ("Exponential(θ) 2", (θ, x) -> logpdf(Exponential(θ), x), (1.4, 0.05), 2),
        ("FDist 1", (ν1, ν2, x) -> logpdf(FDist(ν1, ν2), x), (2.1, 3.5, 0.7), 3),
        ("FDist 2", (ν1, ν2, x) -> logpdf(FDist(ν1, ν2), x), (1.4, 5.4, 3.5), 3),
        ("FDist 3", (ν1, ν2, x) -> logpdf(FDist(ν1, ν2), x), (5.5, 3.3, 7.2), 3),
        ("Frechet() 1", x -> logpdf(Frechet(), x), (0.1,), 1),
        ("Frechet() 2", x -> logpdf(Frechet(), x), (1.1,), 1),
        ("Frechet(α,θ) 1", (α, θ, x) -> logpdf(Frechet(α, θ), x), (1.5, 2.4, 0.1), 3),
        ("Gamma 1", (α, θ, x) -> logpdf(Gamma(α, θ), x), (0.9, 1.2, 4.5), 3),
        ("Gamma 2", (α, θ, x) -> logpdf(Gamma(α, θ), x), (0.5, 1.9, 1.5), 3),
        ("Gamma 3", (α, θ, x) -> logpdf(Gamma(α, θ), x), (1.8, 3.2, 1.0), 3),
        (
            "GeneralizedExtremeValue 1",
            (μ, σ, ξ, x) -> logpdf(GeneralizedExtremeValue(μ, σ, ξ), x),
            (0.3, 1.3, 0.1, 2.4),
            4,
        ),
        (
            "GeneralizedExtremeValue 2",
            (μ, σ, ξ, x) -> logpdf(GeneralizedExtremeValue(μ, σ, ξ), x),
            (-0.7, 2.2, 0.4, 1.1),
            4,
        ),
        (
            "GeneralizedExtremeValue 3",
            (μ, σ, ξ, x) -> logpdf(GeneralizedExtremeValue(μ, σ, ξ), x),
            (0.5, 0.9, -0.5, -7.0),
            4,
        ),
        (
            "GeneralizedPareto 1",
            (μ, σ, ξ, x) -> logpdf(GeneralizedPareto(μ, σ, ξ), x),
            (0.3, 1.1, 1.1, 5.0),
            4,
        ),
        (
            "GeneralizedPareto 2",
            (μ, σ, ξ, x) -> logpdf(GeneralizedPareto(μ, σ, ξ), x),
            (-0.25, 0.9, 0.1, 0.8),
            4,
        ),
        (
            "GeneralizedPareto 3",
            (μ, σ, ξ, x) -> logpdf(GeneralizedPareto(μ, σ, ξ), x),
            (0.3, 1.1, -5.1, 0.31),
            4,
        ),
        ("Gumbel 1", (μ, σ, x) -> logpdf(Gumbel(μ, σ), x), (0.1, 0.5, 0.1), 3),
        ("Gumbel 2", (μ, σ, x) -> logpdf(Gumbel(μ, σ), x), (-0.5, 1.1, -0.1), 3),
        ("Gumbel 3", (μ, σ, x) -> logpdf(Gumbel(μ, σ), x), (0.3, 0.1, 0.3), 3),
        ("InverseGamma 1", (a, b, x) -> logpdf(InverseGamma(a, b), x), (1.5, 1.4, 0.4), 3),
        (
            "InverseGaussian 1",
            (μ, λ, x) -> logpdf(InverseGaussian(μ, λ), x),
            (0.1, 0.5, 1.1),
            3,
        ),
        (
            "InverseGaussian 2",
            (μ, λ, x) -> logpdf(InverseGaussian(μ, λ), x),
            (0.2, 1.1, 3.2),
            3,
        ),
        (
            "InverseGaussian 3",
            (μ, λ, x) -> logpdf(InverseGaussian(μ, λ), x),
            (0.1, 1.2, 0.5),
            3,
        ),
        (
            "JohnsonSU 1",
            (γ, δ, ξ, λ, x) -> logpdf(JohnsonSU(γ, δ, ξ, λ), x),
            (0.1, 0.95, 0.1, 1.1, 0.1),
            5,
        ),
        (
            "JohnsonSU 2",
            (γ, δ, ξ, λ, x) -> logpdf(JohnsonSU(γ, δ, ξ, λ), x),
            (0.15, 0.9, 0.12, 0.94, 0.5),
            5,
        ),
        (
            "JohnsonSU 3",
            (γ, δ, ξ, λ, x) -> logpdf(JohnsonSU(γ, δ, ξ, λ), x),
            (0.1, 0.95, 0.1, 1.1, -0.3),
            5,
        ),
        ("Kolmogorov 1", x -> logpdf(Kolmogorov(), x), (1.1,), 1),
        ("Kolmogorov 2", x -> logpdf(Kolmogorov(), x), (0.9,), 1),
        ("Kolmogorov 3", x -> logpdf(Kolmogorov(), x), (1.5,), 1),
        ("Kumaraswamy 1", (a, b, x) -> logpdf(Kumaraswamy(a, b), x), (2.0, 5.0, 0.71), 3),
        ("Kumaraswamy 2", (a, b, x) -> logpdf(Kumaraswamy(a, b), x), (0.1, 5.0, 0.2), 3),
        ("Kumaraswamy 3", (a, b, x) -> logpdf(Kumaraswamy(a, b), x), (0.5, 4.5, 0.1), 3),
        ("Laplace 1", (μ, β, x) -> logpdf(Laplace(μ, β), x), (0.1, 1.0, 0.2), 3),
        ("Laplace 2", (μ, β, x) -> logpdf(Laplace(μ, β), x), (-0.5, 2.1, 0.5), 3),
        ("Laplace 3", (μ, β, x) -> logpdf(Laplace(μ, β), x), (-0.35, 0.4, -0.3), 3),
        ("Levy 1", (μ, c, x) -> logpdf(Levy(μ, c), x), (0.1, 0.9, 4.1), 3),
        ("Levy 2", (μ, c, x) -> logpdf(Levy(μ, c), x), (0.5, 0.9, 0.6), 3),
        ("Levy 3", (μ, c, x) -> logpdf(Levy(μ, c), x), (1.1, 0.5, 2.2), 3),
        ("Lindley 1", (θ, x) -> logpdf(Lindley(θ), x), (0.5, 2.1), 2),
        ("Lindley 2", (θ, x) -> logpdf(Lindley(θ), x), (1.1, 3.1), 2),
        ("Lindley 3", (θ, x) -> logpdf(Lindley(θ), x), (1.9, 3.5), 2),
        ("Logistic 1", (μ, s, x) -> logpdf(Logistic(μ, s), x), (0.1, 1.2, 1.1), 3),
        ("Logistic 2", (μ, s, x) -> logpdf(Logistic(μ, s), x), (0.5, 0.7, 0.6), 3),
        ("Logistic 3", (μ, s, x) -> logpdf(Logistic(μ, s), x), (-0.5, 0.1, -0.4), 3),
        ("LogitNormal 1", (μ, σ, x) -> logpdf(LogitNormal(μ, σ), x), (0.1, 1.1, 0.5), 3),
        ("LogitNormal 2", (μ, σ, x) -> logpdf(LogitNormal(μ, σ), x), (0.5, 0.7, 0.6), 3),
        ("LogitNormal 3", (μ, σ, x) -> logpdf(LogitNormal(μ, σ), x), (-0.12, 1.1, 0.1), 3),
        ("LogNormal 1", (μ, σ, x) -> logpdf(LogNormal(μ, σ), x), (0.0, 1.0, 0.5), 3),
        ("LogNormal 2", (μ, σ, x) -> logpdf(LogNormal(μ, σ), x), (0.5, 1.0, 0.5), 3),
        ("LogNormal 3", (μ, σ, x) -> logpdf(LogNormal(μ, σ), x), (-0.1, 1.3, 0.75), 3),
        ("LogUniform 1", (a, b, x) -> logpdf(LogUniform(a, b), x), (0.1, 0.9, 0.75), 3),
        ("LogUniform 2", (a, b, x) -> logpdf(LogUniform(a, b), x), (0.15, 7.8, 7.1), 3),
        ("LogUniform 3", (a, b, x) -> logpdf(LogUniform(a, b), x), (2.0, 3.0, 2.1), 3),
        ("Normal() 1", x -> logpdf(Normal(), x), (0.1,), 1),
        ("Normal(μ,σ) 1", (μ, σ, x) -> logpdf(Normal(μ, σ), x), (0.0, 1.0, 1.0), 3),
        ("Normal(μ,σ) 2", (μ, σ, x) -> logpdf(Normal(μ, σ), x), (0.5, 1.0, 0.05), 3),
        ("Normal(μ,σ) 3", (μ, σ, x) -> logpdf(Normal(μ, σ), x), (0.0, 1.5, -0.1), 3),
        ("Normal(μ,σ) 4", (μ, σ, x) -> logpdf(Normal(μ, σ), x), (-0.1, 0.9, -0.3), 3),
        ("NormalCanon 1", (m, s, x) -> logpdf(NormalCanon(m, s), x), (0.1, 1.0, -0.5), 3),
        (
            "NormalInverseGaussian 1",
            (μ, α, β, δ, x) -> logpdf(NormalInverseGaussian(μ, α, β, δ), x),
            (0.0, 1.0, 0.2, 0.1, 0.1),
            5,
        ),
        ("Pareto 1", (α, θ, x) -> logpdf(Pareto(α, θ), x), (1.0, 1.0, 3.5), 3),
        ("Pareto 2", (α, θ, x) -> logpdf(Pareto(α, θ), x), (1.1, 0.9, 3.1), 3),
        ("Pareto 3", (α, θ, x) -> logpdf(Pareto(α, θ), x), (1.0, 1.0, 1.4), 3),
        (
            "PGeneralizedGaussian 1",
            (p, x) -> logpdf(PGeneralizedGaussian(p), x),
            (0.2, 5.0),
            2,
        ),
        (
            "PGeneralizedGaussian 2",
            (μ, α, p, x) -> logpdf(PGeneralizedGaussian(μ, α, p), x),
            (0.5, 1.0, 0.3, 5.0),
            4,
        ),
        (
            "PGeneralizedGaussian 3",
            (μ, α, p, x) -> logpdf(PGeneralizedGaussian(μ, α, p), x),
            (-0.1, 11.1, 6.5, -0.3),
            4,
        ),
        ("Rayleigh 1", (σ, x) -> logpdf(Rayleigh(σ), x), (0.5, 0.6), 2),
        ("Rayleigh 2", (σ, x) -> logpdf(Rayleigh(σ), x), (0.9, 1.1), 2),
        ("Rayleigh 3", (σ, x) -> logpdf(Rayleigh(σ), x), (0.55, 0.63), 2),
        ("Semicircle 1", (r, x) -> logpdf(Semicircle(r), x), (1.0, 0.9), 2),
        ("Semicircle 2", (r, x) -> logpdf(Semicircle(r), x), (5.1, 5.05), 2),
        ("Semicircle 3", (r, x) -> logpdf(Semicircle(r), x), (0.5, -0.1), 2),
        (
            "SkewedExponentialPower 1",
            (μ, σ, p, α, x) -> logpdf(SkewedExponentialPower(μ, σ, p, α), x),
            (0.1, 1.0, 0.97, 0.7, -2.0),
            5,
        ),
        (
            "SkewedExponentialPower 2",
            (μ, σ, p, α, x) -> logpdf(SkewedExponentialPower(μ, σ, p, α), x),
            (0.15, 1.0, 0.97, 0.7, -2.0),
            5,
        ),
        (
            "SkewedExponentialPower 3",
            (μ, σ, p, α, x) -> logpdf(SkewedExponentialPower(μ, σ, p, α), x),
            (0.1, 1.1, 0.99, 0.7, 0.5),
            5,
        ),
        (
            "SkewNormal 1",
            (μ, σ, α, x) -> logpdf(SkewNormal(μ, σ, α), x),
            (0.0, 1.0, -1.0, 0.1),
            4,
        ),
        (
            "SkewNormal 2",
            (μ, σ, α, x) -> logpdf(SkewNormal(μ, σ, α), x),
            (0.5, 2.0, 1.1, 0.1),
            4,
        ),
        (
            "SkewNormal 3",
            (μ, σ, α, x) -> logpdf(SkewNormal(μ, σ, α), x),
            (-0.5, 1.0, 0.0, 0.1),
            4,
        ),
        (
            "SymTriangularDist 1",
            (μ, σ, x) -> logpdf(SymTriangularDist(μ, σ), x),
            (0.0, 1.0, 0.5),
            3,
        ),
        (
            "SymTriangularDist 2",
            (μ, σ, x) -> logpdf(SymTriangularDist(μ, σ), x),
            (-0.5, 2.1, -2.0),
            3,
        ),
        (
            "SymTriangularDist 3",
            (μ, σ, x) -> logpdf(SymTriangularDist(μ, σ), x),
            (1.7, 0.3, 1.75),
            3,
        ),
        ("TDist 1", (ν, x) -> logpdf(TDist(ν), x), (1.1, 99.1), 2),
        ("TDist 2", (ν, x) -> logpdf(TDist(ν), x), (10.1, 25.0), 2),
        ("TDist 3", (ν, x) -> logpdf(TDist(ν), x), (2.1, -89.5), 2),
        (
            "TriangularDist 1",
            (a, b, c, x) -> logpdf(TriangularDist(a, b, c), x),
            (0.0, 1.5, 0.5, 0.45),
            4,
        ),
        (
            "TriangularDist 2",
            (a, b, c, x) -> logpdf(TriangularDist(a, b, c), x),
            (0.1, 1.4, 0.45, 0.12),
            4,
        ),
        (
            "TriangularDist 3",
            (a, b, c, x) -> logpdf(TriangularDist(a, b, c), x),
            (0.0, 1.5, 0.5, 0.2),
            4,
        ),
        ("Triweight 1", (μ, σ, x) -> logpdf(Triweight(μ, σ), x), (1.0, 1.0, 1.0), 3),
        ("Triweight 2", (μ, σ, x) -> logpdf(Triweight(μ, σ), x), (1.1, 2.1, 1.0), 3),
        ("Triweight 3", (μ, σ, x) -> logpdf(Triweight(μ, σ), x), (1.9, 10.0, -0.1), 3),
        ("Uniform 1", (a, b, x) -> logpdf(Uniform(a, b), x), (0.0, 1.0, 0.2), 3),
        ("Uniform 2", (a, b, x) -> logpdf(Uniform(a, b), x), (-0.1, 1.1, 1.0), 3),
        ("Uniform 3", (a, b, x) -> logpdf(Uniform(a, b), x), (99.5, 100.5, 100.0), 3),
        ("VonMises 1", (κ, x) -> logpdf(VonMises(κ), x), (0.5, 0.1), 2),
        ("VonMises 2", (κ, x) -> logpdf(VonMises(κ), x), (0.3, -0.1), 2),
        ("VonMises 3", (κ, x) -> logpdf(VonMises(κ), x), (0.2, -0.5), 2),
        ("Weibull 1", (α, θ, x) -> logpdf(Weibull(α, θ), x), (0.5, 1.0, 0.45), 3),
        ("Weibull 2", (α, θ, x) -> logpdf(Weibull(α, θ), x), (0.3, 1.1, 0.66), 3),
        ("Weibull 3", (α, θ, x) -> logpdf(Weibull(α, θ), x), (0.75, 1.3, 0.99), 3),

        # ── Multivariate ──────────────────────────────────────────────────────────

        (
            "MvNormal Diagonal Fill 1",
            (σ, x) -> logpdf(MvNormal(Diagonal(Fill(σ, 1))), [x]),
            (1.5, -0.3),
            2,
        ),
        (
            "MvNormal Diagonal Fill 2",
            (σ, x1, x2) -> logpdf(MvNormal(Diagonal(Fill(σ, 2))), [x1, x2]),
            (0.5, 0.2, -0.3),
            3,
        ),
        (
            "MvNormal mean scalar_var 1",
            (m, σ, x) -> logpdf(MvNormal([m], σ), [x]),
            (0.0, 0.9, 0.1),
            3,
        ),
        (
            "MvNormal mean scalar_var 2",
            (m1, m2, σ, x1, x2) -> logpdf(MvNormal([m1, m2], σ), [x1, x2]),
            (0.0, 0.1, 0.9, 0.1, -0.05),
            5,
        ),
        (
            "MvNormal Diagonal vec 1",
            (σ, x) -> logpdf(MvNormal(Diagonal([σ])), [x]),
            (0.1, 0.1),
            2,
        ),
        (
            "MvNormal Diagonal vec 2",
            (σ1, σ2, x1, x2) -> logpdf(MvNormal(Diagonal([σ1, σ2])), [x1, x2]),
            (0.1, 0.2, 0.1, 0.15),
            4,
        ),
        (
            "MvNormal mean Diagonal Fill 1",
            (m1, m2, σ, x1, x2) ->
                logpdf(MvNormal([m1, m2], Diagonal(Fill(σ, 2))), [x1, x2]),
            (0.1, -0.3, 0.9, 0.1, -0.1),
            5,
        ),
        (
            "MvNormal mean scalar_I 1",
            (m1, m2, σ, x1, x2) -> logpdf(MvNormal([m1, m2], σ * I), [x1, x2]),
            (0.1, -0.1, 0.4, -0.1, 0.15),
            5,
        ),
        (
            "MvNormal mean Hermitian Diagonal 1",
            (m1, m2, σ1, σ2, x1, x2) ->
                logpdf(MvNormal([m1, m2], Hermitian(Diagonal([σ1, σ2]))), [x1, x2]),
            (0.2, 0.3, 0.5, 0.4, -0.1, 0.05),
            6,
        ),
        (
            "MvNormal mean Symmetric Diagonal 1",
            (m1, m2, σ1, σ2, x1, x2) ->
                logpdf(MvNormal([m1, m2], Symmetric(Diagonal([σ1, σ2]))), [x1, x2]),
            (0.2, 0.3, 0.5, 0.4, -0.1, 0.05),
            6,
        ),
        (
            "MvNormal mean Diagonal 1",
            (m1, m2, σ1, σ2, x1, x2) ->
                logpdf(MvNormal([m1, m2], Diagonal([σ1, σ2])), [x1, x2]),
            (0.2, 0.3, 0.5, 0.4, -0.1, 0.05),
            6,
        ),
        (
            "MvNormal mean var_vec 1",
            (m1, m2, v1, v2, x1, x2) -> logpdf(MvNormal([m1, m2], [v1, v2]), [x1, x2]),
            (0.2, -0.3, 0.5, 0.6, 0.4, -0.3),
            6,
        ),

        # PDMat-based MvNormal — workaround: use Symmetric instead of PDMat for the covariance.
        # PDMat wraps a Cholesky factorization and does not support NDual element types, so
        # `MvNormal(μ, PDMat(NDual_matrix))` would error.  Constructing the covariance via
        # `Symmetric([s11 s12; s12 s22])` keeps the matrix as a plain AbstractMatrix{NDual}
        # which Distributions can consume without going through PDMat's factorization path.
        (
            "MvNormal Symmetric 1x1",
            (s, x) -> logpdf(MvNormal([-0.15], Symmetric(reshape([s], 1, 1))), [x]),
            (1.21, -0.05),
            2,
        ),
        (
            "MvNormal Symmetric 2x2",
            (s11, s12, s22, x1, x2) ->
                logpdf(MvNormal([0.2, -0.15], Symmetric([s11 s12; s12 s22])), [x1, x2]),
            (2.01, 0.63, 1.21, 0.05, -0.05),
            5,
        ),
        (
            "MvNormalCanon Symmetric 2x2",
            (s11, s12, s22, x1, x2) ->
                logpdf(MvNormalCanon([0.1, -0.1], Symmetric([s11 s12; s12 s22])), [x1, x2]),
            (1.45, 0.9, 1.21, 0.2, -0.25),
            5,
        ),
        (
            "MvLogNormal Symmetric 2x2",
            (s11, s12, s22, x1, x2) -> logpdf(
                MvLogNormal(MvNormal([0.2, -0.1], Symmetric([s11 s12; s12 s22]))),
                [x1, x2],
            ),
            (2.01, 0.63, 1.21, 0.5, 0.1),
            5,
        ),

        # product_distribution — observation-only differentiation; component distributions hardcoded.
        # Distribution objects are not parameterized by floats and cannot be NDual inputs.
        (
            "product_distribution Normal x",
            x -> logpdf(product_distribution([Normal()]), [x]),
            (0.3,),
            1,
        ),
        (
            "product_distribution Normal+Uniform x",
            (x1, x2) -> logpdf(product_distribution([Normal(), Uniform()]), [x1, x2]),
            (-0.4, 0.3),
            2,
        ),

        # Categorical — differentiate w.r.t. probability parameter
        ("Categorical 1", x -> logpdf(Categorical(x, 1 - x), 1), (0.3,), 1),

        # Dirichlet — full differentiation w.r.t. concentration params and observation
        (
            "Dirichlet α+x",
            (α1, α2, x1, x2) -> logpdf(Dirichlet([α1, α2]), [x1, x2]),
            (1.5, 1.1, 0.4, 0.6),
            4,
        ),

        # MvLogitNormal — covariance via Symmetric (same workaround as PDMat MvNormal above)
        (
            "MvLogitNormal m+Σ+x",
            (m1, m2, s11, s12, s22, x1, x2) -> logpdf(
                MvLogitNormal([m1, m2], Symmetric([s11 s12; s12 s22])),
                vcat([x1, x2], 1 - x1 - x2),
            ),
            (0.4, 0.6, 2.01, 0.63, 1.21, 0.27, 0.24),
            7,
        ),

        # ── Matrix-variate ────────────────────────────────────────────────────────
        # Observation-only differentiation; all distribution parameters are hardcoded.
        # PDMat scale params do not support NDual element types.

        (
            "MatrixNormal X",
            X -> logpdf(
                MatrixNormal(
                    randn(StableRNG(0), 2, 3),
                    _pdmat(randn(StableRNG(1), 2, 2)),
                    _pdmat(randn(StableRNG(2), 3, 3)),
                ),
                X,
            ),
            (randn(StableRNG(4), 2, 3),),
            6,
        ),

        # Wishart and InverseWishart are covered by the standard logpdf_test_cases above
        # (which use Symmetric observations so FD perturbations preserve positive-definiteness).
        # Passing a plain Matrix here causes element-wise FD perturbations that break PD.

        (
            "MatrixTDist X",
            X -> logpdf(
                MatrixTDist(
                    3.1,
                    randn(StableRNG(9), 2, 3),
                    _pdmat(randn(StableRNG(0), 2, 2)),
                    _pdmat(randn(StableRNG(1), 3, 3)),
                ),
                X,
            ),
            (randn(StableRNG(2), 2, 3),),
            6,
        ),
        (
            "MatrixBeta X",
            X -> logpdf(MatrixBeta(5, 9.0, 10.0), X),
            (rand(StableRNG(123456), MatrixBeta(5, 9.0, 10.0)),),
            25,
        ),
        (
            "MatrixFDist X",
            X -> logpdf(MatrixFDist(6.0, 7.0, _pdmat(randn(StableRNG(1234), 5, 5))), X),
            (
                rand(
                    StableRNG(13), MatrixFDist(6.0, 7.0, _pdmat(randn(StableRNG(11), 5, 5)))
                ),
            ),
            25,
        ),
        ("LKJ η", η -> logpdf(LKJ(5, η), rand(StableRNG(123456), LKJ(5, 1.1))), (1.1,), 1),
        (
            "LKJ R",
            Rmat -> logpdf(LKJ(5, 1.1), Rmat),
            (collect(rand(StableRNG(123456), LKJ(5, 1.1))),),
            25,
        ),
        (
            "LKJ η+R",
            (η, Rmat) -> logpdf(LKJ(5, η), Rmat),
            (1.1, collect(rand(StableRNG(123456), LKJ(5, 1.1)))),
            26,
        ),

        # ── Truncated distributions ───────────────────────────────────────────────

        # truncated Beta — differentiate w.r.t. truncation bounds and observation only.
        # Shape params α, β are fixed constants; the CDF normalisation calls
        # beta_inc(Float64, Float64, NDual) which requires only the x-partial of I_x(a,b).
        # Differentiating w.r.t. shape params would require ∂I_x/∂a, ∂I_x/∂b (not implemented).
        (
            "truncated Beta 1",
            (a, b, x) -> logpdf(truncated(Beta(1.1, 1.3), a, b), x),
            (0.1, 0.9, 0.4),
            3,
        ),
        (
            "truncated Beta lower 1",
            (a, x) -> logpdf(truncated(Beta(1.1, 1.3); lower=a), x),
            (0.1, 0.4),
            2,
        ),
        (
            "truncated Normal 1",
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1),
            3,
        ),
        (
            "truncated Uniform 1",
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4),
            5,
        ),

        # LKJCholesky — workaround: pass lower-triangular factor L as a plain Matrix.
        # Cholesky's constructor requires a numeric matrix, not NDual, so we accept Lmat::Matrix
        # as the NDual input and reconstruct Cholesky(Lmat, 'L', 0) inside the lambda.
        (
            "LKJCholesky L",
            Lmat -> logpdf(LKJCholesky(5, 1.1), Cholesky(Lmat, 'L', 0)),
            (Matrix(rand(StableRNG(123456), LKJCholesky(5, 1.1)).L),),
            25,
        ),
        (
            "LKJCholesky η+L",
            (η, Lmat) -> logpdf(LKJCholesky(5, η), Cholesky(Lmat, 'L', 0)),
            (1.1, Matrix(rand(StableRNG(123456), LKJCholesky(5, 1.1)).L)),
            26,
        ),
    ]

    @testset "nforward $name" for (name, f, args, C) in nforward_test_cases
        test_rule(
            StableRNG(123456),
            f,
            args...;
            is_primitive=false,
            perf_flag=:none,
            mode=Mooncake.ReverseMode,
            rrule=_NForwardMode(f, args, C),
        )
    end
end
