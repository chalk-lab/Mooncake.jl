module MooncakeDistributionsExt

using Distributions, Mooncake, LinearAlgebra
using Base: IEEEFloat
using Distributions: loglikelihood, sqmahal
using Distributions.FillArrays: Fill
using Distributions.PDMats: PDiagMat, PDMat, ScalMat
using PrecompileTools: @setup_workload, @compile_workload

import Mooncake:
    @is_primitive,
    CoDual,
    ContiguousSubVector,
    DefaultCtx,
    Dual,
    NoFData,
    NoRData,
    RData,
    ReverseMode,
    Tangent,
    _fields,
    arrayify,
    frule!!,
    increment!!,
    primal,
    rrule!!,
    tangent,
    zero_fcodual

# The rules below exist purely to work around performance limitations of Mooncake.jl: the
# derived rules are correct, but slow. Both fuse an elementwise broadcast that AD would
# otherwise run at roughly 25x the primal's runtime. As in `src/rules/performance_patches.jl`,
# each signature covers a finite set of concrete types, all of which are tested.

# Both rules below work in `z = (x - μ) / σ` rather than `σ^2`: squaring flushes the
# derivative to `Inf` for `σ` well inside the range where it is representable
# (`σ < 1e-20` in `Float32`).
@is_primitive DefaultCtx Tuple{typeof(logpdf),Normal{P},P} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{Normal{P}}, x::Dual{P}
) where {P<:IEEEFloat}
    dp = primal(d)
    ḋ = _fields(tangent(d))
    inv_σ = inv(dp.σ)
    z = (primal(x) - dp.μ) * inv_σ
    ẏ = inv_σ * ((abs2(z) - one(P)) * ḋ.σ - z * (tangent(x) - ḋ.μ))
    return Dual(logpdf(dp, primal(x)), ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{Normal{P}}, x::CoDual{P}
) where {P<:IEEEFloat}
    dp = primal(d)
    inv_σ = inv(dp.σ)
    z = (primal(x) - dp.μ) * inv_σ
    function normal_logpdf_pb!!(dy::P)
        dx = -dy * z * inv_σ
        dd = RData((μ=(-dx), σ=dy * (abs2(z) - one(P)) * inv_σ))
        return NoRData(), dd, dx
    end
    return zero_fcodual(logpdf(dp, primal(x))), normal_logpdf_pb!!
end

# `sqmahal` is the only part of `logpdf(::MvNormal, ::AbstractVector)` whose cost grows with
# the dimension. A `Fill` mean is what `product_distribution(Fill(Normal(μ, σ), n))`
# produces, which is how DynamicPPL.jl represents an i.i.d. normal prior; a `Vector` mean is
# what `MvNormal(μ, σ^2 * I)` produces. `logpdf(::MvNormal, ::AbstractMatrix)` is not
# covered: it routes through `sqmahal!` rather than `sqmahal`.
const ScalMvNormal{P} = MvNormal{P,<:ScalMat{P},<:Union{Vector{P},Fill{P,1}}}

# `AbstractVector` here would also capture array types that `arrayify` rejects, GPU arrays
# among them.
const DenseVec{P} = Union{Vector{P},ContiguousSubVector{P}}

# The primal reaches this check via broadcasting `x .- d.μ`, which these rules replace.
function _check_dims(d::MvNormal, x::AbstractVector)
    length(x) == length(d) && return nothing
    throw(DimensionMismatch(lazy"x has length $(length(x)), expected $(length(d))"))
end

# A `Fill` mean has a single tangent value shared by every element.
_mean_tangent(μ̇::Tangent, ::Int) = _fields(μ̇).value
_mean_tangent(μ̇::Vector, i::Int) = @inbounds μ̇[i]

# A `Vector` mean carries its gradient in fdata; a `Fill` mean carries it in rdata, which
# leaves the distribution with no fdata at all.
_mean_fdata(::CoDual{<:ScalMvNormal,NoFData}) = NoFData()
_mean_fdata(d::CoDual{<:ScalMvNormal}) = _fields(d.dx).μ

# Accumulate the gradients of `x` and of the mean in one pass, returning the mean's rdata.
# Recomputing the residuals is cheaper than allocating a buffer for them in the forward
# pass and holding it until the pullback runs.
function _accum_sqmahal!(dx, ::NoFData, px, μ, scale::P) where {P}
    total = zero(P)
    @inbounds @simd for i in eachindex(dx, px)
        residual = px[i] - μ[i]
        dx[i] += scale * residual
        total += residual
    end
    return RData((value=-scale * total, axes=NoRData()))
end
function _accum_sqmahal!(dx, dμ::Vector{P}, px, μ, scale::P) where {P}
    @inbounds @simd for i in eachindex(dx, dμ, px)
        contribution = scale * (px[i] - μ[i])
        dx[i] += contribution
        dμ[i] -= contribution
    end
    return NoRData()
end

@is_primitive DefaultCtx Tuple{
    typeof(sqmahal),ScalMvNormal{P},DenseVec{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(sqmahal)}, d::Dual{<:ScalMvNormal{P}}, x::Dual{<:DenseVec{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    ḋ = _fields(tangent(d))
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    μ = dp.μ
    variance = dp.Σ.value
    y = zero(P)
    ẏ = zero(P)
    @inbounds @simd for i in eachindex(px, ẋ)
        residual = px[i] - μ[i]
        y += abs2(residual)
        ẏ += residual * (ẋ[i] - _mean_tangent(ḋ.μ, i))
    end
    y /= variance
    return Dual(y, (2 * ẏ - y * _fields(ḋ.Σ).value) / variance)
end
function rrule!!(
    ::CoDual{typeof(sqmahal)}, d::CoDual{<:ScalMvNormal{P}}, x::CoDual{<:DenseVec{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    dμ = _mean_fdata(d)
    μ = dp.μ
    variance = dp.Σ.value
    # Assigned once: a captured variable that a loop reassigns gets boxed, which would make
    # the pullback type-unstable.
    y = let total = zero(P)
        @inbounds @simd for i in eachindex(px)
            total += abs2(px[i] - μ[i])
        end
        total / variance
    end
    function sqmahal_pb!!(dy::P)
        dd = RData((
            μ=_accum_sqmahal!(dx, dμ, px, μ, 2 * dy / variance),
            Σ=RData((dim=NoRData(), value=-dy * y / variance)),
        ))
        return NoRData(), dd, NoRData()
    end
    return zero_fcodual(y), sqmahal_pb!!
end

# `product_distribution(::AbstractVector{<:Normal})` specializes to this representation.
# The rules from here on accumulate into fdata, so every container they accept has to keep
# its parameters there. A `Fill` diagonal, or a `Fill` of distributions, keeps them in rdata
# instead and must go on using the derived rules.
const DiagMvNormal{P} = MvNormal{P,<:PDiagMat{P,Vector{P}},<:Vector{P}}

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(logpdf),DiagMvNormal{P},DenseVec{P}
} where {P<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:DiagMvNormal{P}}, x::CoDual{<:DenseVec{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    variance = dp.Σ.diag
    logdetΣ = zero(P)
    mahalanobis = zero(P)
    @inbounds @simd for i in eachindex(px, dp.μ, variance)
        logdetΣ += log(variance[i])
        mahalanobis += abs2(px[i] - dp.μ[i]) / variance[i]
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + logdetΣ + mahalanobis)
    fields = _fields(tangent(d))
    dμ = fields.μ
    dvariance = _fields(fields.Σ).diag
    function diag_normal_logpdf_pb!!(dy::P)
        @inbounds @simd for i in eachindex(px, dp.μ, variance, dμ, dvariance)
            residual = px[i] - dp.μ[i]
            inverse_variance = inv(variance[i])
            dx_i = -dy * residual * inverse_variance
            dx[i] += dx_i
            dμ[i] -= dx_i
            dvariance[i] +=
                dy * P(0.5) * (abs2(residual) * abs2(inverse_variance) - inverse_variance)
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), diag_normal_logpdf_pb!!
end

# The heterogeneous case: Distributions represents a `Fill` vector of `Normal`s as an
# isotropic `MvNormal`, which the `sqmahal` rules above cover.
const NormalProduct{P,N} = Distributions.ProductDistribution{N,0,Array{Normal{P},N},<:Any,P}

_dists(d::Distributions.ProductDistribution) = d.dists
_dists_fdata(::Distributions.ProductDistribution, dd) = _fields(dd).dists

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(logpdf),NormalProduct{P,N},Array{P,N}
} where {P<:IEEEFloat,N}
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:NormalProduct{P,N}}, x::CoDual{Array{P,N}}
) where {P<:IEEEFloat,N}
    dp = primal(d)
    px, dx = arrayify(x)
    size(dp) == size(px) ||
        throw(DimensionMismatch(lazy"x has size $(size(px)), expected $(size(dp))"))
    dists = _dists(dp)
    y = zero(P)
    @inbounds @simd for i in eachindex(px, dists)
        dist = dists[i]
        z = (px[i] - dist.μ) / dist.σ
        y -= P(0.5) * log(P(2π)) + log(dist.σ) + P(0.5) * abs2(z)
    end
    ddists = _dists_fdata(dp, tangent(d))
    function normal_product_logpdf_pb!!(dy::P)
        @inbounds for i in eachindex(px, dists, ddists)
            dist = dists[i]
            inv_σ = inv(dist.σ)
            z = (px[i] - dist.μ) * inv_σ
            dx_i = -dy * z * inv_σ
            dx[i] += dx_i
            fields = _fields(ddists[i])
            ddists[i] = Tangent((
                μ=(fields.μ - dx_i), σ=(fields.σ + dy * (abs2(z) - one(P)) * inv_σ)
            ))
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), normal_product_logpdf_pb!!
end

# `product_distribution(::AbstractVector{<:UnivariateDistribution})` returns the deprecated
# `Product` today and a `ProductDistribution` once Distributions removes it. Matching only
# `Product` would silently stop firing at that point, so both are covered here.
const UnivariateProduct{D} = Union{
    Distributions.Product{<:Distributions.ValueSupport,D,Vector{D}},
    Distributions.ProductDistribution{1,0,Vector{D}},
}
const CountingProduct{P} = Union{
    UnivariateProduct{BernoulliLogit{P}},UnivariateProduct{Poisson{P}}
}

_dists(d::Distributions.Product) = d.v
_dists_fdata(::Distributions.Product, dd) = _fields(dd).v

# One observation's contribution to its distribution's cotangent. Add a method here to
# cover another counting distribution.
function _param_cotangent(d::BernoulliLogit, k, dy::P) where {P}
    return Tangent((logitp=dy * (k - Distributions.logistic(d.logitp)),))
end
function _param_cotangent(d::Poisson, k, dy::P) where {P}
    # `k / λ` is `NaN` at the degenerate `λ = 0`, where the primal's `xlogy(k, λ)` term is
    # flat in `λ` and so contributes nothing.
    return Tangent((λ=dy * (iszero(k) ? -one(P) : k / d.λ - one(P)),))
end

# The sample stays an `AbstractVector`, unlike the float samples above: it is
# non-differentiable, so it is only ever indexed, never passed to `arrayify`. That admits
# the `BitVector` binary observations arrive as.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(logpdf),CountingProduct{P},AbstractVector{<:Integer}
} where {P<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(logpdf)},
    d::CoDual{<:CountingProduct{P}},
    x::CoDual{<:AbstractVector{<:Integer}},
) where {P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    length(dp) == length(px) ||
        throw(DimensionMismatch(lazy"x has length $(length(px)), expected $(length(dp))"))
    y = logpdf(dp, px)
    dists = _dists(dp)
    ddists = _dists_fdata(dp, tangent(d))
    function counting_product_logpdf_pb!!(dy::P)
        @inbounds for i in eachindex(px, dists, ddists)
            # An out-of-support observation makes the primal `-Inf`, whose gradient is zero.
            insupport(dists[i], px[i]) || continue
            ddists[i] = increment!!(ddists[i], _param_cotangent(dists[i], px[i], dy))
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), counting_product_logpdf_pb!!
end

# Repeated observations from one dense multivariate Normal are represented by
# `loglikelihood(d, X)`, with observations in the columns of `X`. Keeping the shared
# Cholesky factor at this public boundary avoids tracing one triangular solve per column.
const CholeskyMvNormal{P} = MvNormal{
    P,<:PDMat{P,<:Matrix{P},<:Cholesky{P,<:Matrix{P}}},<:Vector{P}
}

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(loglikelihood),CholeskyMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function rrule!!(
    ::CoDual{typeof(loglikelihood)}, d::CoDual{<:CholeskyMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    size(px, 1) == length(dp) ||
        throw(DimensionMismatch(lazy"x has $(size(px, 1)) rows, expected $(length(dp))"))
    L = dp.Σ.chol.L
    upper = dp.Σ.chol.uplo === 'U'
    standardized = L \ (px .- dp.μ)
    n = size(px, 2)
    y =
        -P(0.5) *
        (length(px) * log(P(2π)) + n * logdet(dp.Σ.chol) + sum(abs2, standardized))
    fields = _fields(tangent(d))
    dfactors = _fields(_fields(fields.Σ).chol).factors
    function mvnormal_loglikelihood_pb!!(dy::P)
        x_gradient = -(L' \ standardized)
        dx .+= dy .* x_gradient
        fields.μ .-= dy .* vec(sum(x_gradient; dims=2))
        factor_gradient = L' \ (standardized * standardized')
        @inbounds for j in axes(dfactors, 2), i in j:size(dfactors, 1)
            contribution = factor_gradient[i, j]
            if i == j
                contribution -= n / L[i, i]
            end
            # `factors` holds `L'` when `uplo == 'U'`; the cotangent of `L[i, j]` then
            # belongs at `factors[j, i]`, the entry the primal reads.
            row, column = upper ? (j, i) : (i, j)
            dfactors[row, column] += dy * contribution
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), mvnormal_loglikelihood_pb!!
end

#! format: off

# Skip precompilation on GitHub Actions for Julia versions earlier than 1.11.
# On Julia LTS (1.10), precompilation can cause certain Mooncake allocation tests to fail.
# See also the identical guard in `src/precompile.jl`.
@static if !haskey(ENV, "GITHUB_ACTIONS") || VERSION ≥ v"1.11-"

# Precompile the AD machinery for the most common `logpdf` patterns so that users
# who load Distributions.jl get a much smaller time-to-first-gradient when
# differentiating through distribution log-densities.
#
# The workload exercises `prepare_gradient_cache` → `value_and_gradient!!` for a
# representative subset of distribution families drawn from
# `test/integration_testing/distributions/distributions.jl`:
#   • a simple univariate distribution  (Normal)
#   • a simple multivariate distribution (MvNormal with diagonal covariance)
#   • an i.i.d. normal prior             (`sqmahal`, both mean types)
#   • independent normals with distinct parameters, and a counting likelihood
#   • repeated observations from one dense multivariate normal (`loglikelihood`)

@setup_workload begin
    @compile_workload begin
        # Reverse-mode: univariate logpdf
        d_uni = Normal(0.0, 1.0)
        cache_uni = Mooncake.prepare_gradient_cache(logpdf, d_uni, 0.1)
        Mooncake.value_and_gradient!!(cache_uni, logpdf, d_uni, 0.1)

        # Reverse-mode: multivariate logpdf
        d_mv = MvNormal(Diagonal([1.0, 1.0]))
        x_mv = [0.1, -0.1]
        cache_mv = Mooncake.prepare_gradient_cache(logpdf, d_mv, x_mv)
        Mooncake.value_and_gradient!!(cache_mv, logpdf, d_mv, x_mv)

        # Reverse-mode: scalar covariance, with a `Fill` mean and with a `Vector` mean
        for d_iid in (product_distribution(Fill(Normal(), 2)), MvNormal([0.0, 0.0], 1.0))
            cache_iid = Mooncake.prepare_gradient_cache(logpdf, d_iid, x_mv)
            Mooncake.value_and_gradient!!(cache_iid, logpdf, d_iid, x_mv)
        end

        # Reverse-mode: independent normals with distinct parameters
        d_prod = product_distribution([Normal(0.0, 1.0), Normal(0.1, 1.2)])
        cache_prod = Mooncake.prepare_gradient_cache(logpdf, d_prod, x_mv)
        Mooncake.value_and_gradient!!(cache_prod, logpdf, d_prod, x_mv)

        # Reverse-mode: a counting likelihood
        d_count = product_distribution([Poisson(1.0), Poisson(1.5)])
        y_count = [1, 2]
        cache_count = Mooncake.prepare_gradient_cache(logpdf, d_count, y_count)
        Mooncake.value_and_gradient!!(cache_count, logpdf, d_count, y_count)

        # Reverse-mode: repeated observations from one dense multivariate normal
        d_dense = MvNormal([0.0, 0.0], PDMat([1.0 0.1; 0.1 1.0]))
        X_dense = [0.1 -0.2; -0.1 0.3]
        cache_dense = Mooncake.prepare_gradient_cache(loglikelihood, d_dense, X_dense)
        Mooncake.value_and_gradient!!(cache_dense, loglikelihood, d_dense, X_dense)
    end
end

end # @static if

#! format: on

end
