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
# (`σ < 1e-20` in `Float32`). Dividing by `σ` rather than scaling by `inv(σ)` matters for
# the same reason: `inv(σ)` overflows once `σ < 1 / floatmax(P)`, where `(x - μ) / σ` and
# the derivative are both still finite.
@is_primitive DefaultCtx Tuple{typeof(logpdf),Normal{P},P} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{Normal{P}}, x::Dual{P}
) where {P<:IEEEFloat}
    dp = primal(d)
    ḋ = _fields(tangent(d))
    z = (primal(x) - dp.μ) / dp.σ
    ẏ = ((abs2(z) - one(P)) * ḋ.σ - z * (tangent(x) - ḋ.μ)) / dp.σ
    return Dual(logpdf(dp, primal(x)), ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{Normal{P}}, x::CoDual{P}
) where {P<:IEEEFloat}
    dp = primal(d)
    z = (primal(x) - dp.μ) / dp.σ
    function normal_logpdf_pb!!(dy::P)
        dx = -dy * z / dp.σ
        dd = RData((μ=(-dx), σ=dy * (abs2(z) - one(P)) / dp.σ))
        return NoRData(), dd, dx
    end
    return zero_fcodual(logpdf(dp, primal(x))), normal_logpdf_pb!!
end

# `sqmahal` is the only part of `logpdf(::MvNormal, ::AbstractVector)` whose cost grows with
# the dimension. A `Fill` mean is what `product_distribution(Fill(Normal(μ, σ), n))`
# produces, which is how DynamicPPL.jl represents an i.i.d. normal prior; a `Vector` mean is
# what `MvNormal(μ, σ^2 * I)` produces. A matrix sample never reaches `sqmahal`: it routes
# through `sqmahal!`, and the matrix rules further down intercept `logpdf` and
# `loglikelihood` for a `Vector` mean only, leaving a `Fill` mean to the derived rules.
const ScalMvNormal{P} = MvNormal{P,<:ScalMat{P},<:Union{Vector{P},Fill{P,1}}}

# `AbstractVector` here would also capture array types that `arrayify` rejects, GPU arrays
# among them.
const DenseVec{P} = Union{Vector{P},ContiguousSubVector{P}}

# Accumulate the Cholesky factor's cotangent. `factors` holds `L'` when `uplo == 'U'`, so
# the cotangent of `L[i, j]` belongs at `factors[j, i]`, the entry the primal reads; the
# other triangle would leave the gradient where nothing looks for it. `total_weight` is the
# summed cotangent of the columns, which the log-determinant's diagonal term carries;
# `weight` scales the quadratic term and is `1` where the caller has already folded its
# per-column weights into `factor_gradient`.
function _accum_factor!(
    dfactors::Matrix, factor_gradient::Matrix, L, weight::P, total_weight::P, upper::Bool
) where {P}
    @inbounds for j in axes(dfactors, 2), i in j:size(dfactors, 1)
        contribution = weight * factor_gradient[i, j]
        if i == j
            contribution -= total_weight / L[i, i]
        end
        row, column = upper ? (j, i) : (i, j)
        dfactors[row, column] += contribution
    end
    return nothing
end

# The tangent of `L`. `permutedims` returns a `Matrix` like its argument, so both branches
# have one type and the matrix products at the call sites stay inferable.
function _factor_tangent(chol::Cholesky, factors::Matrix)
    return LowerTriangular(chol.uplo === 'U' ? permutedims(factors) : factors)
end

# `residual^2 - variance`, the numerator of the variance derivative, with the product kept
# exact by `fma`: the two vanish together when the residual is one standard deviation, and a
# `Float64` residual near `1e-80` squares to the variance itself, leaving a plain `r * r`
# nothing to subtract. Callers divide by the variance twice rather than forming `variance^2`,
# which underflows well inside the useful range. Accuracy is limited only where `residual^2`
# is itself subnormal.
function _excess(residual::P, variance::P) where {P<:IEEEFloat}
    squared = residual * residual
    return (squared - variance) + fma(residual, residual, -squared)
end

# One coordinate's contribution to the gradients of the sample and of the variance, given the
# cotangent of its log-density; the mean's is the sample's negated. Dividing rather than
# scaling by a reciprocal matters: `inv(variance)` overflows for variances that are small but
# perfectly ordinary (σ = 0.0039 in `Float16`), where the derivative is representable.
function _coordinate_gradients(residual::P, variance::P, weight::P) where {P<:IEEEFloat}
    return -weight * residual / variance,
    weight * _excess(residual, variance) / variance / (2 * variance)
end

# One coordinate's directional derivative, before the `-1/2` its callers apply, given the
# perturbation of the sample minus that of the mean. The isotropic rules call this with the
# same variance for every coordinate.
function _coordinate_derivative(
    residual::P, variance::P, perturbation::P, v̇::P
) where {P<:IEEEFloat}
    return 2 * residual * perturbation / variance -
           v̇ * _excess(residual, variance) / variance / variance
end

# The primal reaches these checks via broadcasting `x .- d.μ`, which the rules replace.
function _check_dims(d::MvNormal, x::AbstractVector)
    length(x) == length(d) && return nothing
    throw(DimensionMismatch(lazy"x has length $(length(x)), expected $(length(d))"))
end
function _check_dims(d::MvNormal, x::AbstractMatrix)
    size(x, 1) == length(d) && return nothing
    throw(DimensionMismatch(lazy"x has $(size(x, 1)) rows, expected $(length(d))"))
end
function _check_dims(d::Distributions.ProductDistribution, x::AbstractArray)
    size(x) == size(d) && return nothing
    throw(DimensionMismatch(lazy"x has size $(size(x)), expected $(size(d))"))
end
function _check_dims(d::Distributions.Product, x::AbstractVector)
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

@is_primitive DefaultCtx Tuple{
    typeof(logpdf),DiagMvNormal{P},DenseVec{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{<:DiagMvNormal{P}}, x::Dual{<:DenseVec{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    ḋ = _fields(tangent(d))
    μ̇ = ḋ.μ
    v̇ = _fields(ḋ.Σ).diag
    variance = dp.Σ.diag
    y = zero(P)
    ẏ = zero(P)
    @inbounds @simd for i in eachindex(px, dp.μ, variance, μ̇, v̇)
        residual = px[i] - dp.μ[i]
        scaled_residual = residual / variance[i]
        y += log(variance[i]) + residual * scaled_residual
        ẏ += _coordinate_derivative(residual, variance[i], ẋ[i] - μ̇[i], v̇[i])
    end
    return Dual(-P(0.5) * (length(px) * log(P(2π)) + y), -P(0.5) * ẏ)
end
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
            contribution, dv = _coordinate_gradients(px[i] - dp.μ[i], variance[i], dy)
            dx[i] += contribution
            dμ[i] -= contribution
            dvariance[i] += dv
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), diag_normal_logpdf_pb!!
end

# The heterogeneous case: Distributions represents a `Fill` vector of `Normal`s as an
# isotropic `MvNormal`, which the `sqmahal` rules above cover.
const NormalProduct{P,N} = Distributions.ProductDistribution{N,0,Array{Normal{P},N},<:Any,P}

_dists(d::Distributions.ProductDistribution) = d.dists
_dists(::Distributions.ProductDistribution, dd) = _fields(dd).dists

@is_primitive DefaultCtx Tuple{
    typeof(logpdf),NormalProduct{P,N},Array{P,N}
} where {P<:IEEEFloat,N}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{<:NormalProduct{P,N}}, x::Dual{Array{P,N}}
) where {P<:IEEEFloat,N}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    dists = _dists(dp)
    ḋists = _dists(dp, tangent(d))
    y = zero(P)
    ẏ = zero(P)
    @inbounds for i in eachindex(px, dists, ḋists)
        dist = dists[i]
        ḋ = _fields(ḋists[i])
        z = (px[i] - dist.μ) / dist.σ
        y -= P(0.5) * log(P(2π)) + log(dist.σ) + P(0.5) * abs2(z)
        ẏ += ((abs2(z) - one(P)) * ḋ.σ - z * (ẋ[i] - ḋ.μ)) / dist.σ
    end
    # A `σ == 0` component makes the fused sum `Inf - Inf`, where the primal's `iszero(σ)`
    # branch gives ±Inf. Recomputing only when the sum is not a number leaves the loop
    # above vectorisable; the derivative agrees with the derived rule either way.
    isnan(y) && (y = logpdf(dp, px))
    return Dual(y, ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:NormalProduct{P,N}}, x::CoDual{Array{P,N}}
) where {P<:IEEEFloat,N}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    dists = _dists(dp)
    y = zero(P)
    @inbounds @simd for i in eachindex(px, dists)
        dist = dists[i]
        z = (px[i] - dist.μ) / dist.σ
        y -= P(0.5) * log(P(2π)) + log(dist.σ) + P(0.5) * abs2(z)
    end
    # A `σ == 0` component makes the fused sum `Inf - Inf`, where the primal's `iszero(σ)`
    # branch gives ±Inf. Recomputing only when the sum is not a number leaves the loop
    # above vectorisable; the derivative agrees with the derived rule either way.
    isnan(y) && (y = logpdf(dp, px))
    ddists = _dists(dp, tangent(d))
    function normal_product_logpdf_pb!!(dy::P)
        @inbounds for i in eachindex(px, dists, ddists)
            dist = dists[i]
            z = (px[i] - dist.μ) / dist.σ
            dx_i = -dy * z / dist.σ
            dx[i] += dx_i
            fields = _fields(ddists[i])
            ddists[i] = Tangent((
                μ=(fields.μ - dx_i), σ=(fields.σ + dy * (abs2(z) - one(P)) / dist.σ)
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
_dists(::Distributions.Product, dd) = _fields(dd).v

# One observation's contribution to its distribution's cotangent. Add a method here to
# cover another counting distribution.
_param_derivative(d::BernoulliLogit, k) = k - Distributions.logistic(d.logitp)
function _param_derivative(d::Poisson{P}, k) where {P}
    # `k / λ` is `NaN` at the degenerate `λ = 0`, where the primal's `xlogy(k, λ)` term is
    # flat in `λ` and so contributes nothing.
    return iszero(k) ? -one(P) : k / d.λ - one(P)
end

_param_cotangent(d::BernoulliLogit, k, dy) = Tangent((logitp=dy * _param_derivative(d, k),))
_param_cotangent(d::Poisson, k, dy) = Tangent((λ=dy * _param_derivative(d, k),))

# Both distributions carry one differentiable parameter, so its tangent is the only field
# of the element's tangent. A distribution with more would fail here rather than silently
# differentiate one parameter.
_param_tangent(ṫ) = only(values(_fields(ṫ)))

# The sample stays an `AbstractVector`, unlike the float samples above: it is
# non-differentiable, so it is only ever indexed, never passed to `arrayify`. That admits
# the `BitVector` binary observations arrive as.
@is_primitive DefaultCtx Tuple{
    typeof(logpdf),CountingProduct{P},AbstractVector{<:Integer}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)},
    d::Dual{<:CountingProduct{P}},
    x::Dual{<:AbstractVector{<:Integer}},
) where {P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    _check_dims(dp, px)
    dists = _dists(dp)
    ḋists = _dists(dp, tangent(d))
    ẏ = zero(P)
    @inbounds for i in eachindex(px, dists, ḋists)
        insupport(dists[i], px[i]) || continue
        ẏ += _param_derivative(dists[i], px[i]) * _param_tangent(ḋists[i])
    end
    return Dual(logpdf(dp, px), ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)},
    d::CoDual{<:CountingProduct{P}},
    x::CoDual{<:AbstractVector{<:Integer}},
) where {P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    _check_dims(dp, px)
    y = logpdf(dp, px)
    dists = _dists(dp)
    ddists = _dists(dp, tangent(d))
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

# `logpdf(d, X)` scores each column of `X` separately and returns a vector, so unlike
# `loglikelihood` it cannot fold the columns together; the saving is the same per-column
# triangular solve the derived rule traces.
@is_primitive DefaultCtx Tuple{
    typeof(logpdf),DiagMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{<:DiagMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    ḋ = _fields(tangent(d))
    μ̇ = ḋ.μ
    v̇ = _fields(ḋ.Σ).diag
    variance = dp.Σ.diag
    constant = -P(0.5) * (length(dp) * log(P(2π)) + sum(log, variance))
    y = Vector{P}(undef, size(px, 2))
    ẏ = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        derivative = zero(P)
        @simd for i in eachindex(dp.μ, variance, μ̇, v̇)
            residual = px[i, j] - dp.μ[i]
            scaled_residual = residual / variance[i]
            mahalanobis += residual * scaled_residual
            derivative += _coordinate_derivative(
                residual, variance[i], ẋ[i, j] - μ̇[i], v̇[i]
            )
        end
        y[j] = constant - P(0.5) * mahalanobis
        ẏ[j] = -P(0.5) * derivative
    end
    return Dual(y, ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:DiagMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    variance = dp.Σ.diag
    constant = -P(0.5) * (length(dp) * log(P(2π)) + sum(log, variance))
    y = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        @simd for i in eachindex(dp.μ, variance)
            mahalanobis += abs2(px[i, j] - dp.μ[i]) / variance[i]
        end
        y[j] = constant - P(0.5) * mahalanobis
    end
    out = zero_fcodual(y)
    dy = tangent(out)
    fields = _fields(tangent(d))
    dμ = fields.μ
    dvariance = _fields(fields.Σ).diag
    function diag_logpdf_matrix_pb!!(::NoRData)
        @inbounds for j in eachindex(dy)
            dy_j = dy[j]
            @simd for i in eachindex(dp.μ, variance, dμ, dvariance)
                contribution, dv = _coordinate_gradients(
                    px[i, j] - dp.μ[i], variance[i], dy_j
                )
                dx[i, j] += contribution
                dμ[i] -= contribution
                dvariance[i] += dv
            end
        end
        return NoRData(), NoRData(), NoRData()
    end
    return out, diag_logpdf_matrix_pb!!
end

# Repeated observations from a diagonal normal. `loglikelihood(d, X)` sums over the columns
# of `X`, and the derived rule pays one pullback per column; one pass over the matrix
# replaces all of them. The `-n / variance` half of the log-determinant's derivative is the
# `- variance` inside `_excess`, summed over the columns.
@is_primitive DefaultCtx Tuple{
    typeof(loglikelihood),DiagMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(loglikelihood)}, d::Dual{<:DiagMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    ḋ = _fields(tangent(d))
    μ̇ = ḋ.μ
    v̇ = _fields(ḋ.Σ).diag
    variance = dp.Σ.diag
    mahalanobis = zero(P)
    ẏ = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(dp.μ, variance, μ̇, v̇)
            residual = px[i, j] - dp.μ[i]
            scaled_residual = residual / variance[i]
            mahalanobis += residual * scaled_residual
            ẏ += _coordinate_derivative(residual, variance[i], ẋ[i, j] - μ̇[i], v̇[i])
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + size(px, 2) * sum(log, variance) + mahalanobis)
    return Dual(y, -P(0.5) * ẏ)
end
function rrule!!(
    ::CoDual{typeof(loglikelihood)}, d::CoDual{<:DiagMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    variance = dp.Σ.diag
    mahalanobis = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(dp.μ, variance)
            mahalanobis += abs2(px[i, j] - dp.μ[i]) / variance[i]
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + size(px, 2) * sum(log, variance) + mahalanobis)
    fields = _fields(tangent(d))
    dμ = fields.μ
    dvariance = _fields(fields.Σ).diag
    function diag_loglikelihood_pb!!(dy::P)
        @inbounds for j in axes(px, 2)
            @simd for i in eachindex(dp.μ, variance, dμ, dvariance)
                contribution, dv = _coordinate_gradients(
                    px[i, j] - dp.μ[i], variance[i], dy
                )
                dx[i, j] += contribution
                dμ[i] -= contribution
                dvariance[i] += dv
            end
        end
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), diag_loglikelihood_pb!!
end

# `MvNormal(μ, σ)` and `MvNormal(μ, σ^2 * I)` produce this shape, with one variance shared
# across the dimensions. A `Fill` mean keeps its gradient in rdata, which these passes cannot
# accumulate into, so it stays with the derived rules.
const IsoMvNormal{P} = MvNormal{P,<:ScalMat{P},Vector{P}}

@is_primitive DefaultCtx Tuple{typeof(logpdf),IsoMvNormal{P},Matrix{P}} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{<:IsoMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    ḋ = _fields(tangent(d))
    μ̇ = ḋ.μ
    v̇ = _fields(ḋ.Σ).value
    μ = dp.μ
    variance = dp.Σ.value
    constant = -P(0.5) * length(μ) * (log(P(2π)) + log(variance))
    y = Vector{P}(undef, size(px, 2))
    ẏ = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        derivative = zero(P)
        @simd for i in eachindex(μ, μ̇)
            residual = px[i, j] - μ[i]
            scaled_residual = residual / variance
            mahalanobis += residual * scaled_residual
            derivative += _coordinate_derivative(residual, variance, ẋ[i, j] - μ̇[i], v̇)
        end
        y[j] = constant - P(0.5) * mahalanobis
        ẏ[j] = -P(0.5) * derivative
    end
    return Dual(y, ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:IsoMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    μ = dp.μ
    variance = dp.Σ.value
    constant = -P(0.5) * length(μ) * (log(P(2π)) + log(variance))
    y = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        @simd for i in eachindex(μ)
            mahalanobis += abs2(px[i, j] - μ[i]) / variance
        end
        y[j] = constant - P(0.5) * mahalanobis
    end
    out = zero_fcodual(y)
    dy = tangent(out)
    dμ = _fields(tangent(d)).μ
    function iso_logpdf_matrix_pb!!(::NoRData)
        # The variance is shared, so it is divided once here rather than once per
        # observation, which `_coordinate_gradients` would do.
        excess_total = zero(P)
        @inbounds for j in eachindex(dy)
            dy_j = dy[j]
            @simd for i in eachindex(μ, dμ)
                residual = px[i, j] - μ[i]
                contribution = -dy_j * residual / variance
                dx[i, j] += contribution
                dμ[i] -= contribution
                excess_total += dy_j * _excess(residual, variance)
            end
        end
        dd = RData((
            μ=NoRData(),
            Σ=RData((dim=NoRData(), value=excess_total / variance / (2 * variance))),
        ))
        return NoRData(), dd, NoRData()
    end
    return out, iso_logpdf_matrix_pb!!
end

@is_primitive DefaultCtx Tuple{
    typeof(loglikelihood),IsoMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(loglikelihood)}, d::Dual{<:IsoMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    ḋ = _fields(tangent(d))
    μ̇ = ḋ.μ
    v̇ = _fields(ḋ.Σ).value
    μ = dp.μ
    variance = dp.Σ.value
    mahalanobis = zero(P)
    ẏ = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(μ, μ̇)
            residual = px[i, j] - μ[i]
            scaled_residual = residual / variance
            mahalanobis += residual * scaled_residual
            ẏ += _coordinate_derivative(residual, variance, ẋ[i, j] - μ̇[i], v̇)
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + length(px) * log(variance) + mahalanobis)
    return Dual(y, -P(0.5) * ẏ)
end
function rrule!!(
    ::CoDual{typeof(loglikelihood)}, d::CoDual{<:IsoMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    μ = dp.μ
    variance = dp.Σ.value
    mahalanobis = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(μ)
            mahalanobis += abs2(px[i, j] - μ[i]) / variance
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + length(px) * log(variance) + mahalanobis)
    dμ = _fields(tangent(d)).μ
    function iso_loglikelihood_pb!!(dy::P)
        # The shared variance takes a single cotangent, so it is accumulated here and
        # returned as rdata rather than written into fdata.
        # The variance is shared, so it is divided once here rather than once per
        # observation, which `_coordinate_gradients` would do.
        excess_total = zero(P)
        @inbounds for j in axes(px, 2)
            @simd for i in eachindex(μ, dμ)
                residual = px[i, j] - μ[i]
                contribution = -dy * residual / variance
                dx[i, j] += contribution
                dμ[i] -= contribution
                excess_total += dy * _excess(residual, variance)
            end
        end
        dd = RData((
            μ=NoRData(),
            Σ=RData((dim=NoRData(), value=excess_total / variance / (2 * variance))),
        ))
        return NoRData(), dd, NoRData()
    end
    return zero_fcodual(y), iso_loglikelihood_pb!!
end

# Repeated observations from one dense multivariate Normal are represented by
# `loglikelihood(d, X)`, with observations in the columns of `X`. Keeping the shared
# Cholesky factor at this public boundary avoids tracing one triangular solve per column.
#
# `PDMat`'s third type parameter, the factorisation, exists only from PDMats 0.11.40, and
# Distributions permits 0.11.35 upwards; naming it here fails to parse against the older
# layout and unloads the whole extension. The rules need `chol.factors` to be a `Matrix`,
# which `_factor_tangent` and `_accum_factor!` require by signature, so an exotic
# factorisation raises a `MethodError` rather than being silently mishandled.
const CholeskyMvNormal{P} = MvNormal{P,<:PDMat{P,<:Matrix{P}},<:Vector{P}}

@is_primitive DefaultCtx Tuple{
    typeof(logpdf),CholeskyMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logpdf)}, d::Dual{<:CholeskyMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    L = dp.Σ.chol.L
    ḋ = _fields(tangent(d))
    L̇ = _factor_tangent(dp.Σ.chol, _fields(_fields(ḋ.Σ).chol).factors)
    standardized = L \ (px .- dp.μ)
    perturbed = L \ ((ẋ .- ḋ.μ) - L̇ * standardized)
    constant = -P(0.5) * (length(dp) * log(P(2π)) + logdet(dp.Σ.chol))
    logdet_derivative = sum(i -> L̇[i, i] / L[i, i], axes(L, 1))
    y = Vector{P}(undef, size(px, 2))
    ẏ = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        column = view(standardized, :, j)
        y[j] = constant - P(0.5) * sum(abs2, column)
        ẏ[j] = -logdet_derivative - dot(column, view(perturbed, :, j))
    end
    return Dual(y, ẏ)
end
function rrule!!(
    ::CoDual{typeof(logpdf)}, d::CoDual{<:CholeskyMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
    L = dp.Σ.chol.L
    upper = dp.Σ.chol.uplo === 'U'
    standardized = L \ (px .- dp.μ)
    constant = -P(0.5) * (length(dp) * log(P(2π)) + logdet(dp.Σ.chol))
    y = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        y[j] = constant - P(0.5) * sum(abs2, view(standardized, :, j))
    end
    out = zero_fcodual(y)
    dy = tangent(out)
    fields = _fields(tangent(d))
    dfactors = _fields(_fields(fields.Σ).chol).factors
    function chol_logpdf_matrix_pb!!(::NoRData)
        # Each column carries its own cotangent, so the columns are weighted before the
        # solve rather than after; `loglikelihood` is this with one weight throughout.
        weighted = standardized .* transpose(dy)
        x_gradient = -(L' \ weighted)
        dx .+= x_gradient
        fields.μ .-= vec(sum(x_gradient; dims=2))
        factor_gradient = L' \ (weighted * standardized')
        _accum_factor!(dfactors, factor_gradient, L, one(P), sum(dy), upper)
        return NoRData(), NoRData(), NoRData()
    end
    return out, chol_logpdf_matrix_pb!!
end

@is_primitive DefaultCtx Tuple{
    typeof(loglikelihood),CholeskyMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(loglikelihood)}, d::Dual{<:CholeskyMvNormal{P}}, x::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, ẋ = arrayify(x)
    _check_dims(dp, px)
    L = dp.Σ.chol.L
    ḋ = _fields(tangent(d))
    L̇ = _factor_tangent(dp.Σ.chol, _fields(_fields(ḋ.Σ).chol).factors)
    standardized = L \ (px .- dp.μ)
    n = size(px, 2)
    y =
        -P(0.5) *
        (length(px) * log(P(2π)) + n * logdet(dp.Σ.chol) + sum(abs2, standardized))
    ẏ = -(
        n * sum(i -> L̇[i, i] / L[i, i], axes(L, 1)) +
        dot(standardized, L \ ((ẋ .- ḋ.μ) - L̇ * standardized))
    )
    return Dual(y, ẏ)
end
function rrule!!(
    ::CoDual{typeof(loglikelihood)}, d::CoDual{<:CholeskyMvNormal{P}}, x::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    dp = primal(d)
    px, dx = arrayify(x)
    _check_dims(dp, px)
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
        _accum_factor!(dfactors, factor_gradient, L, dy, dy * n, upper)
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
