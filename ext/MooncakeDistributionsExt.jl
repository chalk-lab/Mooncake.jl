module MooncakeDistributionsExt

using Distributions, Mooncake, LinearAlgebra
using Base: IEEEFloat
using Distributions: ArrayLikeVariate, Sampleable, loglikelihood, sqmahal
using Distributions.FillArrays: Fill
using Distributions.PDMats: PDiagMat, PDMat, ScalMat
using PrecompileTools: @setup_workload, @compile_workload
using Random: AbstractRNG, rand!

import Mooncake:
    @is_primitive,
    CoDual,
    ContiguousSubVector,
    DefaultCtx,
    Lifted,
    MinimalCtx,
    Nfwd,
    NoFData,
    NoRData,
    RData,
    ReverseMode,
    Tangent,
    _arrayify_lane,
    _fields,
    _scalar_ndual,
    arrayify,
    frule!!,
    increment!!,
    primal,
    rrule!!,
    tangent,
    zero_dual,
    zero_fcodual,
    zero_rdata

const ArrayLikeSampleable = Sampleable{<:ArrayLikeVariate}
const DenseFloatArray = DenseArray{<:IEEEFloat}

# A draw is random state, not a reparameterised sample. The result aliases `x`, so clear
# its fdata and restore both arrays in reverse, but leave the RNG advanced.
@is_primitive MinimalCtx Tuple{
    typeof(rand!),AbstractRNG,<:ArrayLikeSampleable,<:DenseFloatArray
}
function frule!!(
    ::Lifted{typeof(rand!),Nw},
    rng::Lifted{<:AbstractRNG,Nw},
    sampler::Lifted{<:ArrayLikeSampleable,Nw},
    x::Lifted{P,Nw},
) where {P<:DenseFloatArray,Nw}
    px, dx = arrayify(x)
    rand!(primal(rng), primal(sampler), px)
    for lane in 1:Nw
        fill!(dx[lane], zero(eltype(P)))
    end
    return x
end
function rrule!!(
    ::CoDual{typeof(rand!)},
    rng::CoDual{<:AbstractRNG},
    sampler::CoDual{<:ArrayLikeSampleable},
    x::CoDual{P,P},
) where {P<:DenseFloatArray}
    px, dx = primal(x), tangent(x)
    px_copy = copy(px)
    dx_copy = copy(dx)
    rand!(primal(rng), primal(sampler), px)
    fill!(dx, zero(eltype(P)))
    rng_rdata = zero_rdata(primal(rng))
    sampler_rdata = zero_rdata(primal(sampler))
    function rand!_pb!!(::NoRData)
        copyto!(px, px_copy)
        copyto!(dx, dx_copy)
        return NoRData(), rng_rdata, sampler_rdata, NoRData()
    end
    return x, rand!_pb!!
end

# The rules below exist purely to work around performance limitations of Mooncake.jl: the
# derived rules are correct, but slow. As in `src/rules/performance_patches.jl`, each
# signature covers a finite set of concrete types, all of which are tested.

# Both rules below work in `z = (x - μ) / σ` rather than `σ^2`: squaring flushes the
# derivative to `Inf` for `σ` well inside the range where it is representable
# (`σ < 1e-20` in `Float32`). Dividing by `σ` rather than scaling by `inv(σ)` matters for
# the same reason: `inv(σ)` overflows once `σ < 1 / floatmax(P)`, where `(x - μ) / σ` and
# the derivative are both still finite.
@is_primitive DefaultCtx ReverseMode Tuple{typeof(logpdf),Normal{P},P} where {P<:IEEEFloat}
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

# `sum(f, collection)` reaches `Base.mapreduce_empty` by a runtime dispatch on Julia 1.10,
# so a rule that reduces that way is not inferable there — including through
# `logpdf(::Product, x)`, which reduces the same way internally. These two accumulate
# explicitly instead.
function _sum_logpdf(dists, x, ::Type{P}) where {P}
    y = zero(P)
    @inbounds for i in eachindex(x, dists)
        y += logpdf(dists[i], x[i])
    end
    return y
end

function _logdet_derivative(L, L̇)
    total = zero(eltype(L))
    @inbounds for i in axes(L, 1)
        total += L̇[i, i] / L[i, i]
    end
    return total
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
function _check_dims(d::Union{MvNormal,Distributions.Product}, x::AbstractVector)
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

# A `Fill` mean has a single tangent value shared by every element.
_mean_tangent(μ̇::Tangent, ::Int) = _fields(μ̇).value
_mean_tangent(μ̇::Vector, i::Int) = @inbounds μ̇[i]

# Forward analogues, hoisted out of the element loop: lane `k` of the mean's tangent, as
# something indexable by element. A `Fill` mean shares one value, so its lane is a scalar
# wrapped to answer any index.
struct _ConstLane{P}
    value::P
end
Base.@propagate_inbounds Base.getindex(c::_ConstLane, ::Int) = c.value
_mean_lane(μ̇::Nfwd.NDualArray, k::Int) = Nfwd.tangent_view(μ̇, k)
_mean_lane(μ̇, k::Int) = _ConstLane(μ̇.fields.value.partials[k])

# `arrayify` is restricted to `BlasFloat`, but these rules are claimed for every `IEEEFloat`,
# `Float16` included. `_arrayify_lane` is generic over the dual eltypes, so the lanes come from it.
function _lanes(x, ::Val{N}) where {N}
    return ntuple(k -> _arrayify_lane(primal(x), tangent(x), k), Val(N))
end

# A `Vector` mean carries its gradient in fdata; a `Fill` mean carries it in rdata, which
# leaves the distribution with no fdata at all.
_mean_fdata(::CoDual{<:ScalMvNormal,NoFData}) = NoFData()
_mean_fdata(d::CoDual{<:ScalMvNormal}) = _fields(tangent(d)).μ

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

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(sqmahal),ScalMvNormal{P},DenseVec{P}
} where {P<:IEEEFloat}
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
    ::Lifted{typeof(logpdf),N}, d::Lifted{<:DiagMvNormal{P},N}, x::Lifted{<:DenseVec{P},N}
) where {N,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(N))
    _check_dims(dp, px)
    ḋ = tangent(d).fields
    variance = dp.Σ.diag
    y = zero(P)
    @inbounds @simd for i in eachindex(px, dp.μ, variance)
        residual = px[i] - dp.μ[i]
        y += log(variance[i]) + residual * (residual / variance[i])
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + y)
    lanes = ntuple(Val(N)) do k
        ẋ = ẋs[k]
        μ̇ = _mean_lane(ḋ.μ, k)
        v̇ = Nfwd.tangent_view(ḋ.Σ.fields.diag, k)
        acc = zero(P)
        @inbounds @simd for i in eachindex(px, dp.μ, variance)
            residual = px[i] - dp.μ[i]
            acc += _coordinate_derivative(residual, variance[i], ẋ[i] - μ̇[i], v̇[i])
        end
        -P(0.5) * acc
    end
    return Lifted{P,N}(y, _scalar_ndual(y, lanes))
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
_fwd_dists(::Distributions.ProductDistribution, dd) = dd.fields.dists

@is_primitive DefaultCtx Tuple{
    typeof(logpdf),NormalProduct{P,N},Array{P,N}
} where {P<:IEEEFloat,N}
function frule!!(
    ::Lifted{typeof(logpdf),Nw},
    d::Lifted{<:NormalProduct{P,N},Nw},
    x::Lifted{Array{P,N},Nw},
) where {Nw,P<:IEEEFloat,N}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(Nw))
    _check_dims(dp, px)
    dists = _dists(dp)
    ḋists = _fwd_dists(dp, tangent(d))
    y = zero(P)
    @inbounds for i in eachindex(px, dists)
        dist = dists[i]
        y -= P(0.5) * log(P(2π)) + log(dist.σ) + P(0.5) * abs2((px[i] - dist.μ) / dist.σ)
    end
    # A `σ == 0` component makes the fused sum `Inf - Inf`, where the primal's `iszero(σ)`
    # branch gives ±Inf. Recomputing only when the sum is not a number leaves the loop above
    # vectorisable; the derivative agrees with the derived rule either way.
    isnan(y) && (y = logpdf(dp, px))
    lanes = ntuple(Val(Nw)) do k
        ẋ = ẋs[k]
        acc = zero(P)
        @inbounds for i in eachindex(px, dists, ḋists)
            dist = dists[i]
            ḋ = ḋists[i].fields
            z = (px[i] - dist.μ) / dist.σ
            acc +=
                ((abs2(z) - one(P)) * ḋ.σ.partials[k] - z * (ẋ[i] - ḋ.μ.partials[k])) /
                dist.σ
        end
        acc
    end
    return Lifted{P,Nw}(y, _scalar_ndual(y, lanes))
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
    # As in the `frule!!` above: recompute only where the fused sum is `Inf - Inf`.
    isnan(y) && (y = logpdf(dp, px))
    ddists = _dists(dp, tangent(d))
    function normal_product_logpdf_pb!!(dy::P)
        @inbounds for i in eachindex(px, dists, ddists)
            dist = dists[i]
            z = (px[i] - dist.μ) / dist.σ
            dx_i = -dy * z / dist.σ
            dx[i] += dx_i
            ddists[i] = increment!!(
                ddists[i], Tangent((μ=(-dx_i), σ=dy * (abs2(z) - one(P)) / dist.σ))
            )
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
_fwd_dists(::Distributions.Product, dd) = dd.fields.v

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
    ::Lifted{typeof(logpdf),N},
    d::Lifted{<:CountingProduct{P},N},
    x::Lifted{<:AbstractVector{<:Integer},N},
) where {N,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    _check_dims(dp, px)
    dists = _dists(dp)
    ḋists = _fwd_dists(dp, tangent(d))
    y = _sum_logpdf(dists, px, P)
    lanes = ntuple(Val(N)) do k
        acc = zero(P)
        @inbounds for i in eachindex(px, dists, ḋists)
            insupport(dists[i], px[i]) || continue
            acc +=
                _param_derivative(dists[i], px[i]) *
                only(values(ḋists[i].fields)).partials[k]
        end
        acc
    end
    return Lifted{P,N}(y, _scalar_ndual(y, lanes))
end
function rrule!!(
    ::CoDual{typeof(logpdf)},
    d::CoDual{<:CountingProduct{P}},
    x::CoDual{<:AbstractVector{<:Integer}},
) where {P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    _check_dims(dp, px)
    dists = _dists(dp)
    ddists = _dists(dp, tangent(d))
    y = _sum_logpdf(dists, px, P)
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
    ::Lifted{typeof(logpdf),Nw}, d::Lifted{<:DiagMvNormal{P},Nw}, x::Lifted{Matrix{P},Nw}
) where {Nw,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(Nw))
    _check_dims(dp, px)
    ḋ = tangent(d).fields
    variance = dp.Σ.diag
    constant = -P(0.5) * (length(dp) * log(P(2π)) + sum(log, variance))
    y = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        @simd for i in eachindex(dp.μ, variance)
            residual = px[i, j] - dp.μ[i]
            mahalanobis += residual * (residual / variance[i])
        end
        y[j] = constant - P(0.5) * mahalanobis
    end
    V = zero_dual(Val(Nw), y)
    blk = getfield(V, :partials_block)
    for k in 1:Nw
        ẋ = ẋs[k]
        μ̇ = _mean_lane(ḋ.μ, k)
        v̇ = Nfwd.tangent_view(ḋ.Σ.fields.diag, k)
        lane = view(blk, k, :)
        @inbounds for j in axes(px, 2)
            derivative = zero(P)
            @simd for i in eachindex(dp.μ, variance)
                residual = px[i, j] - dp.μ[i]
                derivative += _coordinate_derivative(
                    residual, variance[i], ẋ[i, j] - μ̇[i], v̇[i]
                )
            end
            lane[j] = -P(0.5) * derivative
        end
    end
    return Lifted{Vector{P},Nw}(y, V)
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
    ::Lifted{typeof(loglikelihood),Nw},
    d::Lifted{<:DiagMvNormal{P},Nw},
    x::Lifted{Matrix{P},Nw},
) where {Nw,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(Nw))
    _check_dims(dp, px)
    ḋ = tangent(d).fields
    variance = dp.Σ.diag
    mahalanobis = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(dp.μ, variance)
            residual = px[i, j] - dp.μ[i]
            mahalanobis += residual * (residual / variance[i])
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + size(px, 2) * sum(log, variance) + mahalanobis)
    lanes = ntuple(Val(Nw)) do k
        ẋ = ẋs[k]
        μ̇ = _mean_lane(ḋ.μ, k)
        v̇ = Nfwd.tangent_view(ḋ.Σ.fields.diag, k)
        acc = zero(P)
        @inbounds for j in axes(px, 2)
            @simd for i in eachindex(dp.μ, variance)
                residual = px[i, j] - dp.μ[i]
                acc += _coordinate_derivative(residual, variance[i], ẋ[i, j] - μ̇[i], v̇[i])
            end
        end
        -P(0.5) * acc
    end
    return Lifted{P,Nw}(y, _scalar_ndual(y, lanes))
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
    ::Lifted{typeof(logpdf),Nw}, d::Lifted{<:IsoMvNormal{P},Nw}, x::Lifted{Matrix{P},Nw}
) where {Nw,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(Nw))
    _check_dims(dp, px)
    ḋ = tangent(d).fields
    μ = dp.μ
    variance = dp.Σ.value
    constant = -P(0.5) * length(μ) * (log(P(2π)) + log(variance))
    y = Vector{P}(undef, size(px, 2))
    @inbounds for j in axes(px, 2)
        mahalanobis = zero(P)
        @simd for i in eachindex(μ)
            residual = px[i, j] - μ[i]
            mahalanobis += residual * (residual / variance)
        end
        y[j] = constant - P(0.5) * mahalanobis
    end
    V = zero_dual(Val(Nw), y)
    blk = getfield(V, :partials_block)
    for k in 1:Nw
        ẋ = ẋs[k]
        μ̇ = _mean_lane(ḋ.μ, k)
        v̇ = ḋ.Σ.fields.value.partials[k]
        lane = view(blk, k, :)
        @inbounds for j in axes(px, 2)
            derivative = zero(P)
            @simd for i in eachindex(μ)
                residual = px[i, j] - μ[i]
                derivative += _coordinate_derivative(
                    residual, variance, ẋ[i, j] - μ̇[i], v̇
                )
            end
            lane[j] = -P(0.5) * derivative
        end
    end
    return Lifted{Vector{P},Nw}(y, V)
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
    ::Lifted{typeof(loglikelihood),Nw},
    d::Lifted{<:IsoMvNormal{P},Nw},
    x::Lifted{Matrix{P},Nw},
) where {Nw,P<:IEEEFloat}
    dp = primal(d)
    px = primal(x)
    ẋs = _lanes(x, Val(Nw))
    _check_dims(dp, px)
    ḋ = tangent(d).fields
    μ = dp.μ
    variance = dp.Σ.value
    mahalanobis = zero(P)
    @inbounds for j in axes(px, 2)
        @simd for i in eachindex(μ)
            residual = px[i, j] - μ[i]
            mahalanobis += residual * (residual / variance)
        end
    end
    y = -P(0.5) * (length(px) * log(P(2π)) + length(px) * log(variance) + mahalanobis)
    lanes = ntuple(Val(Nw)) do k
        ẋ = ẋs[k]
        μ̇ = _mean_lane(ḋ.μ, k)
        v̇ = ḋ.Σ.fields.value.partials[k]
        acc = zero(P)
        @inbounds for j in axes(px, 2)
            @simd for i in eachindex(μ)
                residual = px[i, j] - μ[i]
                acc += _coordinate_derivative(residual, variance, ẋ[i, j] - μ̇[i], v̇)
            end
        end
        -P(0.5) * acc
    end
    return Lifted{P,Nw}(y, _scalar_ndual(y, lanes))
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

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(logpdf),CholeskyMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
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

@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(loglikelihood),CholeskyMvNormal{P},Matrix{P}
} where {P<:IEEEFloat}
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
