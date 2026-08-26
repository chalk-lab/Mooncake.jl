module MooncakeDistributionsExt

using Distributions, Mooncake, LinearAlgebra
using Base: IEEEFloat
using Distributions: sqmahal
using Distributions.FillArrays: Fill
using Distributions.PDMats: ScalMat
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
    Tangent,
    _fields,
    arrayify,
    frule!!,
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
    end
end

end # @static if

#! format: on

end
