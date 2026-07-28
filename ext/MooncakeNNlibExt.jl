module MooncakeNNlibExt

using NNlib, Random, Mooncake
import NNlib.GPUArraysCore: AbstractGPUArray
using Base: IEEEFloat
using LinearAlgebra
using NNlib: conv, depthwiseconv, logsoftmax, softmax, logsumexp, dropout
using Mooncake.Nfwd: NDual

import Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @is_primitive,
    rrule!!,
    CoDual,
    NoRData,
    zero_fcodual,
    primal,
    tangent,
    arrayify,
    frule!!,
    Dual,
    NoPullback

@inline function _nf_logsumexp_accum(
    grad::NTuple{N,T}, w::T, partials::NTuple{N,T}
) where {N,T}
    return ntuple(k -> grad[k] + w * partials[k], Val(N))
end

@inline function _nf_logsumexp_scale(grad::NTuple{N,T}, inv_sw::T) where {N,T}
    return ntuple(k -> grad[k] * inv_sw, Val(N))
end

@inline function _nf_logsumexp_inf(x::AbstractVector{NDual{T,N}}, u::T) where {T,N}
    count_u = 0
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        if xi.value == u
            count_u += 1
            grad = _nf_logsumexp_accum(grad, one(T), xi.partials)
        end
    end
    return NDual{T,N}(u, _nf_logsumexp_scale(grad, inv(T(count_u))))
end

function NNlib.logsumexp(x::AbstractVector{NDual{T,N}}) where {T<:IEEEFloat,N}
    isempty(x) && return NDual{T,N}(typemin(T))
    u = @inbounds x[begin].value
    @inbounds for i in (firstindex(x) + 1):lastindex(x)
        v = x[i].value
        v > u && (u = v)
    end
    isinf(u) && return _nf_logsumexp_inf(x, u)
    sum_w = zero(T)
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        w = exp(xi.value - u)
        sum_w += w
        grad = _nf_logsumexp_accum(grad, w, xi.partials)
    end
    y_val = u + log(sum_w)
    return NDual{T,N}(y_val, _nf_logsumexp_scale(grad, inv(sum_w)))
end

# Array types which we test rules against, so are confident work.
# Parametric on both element type P and dimensionality N.
const SupportedArray{P,N} = Union{
    Array{P,N},
    AbstractGPUArray{P,N},
    Adjoint{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
    Transpose{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
}

# On Julia ≤ 1.11, `maximum(x::Adjoint/Transpose; dims, init)` routes through
# `LinearAlgebra.mapreducedim! → switch_dim12 → PermutedDimsArray`, leaving
# type parameters unresolved and causing JET type-stability failures.
# Collecting CPU-backed wrappers to a plain Array avoids that path.
@static if VERSION < v"1.12"
    function _maximum(
        x::Tx, dims, init
    ) where {T<:IEEEFloat,A<:Array{T},Tx<:Union{Adjoint{T,A},Transpose{T,A}}}
        return maximum(collect(x); dims, init)
    end
end
_maximum(x, dims, init) = maximum(x; dims, init)

@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(batched_mul),
        Union{Array{P,3},AbstractGPUArray{P,3}},
        Union{Array{P,3},AbstractGPUArray{P,3}},
    } where {P<:IEEEFloat},
)
# `dropout` returns its input array itself when `p ≤ 0`: the fast path is
# `convert(AbstractArray{float(eltype(A))}, A)`, which for a float array is the identity. So the
# output aliases the input, while ChainRules' rrule has no such path — it always allocates, and
# always draws from `rng`. AD therefore ran a different program from the primal: mutating the
# result no longer wrote through to the input, changing the value as well as the gradient, and
# the RNG advanced where the primal left it untouched. Returning the input `CoDual` unchanged
# keeps the aliasing invariant, skips the draw, and matches the `p > 0` path on `p` itself,
# for which ChainRules reports `NoTangent()`.
@is_primitive MinimalCtx Tuple{
    typeof(dropout),AbstractRNG,SupportedArray{P,N},P
} where {P<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(dropout),AbstractRNG,SupportedArray{P,N},P
} where {P<:IEEEFloat,N}

function rrule!!(
    f::CoDual{typeof(dropout)},
    rng::CoDual{<:AbstractRNG},
    x::CoDual{<:SupportedArray{P,N}},
    p::CoDual{P},
) where {P<:IEEEFloat,N}
    primal(p) > 0 && return Mooncake.rrule_wrapper(f, rng, x, p)
    return x, NoPullback(f, rng, x, p)
end

function rrule!!(
    kwcall::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    f::CoDual{typeof(dropout)},
    rng::CoDual{<:AbstractRNG},
    x::CoDual{<:SupportedArray{P,N}},
    p::CoDual{P},
) where {P<:IEEEFloat,N}
    primal(p) > 0 && return Mooncake.rrule_wrapper(kwcall, kw, f, rng, x, p)
    return x, NoPullback(kwcall, kw, f, rng, x, p)
end

# logsoftmax rrules
@is_primitive MinimalCtx Tuple{
    typeof(logsoftmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    y = logsoftmax(xp)
    res = zero_fcodual(y)
    function logsoftmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        # TODO: Drop the fallback once NNlib >= 0.9.37 is more widely supported.
        # Until then, use the public softmax backpass API when available and delegate
        # NNlib < 0.9.37 to the legacy `_data` helpers.
        # See https://github.com/chalk-lab/Mooncake.jl/pull/1229 for more context.
        @static if hasmethod(NNlib.∇logsoftmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇logsoftmax(dy, y; dims=1)
        else
            dx .+= NNlib.∇logsoftmax_data(dy, y; dims=1)
        end
        return NoRData(), NoRData()
    end
    return res, logsoftmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsoftmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = logsoftmax(xp; dims)
    res = zero_fcodual(y)
    function logsoftmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇logsoftmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇logsoftmax(dy, y; dims)
        else
            dx .+= NNlib.∇logsoftmax_data(dy, y; dims)
        end
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsoftmax_kw_pb!!
end

# softmax rrules
@is_primitive MinimalCtx Tuple{typeof(softmax),SupportedArray{T,N}} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    y = softmax(xp)
    res = zero_fcodual(y)
    function softmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇softmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇softmax(dy, y; dims=1)
        else
            dx .+= NNlib.∇softmax_data(dy, y; dims=1)
        end
        return NoRData(), NoRData()
    end
    return res, softmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(softmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = softmax(xp; dims)
    res = zero_fcodual(y)
    function softmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        @static if hasmethod(NNlib.∇softmax, Tuple{AbstractArray,AbstractArray})
            dx .+= NNlib.∇softmax(dy, y; dims)
        else
            dx .+= NNlib.∇softmax_data(dy, y; dims)
        end
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, softmax_kw_pb!!
end

# logsumexp rrules
@is_primitive MinimalCtx Tuple{typeof(logsumexp),SupportedArray{T,N}} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    max_ = maximum(xp; init=typemin(T))
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp)
    @fastmath y = max_ + log(s)
    res = zero_fcodual(y)
    function logsumexp_pb!!(dy::T)
        _, dx = arrayify(x)
        dx .+= dy .* tmp ./ s
        return NoRData(), NoRData()
    end
    return res, logsumexp_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    max_ = _maximum(xp, dims, typemin(T))
    # avoids Inf instability when xp[i]==max_==Inf
    @fastmath tmp = ifelse.(xp .== max_, one(T), exp.(xp .- max_))
    s = sum(tmp; dims)
    @fastmath y = max_ .+ log.(s)
    res = zero_fcodual(y)
    function logsumexp_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dx .+= tangent(res) .* tmp ./ s
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsumexp_kw_pb!!
end

@from_rrule(
    MinimalCtx,
    Tuple{typeof(upsample_nearest),SupportedArray{<:IEEEFloat,N},NTuple{M,Int}} where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.fold),SupportedArray{<:IEEEFloat,N},NTuple{M,Int},DenseConvDims
    } where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(NNlib.unfold),SupportedArray{<:IEEEFloat,N},DenseConvDims} where {N},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.scatter),
        Any,
        SupportedArray{P,N},
        SupportedArray{<:Union{Integer,Tuple},M},
    } where {P,N,M},
    true,
)

# ChainRules' `∇scatter_src` for `max`/`min` is `(src .== gather(dst, idx)) .* gather(Δ, idx)`,
# giving the full cotangent to every source tied for its destination's extremum. The gradient
# then sums to the tie multiplicity rather than to 1 — measured 2x for two tied entries, 3x for
# three — so it is not a member of the subdifferential at all, as opposed to a debatable choice
# among its members. Dividing by the number of tied entries restores the sum and picks the
# symmetric member, which is the one central differences agree with. `max(ntied, 1)` only guards
# the division: an `init` above every source leaves a destination with no tied entry, and there
# the mask is already zero.
@inline function _scatter_extremum_dsrc!(
    dsrc, psrc::AbstractArray{P}, pidx, y, dy
) where {P}
    tied = psrc .== NNlib.gather(y, pidx)
    ntied = NNlib.gather(NNlib.scatter(+, P.(tied), pidx), pidx)
    dsrc .+= tied .* NNlib.gather(dy, pidx) ./ max.(ntied, one(P))
    return nothing
end

function rrule!!(
    ::CoDual{typeof(NNlib.scatter)},
    op::CoDual{<:Union{typeof(max),typeof(min)}},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    psrc, dsrc = arrayify(src)
    pidx = primal(idx)
    res = zero_fcodual(NNlib.scatter(primal(op), psrc, pidx))
    function scatter_extremum_pb!!(::NoRData)
        _scatter_extremum_dsrc!(dsrc, psrc, pidx, primal(res), tangent(res))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, scatter_extremum_pb!!
end

function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(NNlib.scatter)},
    op::CoDual{<:Union{typeof(max),typeof(min)}},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    psrc, dsrc = arrayify(src)
    pidx = primal(idx)
    res = zero_fcodual(NNlib.scatter(primal(op), psrc, pidx; primal(kw)...))
    function scatter_extremum_kw_pb!!(::NoRData)
        _scatter_extremum_dsrc!(dsrc, psrc, pidx, primal(res), tangent(res))
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, scatter_extremum_kw_pb!!
end

# ChainRules defines an `rrule` only for the in-place `gather!`, not `gather`, so without
# this AD would trace `gather`'s scalar per-index loop (see
# https://github.com/chalk-lab/Mooncake.jl/issues/1234). The pullback is `∇gather_src`
# accumulating straight into `src`'s fdata, avoiding its intermediate allocation.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(NNlib.gather),SupportedArray{P,N},SupportedArray{<:Union{Integer,Tuple},M}
    } where {P<:IEEEFloat,N,M},
)
function Mooncake.rrule!!(
    ::CoDual{typeof(NNlib.gather)},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    pidx = primal(idx)
    res = zero_fcodual(NNlib.gather(primal(src), pidx))
    function gather_pb!!(::NoRData)
        _, dsrc = arrayify(src)
        NNlib.scatter!(+, dsrc, tangent(res), pidx)
        return NoRData(), NoRData(), NoRData()
    end
    return res, gather_pb!!
end
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])

    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($conv),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($∇conv_data),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
end
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(∇conv_filter),SupportedArray{P,N},SupportedArray{P,M},ConvDims
    } where {P<:IEEEFloat,N,M},
    true,
)
for pool in [:maxpool, :meanpool]
    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof($pool),SupportedArray{<:IEEEFloat,N},PoolDims} where {N},
        true,
    )
end
@from_rrule(
    MinimalCtx, Tuple{typeof(pad_constant),SupportedArray{P,N},Any,Any} where {P,N}, true,
)

# Direct rules for bias_act!(identity, x, b) on CPU and GPU arrays.
# bias_act! modifies x in-place (x .+= b), so we save x's primal before mutation,
# compute in-place, return x as output, and restore x's primal in the pullback.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(bias_act!),
        typeof(identity),
        SupportedArray{<:IEEEFloat,N} where {N},
        SupportedArray{<:IEEEFloat,M} where {M},
    },
)
function frule!!(
    ::Dual{typeof(bias_act!)},
    ::Dual{typeof(identity)},
    x::Dual{<:SupportedArray{<:IEEEFloat,N}},
    b::Dual{<:SupportedArray{<:IEEEFloat,M}},
) where {N,M}
    primal(x) .+= primal(b)
    tangent(x) .+= tangent(b)
    return x
end
function rrule!!(
    ::CoDual{typeof(bias_act!)},
    ::CoDual{typeof(identity)},
    x::CoDual{<:SupportedArray{P}},
    b::CoDual{<:SupportedArray{<:IEEEFloat}},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    px_copy = copy(px)
    px .+= pb
    # Dims over which b is broadcast (size 1 in b but potentially larger in x).
    broadcast_dims = Tuple(filter(d -> size(pb, d) == 1, 1:ndims(px)))
    function bias_act_id_pb!!(::NoRData)
        if isempty(broadcast_dims)
            db .+= dx
        else
            db .+= reshape(sum(dx; dims=broadcast_dims), size(pb))
        end
        copyto!(px, px_copy)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, bias_act_id_pb!!
end

# NNlib computes the smooth `σ` in overflow-safe form, `t = exp(-abs(x));
# ifelse(x ≥ 0, inv(1 + t), t / (1 + t))`. AD through that implementation picks up a factor
# of `sign(x)`, which is `0` at `x == 0`, so it reports `0` there instead of `1/4`: the kink
# `abs` introduces cancels between the two branches analytically, but not via the chain
# rule. Exactly-zero arguments are common (zero-initialised biases, dead ReLU units).
for f in (:σ, :sigmoid_fast)
    @eval @is_primitive MinimalCtx Tuple{typeof(NNlib.$f),P} where {P<:IEEEFloat}
    @eval function frule!!(::Dual{typeof(NNlib.$f)}, x::Dual{P}) where {P<:IEEEFloat}
        Ω = NNlib.$f(primal(x))
        return Dual(Ω, tangent(x) * Ω * (one(P) - Ω))
    end
    @eval function rrule!!(::CoDual{typeof(NNlib.$f)}, x::CoDual{P}) where {P<:IEEEFloat}
        Ω = NNlib.$f(primal(x))
        sigmoid_pb!!(dΩ::P) = NoRData(), dΩ * Ω * (one(P) - Ω)
        return zero_fcodual(Ω), sigmoid_pb!!
    end
    # GPU elementwise and reduction rules evaluate the whole fused broadcast on `NDual`s
    # inside one kernel, so they never reach the rules above and need the derivative too.
    @eval @inline function NNlib.$f(x::NDual{P,N}) where {P<:IEEEFloat,N}
        Ω = NNlib.$f(x.value)
        d = Ω * (one(P) - Ω)
        return NDual{P,N}(Ω, ntuple(i -> x.partials[i] * d, Val(N)))
    end
end

end
