module MooncakeNNlibExt

using GPUArraysCore, NNlib, Random, Mooncake
using Base: IEEEFloat
using LinearAlgebra
using NNlib:
    conv,
    depthwiseconv,
    ∇logsoftmax_data,
    ∇softmax_data,
    logsoftmax,
    softmax,
    logsumexp,
    dropout

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
    Dual

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
@from_rrule(
    MinimalCtx,
    Tuple{typeof(dropout),AbstractRNG,SupportedArray{P,N},P} where {P<:IEEEFloat,N},
    true,
)

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
        dx .+= ∇logsoftmax_data(tangent(res), y; dims=1)
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
        dx .+= ∇logsoftmax_data(tangent(res), y; dims)
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
        dx .+= ∇softmax_data(tangent(res), y; dims=1)
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
        dx .+= ∇softmax_data(tangent(res), y; dims)
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
    x::CoDual{<:SupportedArray{P,N}},
    b::CoDual{<:SupportedArray{<:IEEEFloat}},
) where {P<:IEEEFloat,N}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    px_copy = copy(px)
    px .+= pb
    # Dims over which b is broadcast (size 1 in b but potentially larger in x).
    broadcast_dims = Tuple(filter(d -> size(pb, d) == 1, 1:N))
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

end
