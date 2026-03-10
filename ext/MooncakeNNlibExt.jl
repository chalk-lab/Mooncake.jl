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
    arrayify

# Array types which we test rules against, so are confident work.
const SupportedArray{P} = Union{
    Array{P},
    AbstractGPUArray{P},
    Adjoint{P,<:Union{Array{P},AbstractGPUArray{P}}},
    Transpose{P,<:Union{Array{P},AbstractGPUArray{P}}},
}

# On Julia ≤ 1.11, `maximum(x::Adjoint/Transpose; dims, init)` routes through
# `LinearAlgebra.mapreducedim! → switch_dim12 → PermutedDimsArray`, leaving
# type parameters unresolved and causing JET type-stability failures.
# Collecting CPU-backed wrappers to a plain Array avoids that path.
@static if VERSION < v"1.12"
    function _maximum(
        x::Tx, dims, init
    ) where {T<:IEEEFloat,A<:Array{T},Tx<:Union{Adjoint{T,A},Transpose{T,A}}}
        maximum(collect(x); dims, init)
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
    Tuple{typeof(dropout),AbstractRNG,SupportedArray{P},P} where {P<:IEEEFloat},
    true,
)

# logsoftmax rrules
@is_primitive MinimalCtx Tuple{typeof(logsoftmax),SupportedArray{T}} where {T<:IEEEFloat}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),SupportedArray{T}
} where {T<:IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:SupportedArray{T}}
) where {T<:IEEEFloat}
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
    x::CoDual{<:SupportedArray{T}},
) where {T<:IEEEFloat}
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
@is_primitive MinimalCtx Tuple{typeof(softmax),SupportedArray{T}} where {T<:IEEEFloat}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),SupportedArray{T}
} where {T<:IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:SupportedArray{T}}
) where {T<:IEEEFloat}
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
    x::CoDual{<:SupportedArray{T}},
) where {T<:IEEEFloat}
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
@is_primitive MinimalCtx Tuple{typeof(logsumexp),SupportedArray{T}} where {T<:IEEEFloat}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),SupportedArray{T}
} where {T<:IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:SupportedArray{T}}
) where {T<:IEEEFloat}
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
    x::CoDual{<:SupportedArray{T}},
) where {T<:IEEEFloat}
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
    Tuple{typeof(upsample_nearest),SupportedArray{<:IEEEFloat},NTuple{N,Int} where {N}},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.fold),SupportedArray{<:IEEEFloat},NTuple{N,Int} where {N},DenseConvDims
    },
)
@from_rrule(
    MinimalCtx, Tuple{typeof(NNlib.unfold),SupportedArray{<:IEEEFloat},DenseConvDims}
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(NNlib.scatter),Any,SupportedArray,SupportedArray{<:Union{Integer,Tuple}}},
    true,
)
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])

    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($conv),SupportedArray{P},SupportedArray{P},ConvDims
        } where {P<:IEEEFloat},
        true,
    )
    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($∇conv_data),SupportedArray{P},SupportedArray{P},ConvDims
        } where {P<:IEEEFloat},
        true,
    )
end
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(∇conv_filter),SupportedArray{P},SupportedArray{P},ConvDims
    } where {P<:IEEEFloat},
    true,
)
for pool in [:maxpool, :meanpool]
    @eval @from_rrule(
        MinimalCtx, Tuple{typeof($pool),SupportedArray{<:IEEEFloat},PoolDims}, true
    )
end
@from_rrule(MinimalCtx, Tuple{typeof(pad_constant),SupportedArray,Any,Any}, true)

end
