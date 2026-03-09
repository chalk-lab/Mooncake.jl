module MooncakeNNlibExt

using GPUArraysCore, NNlib, Random, Mooncake
using Base: IEEEFloat
using LinearAlgebra
using LinearAlgebra: Adjoint, Transpose
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
    FData

# Array types which we test rules against, so are confident work.
const SupportedArray{P,N} = Union{Array{P,N},AbstractGPUArray{P,N}}

@from_rrule(
    MinimalCtx,
    Tuple{typeof(batched_mul),SupportedArray{P,3},SupportedArray{P,3}} where {P<:IEEEFloat},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(dropout),AbstractRNG,SupportedArray{P},P} where {P<:IEEEFloat},
    true,
)

# Helper accumulator functions
function _accum_fdata!(
    xf::AbstractArray, ::AbstractArray{T}, g::AbstractArray{T}
) where {T<:IEEEFloat}
    return xf .+= g
end
_accum_fdata!(xf::AbstractArray, ::AbstractArray{T}, g::T) where {T<:IEEEFloat} = xf .+= g
function _accum_fdata!(xf::FData, ::Adjoint{T}, g::AbstractArray{T}) where {T<:IEEEFloat}
    return xf.data.parent .+= g'
end
function _accum_fdata!(xf::FData, ::Transpose{T}, g::AbstractArray{T}) where {T<:IEEEFloat}
    return xf.data.parent .+= transpose(g)
end

# logsoftmax rrules
@is_primitive MinimalCtx Tuple{typeof(logsoftmax),AbstractArray{<:IEEEFloat}}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),AbstractArray{<:IEEEFloat}
}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:AbstractArray{<:IEEEFloat}}
)
    xp = primal(x)
    y = logsoftmax(xp)
    res = zero_fcodual(y)
    function logsoftmax_pb!!(::NoRData)
        _accum_fdata!(tangent(x), xp, ∇logsoftmax_data(tangent(res), y; dims=1))
        return NoRData(), NoRData()
    end
    return res, logsoftmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsoftmax)},
    x::CoDual{<:AbstractArray{<:IEEEFloat}},
)
    dims = primal(kw).dims
    xp = primal(x)
    y = logsoftmax(xp; dims)
    res = zero_fcodual(y)
    function logsoftmax_kw_pb!!(::NoRData)
        _accum_fdata!(tangent(x), xp, ∇logsoftmax_data(tangent(res), y; dims))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsoftmax_kw_pb!!
end

# softmax rrules
@is_primitive MinimalCtx Tuple{typeof(softmax),AbstractArray{<:IEEEFloat}}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),AbstractArray{<:IEEEFloat}
}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:AbstractArray{<:IEEEFloat}}
)
    xp = primal(x)
    y = softmax(xp)
    res = zero_fcodual(y)
    function softmax_pb!!(::NoRData)
        _accum_fdata!(tangent(x), xp, ∇softmax_data(tangent(res), y; dims=1))
        return NoRData(), NoRData()
    end
    return res, softmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(softmax)},
    x::CoDual{<:AbstractArray{<:IEEEFloat}},
)
    dims = primal(kw).dims
    xp = primal(x)
    y = softmax(xp; dims)
    res = zero_fcodual(y)
    function softmax_kw_pb!!(::NoRData)
        _accum_fdata!(tangent(x), xp, ∇softmax_data(tangent(res), y; dims))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, softmax_kw_pb!!
end

# logsumexp rrules
@is_primitive MinimalCtx Tuple{typeof(logsumexp),AbstractArray{<:IEEEFloat}}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),AbstractArray{<:IEEEFloat}
}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:AbstractArray{T}}
) where {T<:IEEEFloat}
    xp = primal(x)
    max_ = maximum(xp; init=typemin(T))
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp)
    @fastmath y = max_ + log(s)
    res = zero_fcodual(y)
    function logsumexp_pb!!(dy::T)
        _accum_fdata!(tangent(x), xp, dy .* tmp ./ s)
        return NoRData(), NoRData()
    end
    return res, logsumexp_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{T}},
) where {T<:IEEEFloat}
    dims = primal(kw).dims
    xp = primal(x)
    max_ = maximum(xp; dims, init=typemin(T))
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp; dims)
    @fastmath y = max_ .+ log.(s)
    res = zero_fcodual(y)
    function logsumexp_kw_pb!!(::NoRData)
        _accum_fdata!(tangent(x), xp, tangent(res) .* tmp ./ s)
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
