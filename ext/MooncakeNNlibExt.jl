module MooncakeNNlibExt

using GPUArraysCore, NNlib, Random, Mooncake
using Base: IEEEFloat
using NNlib: dropout
using LinearAlgebra

using NNlib: conv, depthwiseconv
import Mooncake: @from_rrule, DefaultCtx, MinimalCtx

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

# Helper accumulater functions
_accum_fdata!(xf::AbstractArray, ::Array{T}, g::Array{T}) where {T<:IEEEFloat} = xf .+= g
_accum_fdata!(xf::AbstractArray, ::Array{T}, g::T) where {T<:IEEEFloat} = xf .+= g
function _accum_fdata!(
    xf::Mooncake.FData, ::LinearAlgebra.Adjoint, g::Array{T}
) where {T<:IEEEFloat}
    return xf.data.parent .+= g'
end
function _accum_fdata!(
    xf::Mooncake.FData, ::LinearAlgebra.Transpose, g::Array{T}
) where {T<:IEEEFloat}
    return xf.data.parent .+= transpose(g)
end

# logsoftmax rrules
Mooncake.@is_primitive MinimalCtx Tuple{typeof(logsoftmax),AbstractArray{<:IEEEFloat}}
Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),AbstractArray{<:IEEEFloat}
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(logsoftmax)}, x::Mooncake.CoDual{<:AbstractArray{<:IEEEFloat}}
)
    xp = Mooncake.primal(x)
    y = logsoftmax(xp)
    res = Mooncake.zero_fcodual(y)
    function logsoftmax_pb!!(::Mooncake.NoRData)
        _accum_fdata!(
            Mooncake.tangent(x),
            xp,
            NNlib.∇logsoftmax_data(Mooncake.tangent(res), y; dims=1),
        )
        return Mooncake.NoRData(), Mooncake.NoRData()
    end
    return res, logsoftmax_pb!!
end

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Core.kwcall)},
    kw::Mooncake.CoDual{<:NamedTuple{(:dims,)}},
    ::Mooncake.CoDual{typeof(logsoftmax)},
    x::Mooncake.CoDual{<:AbstractArray{<:IEEEFloat}},
)
    dims = Mooncake.primal(kw).dims
    xp = Mooncake.primal(x)
    y = logsoftmax(xp; dims)
    res = Mooncake.zero_fcodual(y)
    function logsoftmax_kw_pb!!(::Mooncake.NoRData)
        _accum_fdata!(
            Mooncake.tangent(x), xp, NNlib.∇logsoftmax_data(Mooncake.tangent(res), y; dims)
        )
        return Mooncake.NoRData(),
        Mooncake.NoRData(), Mooncake.NoRData(),
        Mooncake.NoRData()
    end
    return res, logsoftmax_kw_pb!!
end

# softmax rrules
Mooncake.@is_primitive MinimalCtx Tuple{typeof(softmax),AbstractArray{<:IEEEFloat}}
Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),AbstractArray{<:IEEEFloat}
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(softmax)}, x::Mooncake.CoDual{<:AbstractArray{<:IEEEFloat}}
)
    xp = Mooncake.primal(x)
    y = softmax(xp)
    res = Mooncake.zero_fcodual(y)
    function softmax_pb!!(::Mooncake.NoRData)
        _accum_fdata!(
            Mooncake.tangent(x), xp, NNlib.∇softmax_data(Mooncake.tangent(res), y; dims=1)
        )
        return Mooncake.NoRData(), Mooncake.NoRData()
    end
    return res, softmax_pb!!
end

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Core.kwcall)},
    kw::Mooncake.CoDual{<:NamedTuple{(:dims,)}},
    ::Mooncake.CoDual{typeof(softmax)},
    x::Mooncake.CoDual{<:AbstractArray{<:IEEEFloat}},
)
    dims = Mooncake.primal(kw).dims
    xp = Mooncake.primal(x)
    y = softmax(xp; dims)
    res = Mooncake.zero_fcodual(y)
    function softmax_kw_pb!!(::Mooncake.NoRData)
        _accum_fdata!(
            Mooncake.tangent(x), xp, NNlib.∇softmax_data(Mooncake.tangent(res), y; dims)
        )
        return Mooncake.NoRData(),
        Mooncake.NoRData(), Mooncake.NoRData(),
        Mooncake.NoRData()
    end
    return res, softmax_kw_pb!!
end

# logsumexp rrules
Mooncake.@is_primitive MinimalCtx Tuple{typeof(logsumexp),AbstractArray{<:IEEEFloat}}
Mooncake.@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),AbstractArray{<:IEEEFloat}
}

# Scalar output (no dims kwarg)
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(logsumexp)}, x::Mooncake.CoDual{<:AbstractArray{T}}
) where {T<:IEEEFloat}
    xp = Mooncake.primal(x)
    max_ = maximum(xp)
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp)
    @fastmath y = max_ + log(s)
    res = Mooncake.zero_fcodual(y)

    function logsumexp_pb!!(dy::T)
        _accum_fdata!(Mooncake.tangent(x), xp, dy .* tmp ./ s)
        return Mooncake.NoRData(), Mooncake.NoRData()
    end
    return res, logsumexp_pb!!
end

# Array output (with dims kwarg)
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Core.kwcall)},
    kw::Mooncake.CoDual{<:NamedTuple{(:dims,)}},
    ::Mooncake.CoDual{typeof(logsumexp)},
    x::Mooncake.CoDual{<:AbstractArray{T}},
) where {T<:IEEEFloat}
    dims = Mooncake.primal(kw).dims
    xp = Mooncake.primal(x)
    max_ = maximum(xp; dims)
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp; dims)
    @fastmath y = max_ .+ log.(s)
    res = Mooncake.zero_fcodual(y)

    function logsumexp_kw_pb!!(::Mooncake.NoRData)
        _accum_fdata!(Mooncake.tangent(x), xp, Mooncake.tangent(res) .* tmp ./ s)
        return Mooncake.NoRData(),
        Mooncake.NoRData(), Mooncake.NoRData(),
        Mooncake.NoRData()
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
