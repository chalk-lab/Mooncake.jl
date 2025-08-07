module MooncakeNNlibExt

using GPUArraysCore, NNlib, Random, Mooncake
using Base: IEEEFloat
using NNlib: dropout

using NNlib: conv, depthwiseconv
import Mooncake: @from_rrule, DefaultCtx, MinimalCtx, ReverseMode

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
@from_rrule(MinimalCtx, Tuple{typeof(softmax),SupportedArray{<:IEEEFloat}}, true)
@from_rrule(MinimalCtx, Tuple{typeof(logsoftmax),SupportedArray{<:IEEEFloat}}, true)
@from_rrule(MinimalCtx, Tuple{typeof(logsumexp),SupportedArray{<:IEEEFloat}}, true)
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
