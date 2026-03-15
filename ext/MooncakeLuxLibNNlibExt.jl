module MooncakeLuxLibNNlibExt

using LuxLib, NNlib, Mooncake, GPUArraysCore
using Base: IEEEFloat
import LuxLib: Impl
import Mooncake:
    MinimalCtx,
    @is_primitive,
    rrule!!,
    CoDual,
    NoRData,
    zero_fcodual,
    primal,
    tangent,
    arrayify

# Define Array support in a hierarchical manner.
const LuxLibNNlibSupportedArray{P} = Union{Array{P,3},AbstractGPUArray{P,3}}

const NNlibBatchedWrapper{P} = Union{
    NNlib.BatchedTranspose{P,<:LuxLibNNlibSupportedArray{P}},
    NNlib.BatchedAdjoint{P,<:LuxLibNNlibSupportedArray{P}},
}

# NNlib.jl specific Mooncake.arrayify dispatches
function Mooncake.arrayify(
    x::NNlib.BatchedTranspose{T,<:LuxLibNNlibSupportedArray{T}}, dx::Mooncake.TangentOrFData
) where {T<:IEEEFloat}
    _, _dx = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
    return x, NNlib.batched_transpose(_dx)
end

function Mooncake.arrayify(
    x::NNlib.BatchedAdjoint{T,<:LuxLibNNlibSupportedArray{T}}, dx::Mooncake.TangentOrFData
) where {T<:IEEEFloat}
    _, _dx = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
    return x, NNlib.batched_adjoint(_dx)
end

@is_primitive MinimalCtx Tuple{
    typeof(Impl.batched_matmul_fallback),
    Union{LuxLibNNlibSupportedArray{P},NNlibBatchedWrapper{P}},
    Union{LuxLibNNlibSupportedArray{P},NNlibBatchedWrapper{P}},
} where {P<:IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(Impl.batched_matmul_fallback)}, x::CoDual{Tx}, y::CoDual{Ty}
) where {
    P<:IEEEFloat,
    Tx<:Union{LuxLibNNlibSupportedArray{P},NNlibBatchedWrapper{P}},
    Ty<:Union{LuxLibNNlibSupportedArray{P},NNlibBatchedWrapper{P}},
}
    px, dx = arrayify(x)
    py, dy = arrayify(y)
    res = zero_fcodual(Impl.batched_matmul_fallback(px, py))
    function batched_matmul_pb!!(::NoRData)
        dout = tangent(res)
        ∂x = let tmp = Impl.batched_matmul_fallback(dout, NNlib.batched_adjoint(py))
            size(px, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        ∂y = let tmp = Impl.batched_matmul_fallback(NNlib.batched_adjoint(px), dout)
            size(py, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        dx .+= ∂x
        dy .+= ∂y
        return NoRData(), NoRData(), NoRData()
    end
    return res, batched_matmul_pb!!
end

end
