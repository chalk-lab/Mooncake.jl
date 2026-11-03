module MooncakeCUDABFloat16sExt

using Mooncake, CUDA, BFloat16s, LinearAlgebra
using CUDA.CUDACore.GPUArrays: derive
using Base.Broadcast: Broadcasted

using Mooncake: MinimalCtx, DefaultCtx, @is_primitive, primal
using Mooncake: Dual, CoDual
import Mooncake: frule!!, rrule!!, tangent_type

# Statistics makes MooncakeCUDAExt's triggers a strict subset, as required by extension precompilation.
const CUDAExt = Base.get_extension(Mooncake, :MooncakeCUDAExt)
using .CUDAExt: _throw_gpu_argument_error, _CuSliceIndex

const CuBFloatArray = CuArray{BFloat16}
const CuBFloatWrappedArray = Union{
    Adjoint{BFloat16,<:CuBFloatArray},
    Transpose{BFloat16,<:CuBFloatArray},
    SubArray{BFloat16,N,P} where {N,P<:CuBFloatArray},
}
function tangent_type(::Type{<:CuBFloatWrappedArray})
    return _throw_gpu_argument_error(
        "Mooncake: wrapped CUDA BFloat16 arrays are unsupported; use a plain CuArray."
    )
end
CUDAExt._cu_isa_gpu_side(::Type{<:CuBFloatArray}) = true

for sig in (
    Tuple{typeof(reshape),CuBFloatArray,NTuple{N,Int}} where {N},
    Tuple{typeof(derive),Type,CuBFloatArray,Dims,Int},
    Tuple{typeof(getindex),CuBFloatArray,AbstractVector{<:Union{Integer,CartesianIndex}}},
    Tuple{typeof(getindex),CuBFloatArray,_CuSliceIndex,_CuSliceIndex,Vararg{_CuSliceIndex}},
    Tuple{typeof(unsafe_copyto!),CuBFloatArray,Integer,Array,Integer,Integer},
    Tuple{typeof(unsafe_copyto!),CuBFloatArray,Integer,CuBFloatArray,Integer,Integer},
    Tuple{typeof(Core.kwcall),NamedTuple,typeof(sum),CuBFloatArray},
    Tuple{typeof(*),CuArray{BFloat16,2},CuArray{BFloat16,2}},
    Tuple{typeof(*),CuArray{BFloat16,2},CuArray{BFloat16,1}},
)
    @eval @is_primitive MinimalCtx $sig
end
@is_primitive DefaultCtx Tuple{typeof(sum),CuBFloatArray}

# BFloat16 atomic scatter is unavailable for the reverse vector-index rule.
function rrule!!(
    ::CoDual{typeof(getindex)},
    x::CoDual{<:CuBFloatArray,<:CuBFloatArray},
    idx::CoDual{<:AbstractVector{<:Union{Integer,CartesianIndex}}},
)
    return _throw_gpu_argument_error(
        "Mooncake: vector-index getindex does not support $(eltype(primal(x)))."
    )
end

const CuCatArray = Union{CuBFloatArray,CUDAExt.CuMaybeWrappedArray}
@is_primitive MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(cat),CuCatArray,Vararg{CuCatArray}
}
for f in (vcat, hcat)
    @eval @is_primitive MinimalCtx Tuple{typeof($f),CuCatArray,Vararg{CuCatArray}}
end
# NDual broadcast kernels do not support BFloat16.
@is_primitive MinimalCtx Tuple{
    typeof(Base.Broadcast.materialize!),CuBFloatArray,Broadcasted
}

for (D, rule) in ((Dual, :frule!!), (CoDual, :rrule!!))
    @eval begin
        function $rule(
            ::$D{typeof(Core.kwcall)},
            ::$D{<:NamedTuple},
            ::$D{typeof(cat)},
            args::$D{<:CuCatArray}...,
        )
            return _throw_gpu_argument_error(
                "Mooncake: CUDA concatenation does not support BFloat16."
            )
        end
        function $rule(
            ::$D{typeof(Base.Broadcast.materialize!)},
            dest::$D{<:CuBFloatArray},
            ::$D{<:Broadcasted},
        )
            return _throw_gpu_argument_error(
                "Mooncake: in-place CUDA broadcast does not support $(eltype(primal(dest))).",
            )
        end
    end
    for f in (vcat, hcat)
        @eval function $rule(::$D{typeof($f)}, args::$D{<:CuCatArray}...)
            return _throw_gpu_argument_error(
                "Mooncake: CUDA concatenation does not support BFloat16."
            )
        end
    end
end

end
