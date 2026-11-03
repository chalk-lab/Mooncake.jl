module MooncakeCUDABFloat16sExt

using Mooncake, CUDA, BFloat16s, LinearAlgebra, Random
using CUDA.CUDACore.GPUArrays: derive
using Base.Broadcast: Broadcasted

using Mooncake: MinimalCtx, DefaultCtx, @is_primitive, @foldable, primal, TestUtils
using Mooncake: Dual, CoDual, MaybeCache, IncCache, NoRData
import Mooncake: frule!!, rrule!!, tangent_type, tangent, arrayify
import Mooncake.TestUtils: __increment_should_allocate

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

Mooncake._copy_output(x::CuBFloatArray) = copy(x)
function Mooncake._copy_to_output!!(dst::P, src::P) where {P<:CuBFloatArray}
    copyto!(dst, src)
    return dst
end

@foldable tangent_type(::Type{P}) where {P<:CuBFloatArray} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuBFloatArray} = P

tangent(p::CuBFloatArray, ::NoRData) = p

function arrayify(x::A, dx::A) where {A<:CuBFloatArray}
    return (x, dx)
end

function Mooncake.zero_tangent_internal(x::CuBFloatArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = zero(x)
    dict[x] = t
    return t
end
function Mooncake.randn_tangent_internal(
    rng::AbstractRNG, x::CuBFloatArray, dict::MaybeCache
)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = copyto!(similar(x), randn(rng, eltype(x), size(x)...))
    dict[x] = t
    return t
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P<:CuBFloatArray}
    return isapprox(x, y; atol=(√eps(real(eltype(P)))), nans=true)
end
function Mooncake.increment_internal!!(c::IncCache, x::A, y::A) where {A<:CuBFloatArray}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
__increment_should_allocate(::Type{<:CuBFloatArray}) = true
Mooncake.set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CuBFloatArray) = fill!(x, 0)

function Mooncake._add_to_primal_internal(
    c::MaybeCache, x::P, y::P, unsafe::Bool
) where {P<:CuBFloatArray}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x .+ y
    c[key] = x′
    return x′
end
function Mooncake.primal_to_tangent_internal!!(t, x::CuBFloatArray, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= x
    return t
end
function Mooncake.tangent_to_primal_internal!!(x::CuBFloatArray, t, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= t
    return x
end
function Mooncake._dot_internal(c::MaybeCache, x::P, y::P) where {P<:CuBFloatArray}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    T = real(eltype(P))
    return Float64(dot(reinterpret(T, vec(x)), reinterpret(T, vec(y))))
end
function Mooncake._scale_internal(c::MaybeCache, x::Float64, y::P) where {P<:CuBFloatArray}
    haskey(c, y) && return c[y]::P
    t′ = real(eltype(P))(x) .* y
    c[y] = t′
    return t′
end

Mooncake.mooncake_tangent(::CuBFloatArray, t::CuBFloatArray) = t
Mooncake.to_cr_tangent(x::CuBFloatArray) = x
function Mooncake.increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuBFloatArray}
    f .+= t
    return NoRData()
end

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
            ::$D{typeof(Mooncake._new_)},
            ::$D{Type{P}},
            data::$D,
            maxsize::$D,
            offset::$D,
            dims::$D,
        ) where {P<:CuBFloatArray}
            y = Mooncake._new_(
                P, primal(data), primal(maxsize), primal(offset), primal(dims)
            )
            dy = Mooncake._new_(
                P, tangent(data), primal(maxsize), primal(offset), primal(dims)
            )
            out = $D(y, dy)
            return $D === Dual ? out : (out, CUDAExt._nopb(Val(6)))
        end
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
