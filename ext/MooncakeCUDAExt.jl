module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA: CuArray, CuRefValue, CuPtr, CuContext, CUmemPoolHandle_st
using CUDA: CUBLAS

import Mooncake:
    MinimalCtx,
    DefaultCtx,
    frule!!,
    rrule!!,
    @is_primitive,
    @unstable,
    @foldable,
    @from_rrule,
    @zero_derivative,
    tangent_type,
    fdata_type,
    rdata_type,
    primal,
    tangent,
    lgetfield,
    zero_fcodual,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    _dot_internal,
    _scale_internal,
    _new_,
    TestUtils,
    Dual,
    CoDual,
    NoTangent,
    NoPullback,
    NoFData,
    to_cr_tangent,
    increment_and_get_rdata!,
    MaybeCache,
    IncCache,
    NoRData,
    arrayify

import Mooncake.TestUtils:
    populate_address_map_internal, AddressMap, __increment_should_allocate

const CuFloatArray = CuArray{<:IEEEFloat}
const CuComplexArray = CuArray{<:Complex{<:IEEEFloat}}
const CuMaybeComplexArray = Union{CuFloatArray,CuComplexArray}

# Tell Mooncake.jl how to handle CuArrays.

@foldable tangent_type(::Type{P}) where {P<:CuMaybeComplexArray} = P
@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuMaybeComplexArray} = P
@unstable @foldable tangent_type(::Type{CuPtr{P}}) where {P} = CuPtr{tangent_type(P)}
@unstable @foldable tangent_type(::Type{CuRefValue{P}}) where {P} = CuRefValue{
    tangent_type(P)
}
tangent_type(::Type{CuContext}) = NoTangent
tangent_type(::Type{Ptr{CUmemPoolHandle_st}}) = NoTangent
tangent_type(::Type{CUBLAS.cublasOperation_t}) = NoTangent
tangent_type(::Type{CUBLAS.cublasComputeType_t}) = NoTangent

tangent(p::CuMaybeComplexArray, ::NoRData) = p

function arrayify(x::A, dx::A) where {A<:CuMaybeComplexArray}
    (x, dx)
end

function zero_tangent_internal(x::CuMaybeComplexArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = zero(x)
    dict[x] = t
    return t
end
function randn_tangent_internal(rng::AbstractRNG, x::CuMaybeComplexArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = CuArray(randn(rng, eltype(x), size(x)...))
    dict[x] = t
    return t
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:CuMaybeComplexArray}
    # allow nan comparisons to return true, real() to cover complex case
    return isapprox(x, y; atol=(√eps(real(eltype(P)))), nans=true)
end
function increment_internal!!(c::IncCache, x::A, y::A) where {A<:CuMaybeComplexArray}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
__increment_should_allocate(::Type{<:CuMaybeComplexArray}) = true
set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CuMaybeComplexArray) = x .= 0

function _add_to_primal_internal(
    c::MaybeCache, x::P, y::P, unsafe::Bool
) where {P<:CuMaybeComplexArray}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[(x, y, unsafe)] = x′
    return x′
end
function primal_to_tangent_internal!!(t, x::CuMaybeComplexArray, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= x
    return t
end
function tangent_to_primal_internal!!(x::CuMaybeComplexArray, t, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= t
    return x
end
function _dot_internal(c::MaybeCache, x::P, y::P) where {P<:CuMaybeComplexArray}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    return Float64(real(dot(x, y)))
end
function _scale_internal(c::MaybeCache, x::Float64, y::P) where {P<:CuMaybeComplexArray}
    haskey(c, y) && return c[y]::P
    t′ = eltype(P)(x) * y
    c[y] = t′
    return t′
end
function populate_address_map_internal(m::AddressMap, p::CuArray, t::CuArray)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    return m
end
function Mooncake.__verify_fdata_value(::IdDict{Any,Nothing}, p::CuArray, f::CuArray)
    if size(p) != size(f)
        throw(InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end
    return nothing
end

# @from_chainrules tools
# TODO: missing `mooncake_tangent` implementation.
to_cr_tangent(x::CuMaybeComplexArray) = x
function increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuMaybeComplexArray}
    f .+= t
    return NoRData()
end

# Basic rules for operating on CuArrays.

# Outer-constructor primitives are required here rather than `_new_` rules because the
# `CuArray(undef, dims)` constructor body performs GPU memory allocation (pool_alloc,
# DataRef construction, GPUArrays.cached_alloc) *before* the `new` call. A `_new_` rule
# alone would force Mooncake to trace through all that CUDA-internal machinery. Making
# the whole outer constructor primitive bypasses that and directly produces the correct
# (primal, tangent) pair — a fresh uninitialized primal and a zero-initialised tangent
# of the same shape.
@zero_derivative MinimalCtx Tuple{Type{<:CuArray},UndefInitializer,NTuple{N,Int}} where {N}

# The Vararg form CuArray(undef, d1, d2, ...) is a distinct dispatch target from the
# NTuple form above, so it needs its own primitive. Tangent is zero-initialised to avoid
# treating uninitialised GPU memory as a gradient accumulator.
@is_primitive(MinimalCtx, Tuple{Type{<:CuArray},UndefInitializer,Vararg{Int,N}} where {N},)
function frule!!(
    ::Dual{Type{P}}, ::Dual{UndefInitializer}, dims::Vararg{Dual{Int},N}
) where {P<:CuMaybeComplexArray,N}
    y = P(undef, map(primal, dims))
    return Dual(y, zero(y))
end
function rrule!!(
    p::CoDual{Type{P}}, init::CoDual{UndefInitializer}, dims::Vararg{CoDual{Int},N}
) where {P<:CuMaybeComplexArray,N}
    y = P(undef, map(primal, dims))
    return CoDual(y, zero(y)), NoPullback(p, init, dims...)
end

# Primitive rule for `reshape`: prevents tracing into CUDA.jl's reshape body which calls
# `copy(DataRef{...})` for reference-count management and hits llvmcall. reshape creates
# a view (shared GPU memory), so the tangent is simply a reshaped view of the input
# tangent — gradient accumulation propagates automatically and NoPullback is correct.
@is_primitive(MinimalCtx, Tuple{typeof(reshape),CuMaybeComplexArray,NTuple{N,Int}} where {N},)
function frule!!(
    ::Dual{typeof(reshape)}, x::Dual{<:CuMaybeComplexArray}, dims::Dual{<:NTuple}
)
    return Dual(reshape(primal(x), primal(dims)), reshape(tangent(x), primal(dims)))
end
function rrule!!(
    ::CoDual{typeof(reshape)}, x::CoDual{<:CuMaybeComplexArray}, dims::CoDual{<:NTuple}
)
    _dims = primal(dims)
    return CoDual(reshape(primal(x), _dims), reshape(x.dx, _dims)),
        NoPullback(ntuple(_ -> NoRData(), 3))
end

# `_new_` rules for the DataRef-based inner CuArray constructor, used by e.g. `reshape`
# and views. The tangent shares the DataRef (gradient memory) from the input tangent
# array, but with the new dims/offset — so gradient accumulation is automatic.
function frule!!(
    ::Dual{typeof(_new_)}, ::Dual{Type{P}},
    data::Dual, maxsize::Dual, offset::Dual, dims::Dual,
) where {P<:CuMaybeComplexArray}
    y = _new_(P, primal(data), primal(maxsize), primal(offset), primal(dims))
    dy = _new_(P, tangent(data), primal(maxsize), primal(offset), primal(dims))
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{P}},
    data::CoDual, maxsize::CoDual, offset::CoDual, dims::CoDual,
) where {P<:CuMaybeComplexArray}
    y = _new_(P, primal(data), primal(maxsize), primal(offset), primal(dims))
    dy = _new_(P, data.dx, primal(maxsize), primal(offset), primal(dims))
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 6))
end

# getfield / lgetfield rules for CuArray.
# CuArray has 4 fields: data (1, differentiable DataRef), maxsize (2), offset (3), dims (4).
function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:CuArray,<:CuArray},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    is_data = name === 1 || name === :data
    dy = is_data ? tangent(x).data : NoTangent()
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuArray,<:CuArray},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    is_data = name === 1 || name === :data
    dy = is_data ? x.dx.data : NoFData()
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuArray,<:CuArray}, ::Dual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    is_data = name === 1 || name === :data
    dy = is_data ? tangent(x).data : NoTangent()
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuArray,<:CuArray}, ::CoDual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    is_data = name === 1 || name === :data
    dy = is_data ? x.dx.data : NoFData()
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 3))
end

# Rule for `sum` is defined as a performance rule. 
# TODO: These rules can be merged with the `sum` rules in `rules/performance_patches`. 
# This would be done by defining `arrayify` for `CuFloatArray`.
@is_primitive(DefaultCtx, Tuple{typeof(sum),CuFloatArray})
function frule!!(::Dual{typeof(sum)}, x::Dual{<:CuFloatArray})
    return Dual(sum(primal(x)), sum(tangent(x)))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:CuFloatArray})
    dx = x.dx
    function sum_pb!!(dz)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(identity, x.x)), sum_pb!!
end

end
