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

@zero_derivative MinimalCtx Tuple{Type{<:CuArray},UndefInitializer,NTuple{N,Int}} where {N}

# TODO: Mooncake defines rules for `_new_` instead of below. See
# https://chalk-lab.github.io/Mooncake.jl/stable/developer_documentation/custom_tangent_type/#Checklist:-Functions-Needed-for-Recursive-Struct-Support
#
@is_primitive(MinimalCtx, Tuple{Type{<:CuArray},UndefInitializer,Vararg{Int,N}} where {N},)
function frule!!(
    p::Dual{Type{P}}, init::Dual{UndefInitializer}, dims::Vararg{Dual{Int},N}
) where {P<:CuMaybeComplexArray,N}
    _dims = map(primal, dims)
    return Dual(P(undef, _dims), P(undef, _dims))
end
function rrule!!(
    p::CoDual{Type{P}}, init::CoDual{UndefInitializer}, dims::Vararg{CoDual{Int},N}
) where {P<:CuMaybeComplexArray,N}
    _dims = map(primal, dims)
    return CoDual(P(undef, _dims), P(undef, _dims)), NoPullback(p, init, dims...)
end

function frule!!(
    f::Dual{typeof(Mooncake._new_),Mooncake.NoTangent},
    p::Dual{A,Mooncake.NoTangent},
    x::Dual{M,TM},
) where {T,M<:CuArray{T},A<:Type{LinearAlgebra.Adjoint{T,M}},TM<:CuArray}
    y = _new_(LinearAlgebra.Adjoint{T,M}, Mooncake.primal(x))
    dy = Mooncake.Tangent((parent=Mooncake.tangent(x),))
    return Dual(y, dy)
end

function frule!!(
    f::Dual{typeof(Mooncake._new_),Mooncake.NoTangent},
    p::Dual{A,Mooncake.NoTangent},
    x::Dual{M,TM},
) where {T,M<:CuArray{T},A<:Type{LinearAlgebra.Transpose{T,M}},TM<:CuArray}
    y = _new_(LinearAlgebra.Transpose{T,M}, Mooncake.primal(x))
    dy = Mooncake.Tangent((parent=Mooncake.tangent(x),))
    return Dual(y, dy)
end

# getfield / lgetfield rules for CuArray.
function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:CuArray,<:CuArray},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_size = name === 2 || name === :dims
    dy = wants_size ? NoTangent() : tangent(x).data
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:CuArray,<:CuArray},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_size = name === 2 || name === :dims
    dy = wants_size ? NoFData() : x.dx
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuArray,<:CuArray}, ::Dual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    wants_size = name === 2 || name === :dims
    dy = wants_size ? NoTangent() : tangent(x).data
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{<:CuArray,<:CuArray}, ::CoDual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
    wants_size = name === 2 || name === :dims
    dy = wants_size ? NoFData() : x.dx
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
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
