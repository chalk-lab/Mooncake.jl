module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA: CuArray, CuRefValue, CuPtr, CuContext, CUmemPoolHandle_st
using CUDA: CUBLAS

import Mooncake:
    MinimalCtx,
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

# Tell Mooncake.jl how to handle CuArrays.

@foldable tangent_type(::Type{<:CuArray{P,N,M}}) where {P<:Union{Complex{<:IEEEFloat},IEEEFloat},N,M} = CuArray{
    tangent_type(P),N,M
}

@unstable @foldable tangent_type(::Type{CuPtr{P}}) where {P} = CuPtr{tangent_type(P)}
@unstable @foldable tangent_type(::Type{CuRefValue{P}}) where {P} = CuRefValue{
    tangent_type(P)
}
tangent_type(::Type{CuContext}) = NoTangent
tangent_type(::Type{Ptr{CUmemPoolHandle_st}}) = NoTangent
tangent_type(::Type{CUBLAS.cublasOperation_t}) = NoTangent

function arrayify(x::A, dx::A) where {A<:CuFloatArray}
    (x, dx)
end
function arrayify(x::CuComplexArray, dx::CuArray{<:Mooncake.Tangent})
    return x, reinterpret(eltype(x), dx)
end

function zero_tangent_internal(x::CuFloatArray, dict::MaybeCache)
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = zero(x)
    dict[x] = t
    return t
end
function zero_tangent_internal(x::CuArray{T}, dict::MaybeCache) where {T<:Complex}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = tangent_type(typeof(x))(undef, size(x))
    t_ = reinterpret(T, t)
    t_ .= zero(T)
    dict[x] = t
    return t
end
function randn_tangent_internal(
    rng::AbstractRNG, x::CuArray{T}, dict::MaybeCache
) where {T<:IEEEFloat}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = CuArray(randn(rng, T, size(x)...))
    dict[x] = t
    return t
end
function randn_tangent_internal(
    rng::AbstractRNG, x::CuArray{T}, dict::MaybeCache
) where {T<:Complex}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))
    t = tangent_type(typeof(x))(undef, size(x))
    t_ = reinterpret(T, t)
    th = randn(rng, T, size(x)...)
    t_ .= CuArray(th)
    dict[x] = t
    return t
end
function TestUtils.has_equal_data_internal(
    x::P, y::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:Union{CuFloatArray,CuComplexArray}}
    # allow nan comparisons to return true, real() to cover complex case
    return isapprox(x, y; atol=(√eps(real(eltype(P)))), nans=true)
end
function TestUtils.has_equal_data_internal(
    x::CuArray{P,N,M}, y::CuArray{P,N,M}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
    x_ = reinterpret(Complex{T}, x)
    y_ = reinterpret(Complex{T}, y)
    # allow nan comparisons to return true
    return isapprox(x_, y_; atol=(√eps(T)), nans=true)
end
function increment_internal!!(
    c::IncCache, x::CuArray{P,N,M}, y::CuArray{P,N,M}
) where {P<:IEEEFloat,N,M}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x .+= y
    return x
end
function increment_internal!!(
    c::IncCache, x::CuArray{P,N,M}, y::CuArray{P,N,M}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x_ = reinterpret(Complex{T}, x)
    y_ = reinterpret(Complex{T}, y)
    x_ .+= y_
    return x
end
__increment_should_allocate(::Type{<:CuFloatArray}) = true
set_to_zero_internal!!(::Mooncake.SetToZeroCache, x::CuFloatArray) = x .= 0
function set_to_zero_internal!!(
    ::Mooncake.SetToZeroCache, x::CuArray{Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
) where {T<:IEEEFloat,N,M}
    x_ = reinterpret(Complex{T}, x)
    x_ .= zero(Complex{T})
    return x
end

function _add_to_primal_internal(
    c::MaybeCache, x::P, y::P, unsafe::Bool
) where {P<:CuFloatArray}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + y
    c[(x, y, unsafe)] = x′
    return x′
end
function _add_to_primal_internal(
    c::MaybeCache, x::P, y::TP, unsafe::Bool
) where {P<:CuComplexArray,TP}
    key = (x, y, unsafe)
    haskey(c, key) && return c[key]::P
    x′ = x + reinterpret(eltype(x), y)
    c[(x, y, unsafe)] = x′
    return x′
end
function primal_to_tangent_internal!!(t, x::CuFloatArray, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= x
    return t
end
function primal_to_tangent_internal!!(t, x::CuComplexArray, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(t)
    c[x] = t
    t .= reinterpret(eltype(t), x)
    return t
end
function tangent_to_primal_internal!!(x::CuFloatArray, t, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= t
    return x
end
function tangent_to_primal_internal!!(x::CuComplexArray, t, c::MaybeCache)
    haskey(c, x) && return c[x]::typeof(x)
    c[x] = x
    x .= reinterpret(eltype(x), t)
    return x
end
function _dot_internal(c::MaybeCache, x::P, y::P) where {P<:CuFloatArray}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    return Float64(dot(x, y))
end
function _dot_internal(
    c::MaybeCache, x::CuArray{P}, y::CuArray{P}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}}}
    key = (x, y)
    haskey(c, key) && return c[key]::Float64
    x_ = reinterpret(Complex{T}, x)
    y_ = reinterpret(Complex{T}, y)
    return Float64(real(dot(x_, y_)))
end
function _scale_internal(c::MaybeCache, x::Float64, y::P) where {T<:IEEEFloat,P<:CuArray{T}}
    haskey(c, y) && return c[y]::P
    t′ = T(x) * y
    c[y] = t′
    return t′
end
function _scale_internal(
    c::MaybeCache, x::Float64, y::CuArray{P,N,M}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
    haskey(c, y) && return c[y]::CuArray{P,N,M}
    t′ = copy(y)
    t′_ = reinterpret(Complex{T}, t′)
    t′_ .*= T(x)
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
Mooncake.@foldable tangent_type(::Type{P}, ::Type{NoRData}) where {P<:CuFloatArray} = P
Mooncake.@foldable tangent_type(::Type{CuArray{P,N,M}}, ::Type{NoRData}) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M} = CuArray{
    P,N,M
}
tangent(p::CuFloatArray, ::NoRData) = p
function tangent(
    p::CuArray{P,N,M}, ::NoRData
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
    p
end

to_cr_tangent(x::CuFloatArray) = x
function increment_and_get_rdata!(f::T, ::NoRData, t::T) where {T<:CuFloatArray}
    f .+= t
    return NoRData()
end

# Basic rules for operating on CuArrays.

@is_primitive(MinimalCtx, Tuple{Type{<:CuArray},UndefInitializer,Vararg{Int,N}} where {N},)
function rrule!!(
    p::CoDual{Type{P}}, init::CoDual{UndefInitializer}, dims::CoDual{Int}...
) where {P<:CuFloatArray}
    _dims = map(primal, dims)
    return CoDual(P(undef, _dims), P(undef, _dims)), NoPullback(p, init, dims...)
end
function rrule!!(
    p::CoDual{Type{P}}, init::CoDual{UndefInitializer}, dims::CoDual{Int}...
) where {P<:CuComplexArray}
    _dims = map(primal, dims)
    return (
        CoDual(P(undef, _dims), tangent_type(P)(undef, _dims)), NoPullback(p, init, dims...)
    )
end

@zero_derivative MinimalCtx Tuple{Type{<:CuArray},UndefInitializer,NTuple{N,Int}} where {N}

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
function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{<:CuArray,<:CuArray}, ::Dual{Val{name}}
) where {name}
    y = getfield(primal(x), name)
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
@is_primitive(MinimalCtx, Tuple{typeof(sum),CuFloatArray})
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
