module MooncakeCUDAExt

using LinearAlgebra, Random, Mooncake

using Base: IEEEFloat
using CUDA: CuArray

import Mooncake:
    MinimalCtx,
    rrule!!,
    @is_primitive,
    tangent_type,
    fdata_type,
    rdata_type,
    primal,
    tangent,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _add_to_primal_internal,
    _diff_internal,
    _dot_internal,
    _scale_internal,
    TestUtils,
    CoDual,
    NoPullback,
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

Mooncake.@foldable tangent_type(::Type{<:CuArray{P,N,M}}) where {P<:Union{Complex{<:IEEEFloat},IEEEFloat},N,M} = CuArray{
    tangent_type(P),N,M
}

Mooncake.@foldable fdata_type(::Type{CuArray{P,N,M}}) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M} = CuArray{
    P,N,M
}

Mooncake.@foldable rdata_type(
    ::Type{<:CuArray{P,N,M}}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M} = Mooncake.NoRData

function arrayify(x::A, dx::A) where {A<:Union{CuArray{<:BlasRealFloat}}}
    (x, dx)
end
function arrayify(x::CuArray{P}, dx::CuArray{<:Tangent}) where {P<:BlasComplexFloat}
    return x, reinterpret(P, dx)
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
    return isapprox(x, y)
end
function TestUtils.has_equal_data_internal(
    x::CuArray{P,N,M}, y::CuArray{P,N,M}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T<:IEEEFloat,P<:Mooncake.Tangent{@NamedTuple{re::T,im::T}},N,M}
    x_ = reinterpret(Complex{T}, x)
    y_ = reinterpret(Complex{T}, y)
    return isapprox(x_, y_)
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
function _diff_internal(c::MaybeCache, x::P, y::P) where {P<:CuFloatArray}
    key = (x, y)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = x - y
    c[key] = t
    return t
end
function _diff_internal(c::MaybeCache, x::P, y::P) where {P<:CuComplexArray}
    key = (x, y)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = tangent_type(P)(undef, size(x))
    t_ = reinterpret(eltype(x), t)
    @. t_ = x - y
    c[key] = t
    return t
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

end
