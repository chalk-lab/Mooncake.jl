# This file was introduced as part of the transition from 1.10 to 1.11. Its purpose is to
# ensure that Mooncake can handle the new implementation of `Array`s. This implementation
# relies on the new `Memory` and `MemoryRef` types (aliases for specific parametrisations of
# `GenericMemory` and `GenericMemoryRef`). Consequently, the code here will make little
# sense unless you are familiar with these types, and how they relate to `Array`s.
# Fortunately, Oscar Smith and Jameson Nash gave an excellent talk at JuliaCon 2024 on
# exactly this topic, which you can find here: https://www.youtube.com/watch?v=L6BFQ1d8xNs .

#
# Memory
#

# Tangent Interface Implementation

const Maybe{T} = Union{Nothing,T}

@foldable tangent_type(::Type{<:Memory{P}}) where {P} = Memory{tangent_type(P)}

function zero_tangent_internal(x::Memory{P}, dict::MaybeCache) where {P}
    T = tangent_type(typeof(x))

    # If no dict is provided, then the caller promises that there is no need for it.
    if dict === nothing
        t = T(undef, length(x))
        return _map_if_assigned!(Base.Fix2(zero_tangent_internal, dict), t, x)::T
    end

    # If we've seen this primal before, then we have a circular reference, and must return
    # the tangent which has already been allocated for it.
    haskey(dict, x) && return dict[x]::T

    # We have not seen this primal before, so allocate + store the tangent for it, and zero
    # out the elements.
    t = T(undef, length(x))
    dict[x] = t
    return _map_if_assigned!(Base.Fix2(zero_tangent_internal, dict), t, x)::T
end

function randn_tangent_internal(rng::AbstractRNG, x::Memory, dict::MaybeCache)
    T = tangent_type(typeof(x))
    haskey(dict, x) && return dict[x]::T

    t = T(undef, length(x))
    dict[x] = t
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, dict), t, x)::T
end

function TestUtils.has_equal_data_internal(
    x::Memory{P}, y::Memory{P}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P}
    length(x) == length(y) || return false
    id_pair = (objectid(x), objectid(y))
    haskey(d, id_pair) && return d[id_pair]

    d[id_pair] = true
    equality = map(1:length(x)) do n
        if isassigned(x, n) != isassigned(y, n)
            return !equal_undefs
        elseif !isassigned(x, n)
            return true
        else
            return TestUtils.has_equal_data_internal(x[n], y[n], equal_undefs, d)
        end
    end
    return all(equality)
end

function increment_internal!!(c::IncCache, x::Memory{P}, y::Memory{P}) where {P}
    (haskey(c, x) || x === y) && return x
    c[x] = true
    return _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Memory)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _add_to_primal_internal(
    c::MaybeCache, p::Memory{P}, t::Memory, unsafe::Bool
) where {P}
    k = (p, t, unsafe)
    haskey(c, k) && return c[k]::Memory{P}
    p′ = Memory{P}(undef, length(p))
    c[k] = p′
    return _map_if_assigned!((p, t) -> _add_to_primal_internal(c, p, t, unsafe), p′, p, t)
end

function _diff_internal(c::MaybeCache, p::Memory{P}, q::Memory{P}) where {P}
    key = (p, q)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = Memory{tangent_type(P)}(undef, length(p))
    c[key] = t
    return _map_if_assigned!((p, q) -> _diff_internal(c, p, q), t, p, q)
end

function _scale_internal(c::MaybeCache, a::Float64, t::Memory{T}) where {T}
    haskey(c, t) && return c[t]::Memory{T}
    t′ = Memory{T}(undef, length(t))
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

import .TestUtils: populate_address_map_internal
function populate_address_map_internal(m::TestUtils.AddressMap, p::Memory, t::Memory)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    if haskey(m, k)
        @assert m[k] == v
        return m
    end
    m[k] = v
    foreach(
        n -> isassigned(p, n) && populate_address_map_internal(m, p[n], t[n]), eachindex(p)
    )
    return m
end

# FData / RData Interface Implementation

@foldable tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Memory} = F

tangent(f::Memory, ::NoRData) = f

function __verify_fdata_value(::IdDict{Any,Nothing}, p::Memory{P}, f::Memory{F}) where {P,F}
    if length(p) != length(f)
        msg =
            "length(p) == $(length(p)) but length(f) == $(length(f)). " *
            "p isa Memory{$P} and f isa Memory{$F}"
        throw(InvalidFDataException(msg))
    end
    return nothing
end

#
# Array -- tangent interface implementation
#

@inline function zero_tangent_internal(x::Array, dict::MaybeCache)
    T = tangent_type(typeof(x))

    # If we already have a tangent for this, just return that.
    haskey(dict, x) && return dict[x]::T

    # Construct a new tangent, log it in the `dict`, and return it.
    dx = _new_(T)
    Base.setfield!(dx, :size, x.size)
    dict[x] = dx
    Base.setfield!(dx, :ref, zero_tangent_internal(x.ref, dict))
    return dx::T
end

function randn_tangent_internal(rng::AbstractRNG, x::Array, dict::MaybeCache)
    T = tangent_type(typeof(x))

    # If we already have a tangent for this, just return that.
    haskey(dict, x) && return dict[x]::T

    # Construct a new tangent, log it in the `dict`, and return it.
    dx = _new_(T)
    Base.setfield!(dx, :size, x.size)
    dict[x] = dx
    Base.setfield!(dx, :ref, randn_tangent_internal(rng, x.ref, dict))
    return dx::T
end

function increment_internal!!(c::IncCache, x::T, y::T) where {T<:Array}
    (haskey(c, x) || x === y) && return x
    c[x] = true
    _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
    return x
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Array)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:Array}
    haskey(c, t) && return c[t]::T
    t′ = T(undef, size(t)...)
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

for A in (Array, Memory)
    @eval function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:$A}
        key = (t, s)
        haskey(c, key) && return c[key]::Float64
        c[key] = 0.0
        bitstype = Val(isbitstype(eltype(T)))
        return sum(eachindex(t, s); init=0.0) do i
            if bitstype isa Val{true} || (isassigned(t, i) && isassigned(s, i))
                _dot_internal(c, t[i], s[i])::Float64
            else
                0.0
            end
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, x::Array{P,N}, t::Array{<:Any,N}, unsafe::Bool
) where {P,N}
    key = (x, t, unsafe)
    haskey(c, key) && return c[key]::Array{P,N}
    x′ = Array{P,N}(undef, size(x)...)
    c[key] = x′
    return _map_if_assigned!((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x′, x, t)
end

function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:Array}
    key = (p, q)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = tangent_type(P)(undef, size(p))
    c[key] = t
    return _map_if_assigned!((p, q) -> _diff_internal(c, p, q), t, p, q)
end

# Rules

@is_primitive(
    MinimalCtx, Tuple{typeof(unsafe_copyto!),MemoryRef{P},MemoryRef{P},Int} where {P}
)
function frule!!(
    ::Dual{typeof(unsafe_copyto!)},
    dest::Dual{MemoryRef{P}},
    src::Dual{MemoryRef{P}},
    n::Dual{Int},
) where {P}
    unsafe_copyto!(primal(dest), primal(src), primal(n))
    unsafe_copyto!(tangent(dest), tangent(src), primal(n))
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{MemoryRef{P}},
    src::CoDual{MemoryRef{P}},
    _n::CoDual{Int},
) where {P}
    n = primal(_n)

    # Copy state of primal and fdata of dest.
    dest_primal_copy = memoryref(Memory{P}(undef, n))
    dest_fdata_copy = memoryref(Memory{tangent_type(P)}(undef, n))
    unsafe_copyto!(dest_primal_copy, dest.x, n)
    unsafe_copyto!(dest_fdata_copy, dest.dx, n)

    # Apply primal computation to both primal and fdata.
    unsafe_copyto!(dest.x, src.x, n)
    unsafe_copyto!(dest.dx, src.dx, n)

    function unsafe_copyto!_adjoint(::NoRData)

        # Increment tangents in src by values in dest.
        tmp = Memory{eltype(dest.dx)}(undef, n)
        unsafe_copyto!(memoryref(tmp), dest.dx, n)

        # Restore state of `dest`.
        unsafe_copyto!(dest.x, dest_primal_copy, n)
        unsafe_copyto!(dest.dx, dest_fdata_copy, n)

        # Increment gradients.
        @inbounds for i in 1:n
            src_ref = memoryref(src.dx, i)
            if isassigned(src_ref)
                src_ref[] = increment!!(src_ref[], memoryref(tmp, i)[])
            end
        end

        return ntuple(_ -> NoRData(), 4)
    end
    return dest, unsafe_copyto!_adjoint
end

#
# MemoryRef
#

# Tangent Interface Implementation

@foldable tangent_type(::Type{<:MemoryRef{P}}) where {P} = MemoryRef{tangent_type(P)}

#=
Given a new chunk of memory `m`, construct a `MemoryRef` which points to the same relative
position in `x`, as `m` points to in its underlying `Memory` object. For example, in the
following:
```julia
original_mem = Memory{Float64}(undef, 10)
x = memoryref(original_mem, 4)
new_mem = Memory{Float64}(undef, 10)
new_x = construct_ref(x, new_mem)
```
`new_x` will point towards the 4th element of `new_mem`. Care is required of the length
of `original_mem` is `0`. See implementation for details.
=#
function construct_ref(x::MemoryRef, m::Memory)
    return isempty(m) ? memoryref(m) : memoryref(m, Core.memoryrefoffset(x))
end

function zero_tangent_internal(x::MemoryRef, dict::MaybeCache)
    return construct_ref(x, zero_tangent_internal(x.mem, dict))
end

function randn_tangent_internal(rng::AbstractRNG, x::MemoryRef, dict::MaybeCache)
    return construct_ref(x, randn_tangent_internal(rng, x.mem, dict))
end

function TestUtils.has_equal_data_internal(
    x::MemoryRef{P}, y::MemoryRef{P}, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P}
    equal_refs = Core.memoryrefoffset(x) == Core.memoryrefoffset(y)
    equal_data = TestUtils.has_equal_data_internal(x.mem, y.mem, equal_undefs, d)
    return equal_refs && equal_data
end

function increment_internal!!(c::IncCache, x::P, y::P) where {P<:MemoryRef}
    return construct_ref(x, increment_internal!!(c, x.mem, y.mem))
end

function set_to_zero_internal!!(c::SetToZeroCache, x::MemoryRef)
    set_to_zero_internal!!(c, x.mem)
    return x
end

function _add_to_primal_internal(c::MaybeCache, p::MemoryRef, t::MemoryRef, unsafe::Bool)
    return construct_ref(p, _add_to_primal_internal(c, p.mem, t.mem, unsafe))
end

function _diff_internal(c::MaybeCache, p::P, q::P) where {P<:MemoryRef}
    @assert Core.memoryrefoffset(p) == Core.memoryrefoffset(q)
    return construct_ref(p, _diff_internal(c, p.mem, q.mem))
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MemoryRef}
    @assert Core.memoryrefoffset(t) == Core.memoryrefoffset(s)
    return _dot_internal(c, t.mem, s.mem)::Float64
end

function _scale_internal(c::MaybeCache, a::Float64, t::MemoryRef)
    return construct_ref(t, _scale_internal(c, a, t.mem))
end

function populate_address_map_internal(m::TestUtils.AddressMap, p::MemoryRef, t::MemoryRef)
    return populate_address_map_internal(m, p.mem, t.mem)
end

# FData / RData Interface Implementation

fdata_type(::Type{<:MemoryRef{T}}) where {T} = MemoryRef{T}

rdata_type(::Type{<:MemoryRef}) = NoRData

@foldable tangent_type(::Type{<:MemoryRef{T}}, ::Type{NoRData}) where {T} = MemoryRef{T}

tangent(f::MemoryRef, ::NoRData) = f

function __verify_fdata_value(
    c::IdDict{Any,Nothing}, p::MemoryRef{P}, f::MemoryRef{T}
) where {P,T}
    return _verify_fdata_value(c, p.mem, f.mem)
end

#
# Rules for `Memory` and `MemoryRef`s
#

_val(::Val{c}) where {c} = c

using Core: memoryref_isassigned, memoryrefget, memoryrefset!, memoryrefnew, memoryrefoffset

@zero_derivative(
    MinimalCtx, Tuple{typeof(memoryref_isassigned),GenericMemoryRef,Symbol,Bool}
)

@inline function lmemoryrefget(
    x::MemoryRef, ::Val{ordering}, ::Val{boundscheck}
) where {ordering,boundscheck}
    return memoryrefget(x, ordering, boundscheck)
end

@is_primitive MinimalCtx Tuple{typeof(lmemoryrefget),MemoryRef,Val,Val}
@inline function frule!!(
    ::Dual{typeof(lmemoryrefget)},
    x::Dual{<:MemoryRef},
    _ordering::Dual{<:Val},
    _boundscheck::Dual{<:Val},
)
    ordering = primal(_ordering)
    bc = primal(_boundscheck)
    y = memoryrefget(primal(x), _val(ordering), _val(bc))
    dy = memoryrefget(tangent(x), _val(ordering), _val(bc))
    return Dual(y, dy)
end
@inline function rrule!!(
    ::CoDual{typeof(lmemoryrefget)},
    x::CoDual{<:MemoryRef},
    _ordering::CoDual{<:Val},
    _boundscheck::CoDual{<:Val},
)
    ordering = primal(_ordering)
    bc = primal(_boundscheck)
    dx = x.dx
    function lmemoryrefget_adjoint(dy)
        new_tangent = increment_rdata!!(memoryrefget(dx, _val(ordering), _val(bc)), dy)
        memoryrefset!(dx, new_tangent, _val(ordering), _val(bc))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    y = memoryrefget(x.x, _val(ordering), _val(bc))
    dy = fdata(memoryrefget(x.dx, _val(ordering), _val(bc)))
    return CoDual(y, dy), lmemoryrefget_adjoint
end

@inline Base.@propagate_inbounds function frule!!(
    ::Dual{typeof(memoryrefget)},
    x::Dual{<:MemoryRef},
    _ordering::Dual{Symbol},
    _boundscheck::Dual{Bool},
)
    ordering = primal(_ordering)
    boundscheck = primal(_boundscheck)
    y = memoryrefget(primal(x), ordering, boundscheck)
    dy = memoryrefget(tangent(x), ordering, boundscheck)
    return Dual(y, dy)
end
@inline Base.@propagate_inbounds function rrule!!(
    ::CoDual{typeof(memoryrefget)},
    x::CoDual{<:MemoryRef},
    _ordering::CoDual{Symbol},
    _boundscheck::CoDual{Bool},
)
    out, adj = rrule!!(
        zero_fcodual(lmemoryrefget),
        x,
        zero_fcodual(Val(primal(_ordering))),
        zero_fcodual(Val(primal(_boundscheck))),
    )
    memoryrefget_adjoint(dy) = adj(dy)
    return out, memoryrefget_adjoint
end

# Core.memoryrefmodify!

@inline function frule!!(::Dual{typeof(memoryrefnew)}, x::Dual{<:Memory})
    return Dual(memoryrefnew(primal(x)), memoryrefnew(tangent(x)))
end
@inline function rrule!!(f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:Memory})
    return CoDual(memoryrefnew(x.x), memoryrefnew(x.dx)), NoPullback(f, x)
end

@inline function frule!!(::Dual{typeof(memoryrefnew)}, x::Dual{<:MemoryRef}, ii::Dual{Int})
    return Dual(memoryrefnew(primal(x), primal(ii)), memoryrefnew(tangent(x), primal(ii)))
end
@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:MemoryRef}, ii::CoDual{Int}
)
    return CoDual(memoryrefnew(x.x, ii.x), memoryrefnew(x.dx, ii.x)), NoPullback(f, x, ii)
end

@inline function frule!!(
    ::Dual{typeof(memoryrefnew)},
    x::Dual{<:MemoryRef},
    ii::Dual{Int},
    boundscheck::Dual{Bool},
)
    y = memoryrefnew(primal(x), primal(ii), primal(boundscheck))
    dy = memoryrefnew(tangent(x), primal(ii), primal(boundscheck))
    return Dual(y, dy)
end
@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)},
    x::CoDual{<:MemoryRef},
    ii::CoDual{Int},
    boundscheck::CoDual{Bool},
)
    y = memoryrefnew(x.x, ii.x, boundscheck.x)
    dy = memoryrefnew(x.dx, ii.x, boundscheck.x)
    return CoDual(y, dy), NoPullback(f, x, ii, boundscheck)
end

@zero_derivative MinimalCtx Tuple{typeof(memoryrefoffset),GenericMemoryRef}

# Core.memoryrefreplace!

@inline function lmemoryrefset!(
    x::MemoryRef, value, ::Val{ordering}, ::Val{boundscheck}
) where {ordering,boundscheck}
    return memoryrefset!(x, value, ordering, boundscheck)
end

@is_primitive MinimalCtx Tuple{typeof(lmemoryrefset!),MemoryRef,Any,Val,Val}

@inline function frule!!(
    ::Dual{typeof(lmemoryrefset!)},
    x::Dual{<:MemoryRef{P},<:MemoryRef{V}},
    value::Dual,
    ::Dual{Val{ordering}},
    ::Dual{Val{boundscheck}},
) where {P,V,ordering,boundscheck}
    memoryrefset!(primal(x), primal(value), ordering, boundscheck)
    memoryrefset!(tangent(x), tangent(value), ordering, boundscheck)
    return value
end
@inline function rrule!!(
    ::CoDual{typeof(lmemoryrefset!)},
    x::CoDual{<:MemoryRef{P},<:MemoryRef{V}},
    value::CoDual,
    _ordering::CoDual{<:Val},
    _boundscheck::CoDual{<:Val},
) where {P,V}
    ordering = primal(_ordering)
    bc = primal(_boundscheck)

    isbitstype(P) && return isbits_lmemoryrefset!_rule(x, value, ordering, bc)

    to_save = isassigned(x.x)
    old_x = Ref{Tuple{P,V}}()
    if to_save
        old_x[] = (
            memoryrefget(x.x, _val(ordering), _val(bc)),
            memoryrefget(x.dx, _val(ordering), _val(bc)),
        )
    end

    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    dx = x.dx
    memoryrefset!(dx, tangent(value.dx, zero_rdata(value.x)), _val(ordering), _val(bc))
    function lmemoryrefset_adjoint(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(dx, _val(ordering), _val(bc))))
        if to_save
            memoryrefset!(x.x, old_x[][1], _val(ordering), _val(bc))
            memoryrefset!(dx, old_x[][2], _val(ordering), _val(bc))
        end
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, lmemoryrefset_adjoint
end

function isbits_lmemoryrefset!_rule(x::CoDual, value::CoDual, ordering::Val, bc::Val)
    old_x = (
        memoryrefget(x.x, _val(ordering), _val(bc)),
        memoryrefget(x.dx, _val(ordering), _val(bc)),
    )
    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    memoryrefset!(x.dx, zero_tangent(value.x), _val(ordering), _val(bc))

    function isbits_lmemoryrefset!_adjoint(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(x.dx, _val(ordering), _val(bc))))
        memoryrefset!(x.x, old_x[1], _val(ordering), _val(bc))
        memoryrefset!(x.dx, old_x[2], _val(ordering), _val(bc))
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, isbits_lmemoryrefset!_adjoint
end

@inline function frule!!(
    ::Dual{typeof(memoryrefset!)},
    x::Dual{<:MemoryRef{P},<:MemoryRef{V}},
    value::Dual,
    ordering::Dual{Symbol},
    boundscheck::Dual{Bool},
) where {P,V}
    return frule!!(
        zero_dual(lmemoryrefset!),
        x,
        value,
        zero_dual(Val(primal(ordering))),
        zero_dual(Val(primal(boundscheck))),
    )
end
@inline function rrule!!(
    ::CoDual{typeof(memoryrefset!)},
    x::CoDual{<:MemoryRef{P},<:MemoryRef{V}},
    value::CoDual,
    ordering::CoDual{Symbol},
    boundscheck::CoDual{Bool},
) where {P,V}
    y, adj = rrule!!(
        zero_fcodual(lmemoryrefset!),
        x,
        value,
        zero_fcodual(Val(primal(ordering))),
        zero_fcodual(Val(primal(boundscheck))),
    )
    memoryrefset_adjoint(dy) = adj(dy)
    return y, memoryrefset_adjoint
end

# Core.memoryrefsetonce!
# Core.memoryrefswap!
# Core.set_binding_type!

# _new_ and _new_-adjacent rules for Memory, MemoryRef, and Array.

@is_primitive MinimalCtx Tuple{Type{<:Memory},UndefInitializer,Int}
function frule!!(::Dual{Type{Memory{P}}}, ::Dual{UndefInitializer}, n::Dual{Int}) where {P}
    x = Memory{P}(undef, primal(n))
    return Dual(x, zero_tangent_internal(x, NoCache()))
end
function rrule!!(
    ::CoDual{Type{Memory{P}}}, ::CoDual{UndefInitializer}, n::CoDual{Int}
) where {P}
    x = Memory{P}(undef, primal(n))
    dx = zero_tangent_internal(x, NoCache())
    return CoDual(x, dx), NoPullback((NoRData(), NoRData(), NoRData()))
end

function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{MemoryRef{P}}},
    ptr_or_offset::CoDual{Ptr{Nothing}},
    mem::CoDual{Memory{P}},
) where {P}
    y = _new_(MemoryRef{P}, ptr_or_offset.x, mem.x)
    dy = _new_(MemoryRef{tangent_type(P)}, bitcast(Ptr{Nothing}, ptr_or_offset.dx), mem.dx)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(_new_)},
    ::Dual{Type{Array{P,N}}},
    ref::Dual{MemoryRef{P}},
    size::Dual{<:NTuple{N,Int}},
) where {P,N}
    y = _new_(Array{P,N}, primal(ref), primal(size))
    dy = _new_(Array{tangent_type(P),N}, tangent(ref), primal(size))
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{Array{P,N}}},
    ref::CoDual{MemoryRef{P}},
    size::CoDual{<:NTuple{N,Int}},
) where {P,N}
    y = _new_(Array{P,N}, ref.x, size.x)
    dy = _new_(Array{tangent_type(P),N}, ref.dx, size.x)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(_foreigncall_)},
    ::Dual{Val{:jl_genericmemory_copy}},
    ::Dual,
    ::Dual{Tuple{Val{Any}}},
    ::Dual{Val{0}},
    ::Dual{Val{:ccall}},
    x::Dual{<:Memory},
)
    return Dual(primal(copy(x)), tangent(copy(x)))
end
function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
    ::CoDual{Val{:jl_genericmemory_copy}},
    ::CoDual,
    ::CoDual{Tuple{Val{Any}}},
    ::CoDual{Val{0}},
    ::CoDual{Val{:ccall}},
    x::CoDual{<:Memory},
)
    dx = x.dx
    dx_copy = copy(dx)
    y = CoDual(copy(x.x), dx_copy)
    function jl_genericmemory_copy_pullback(::NoRData)
        _map_if_assigned!(increment!!, dx, dx, dx_copy)
        return tuple_fill(NoRData(), Val(7))
    end
    return y, jl_genericmemory_copy_pullback
end

# getfield / lgetfield rules for Memory, MemoryRef, and Array.

function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:Memory,<:Memory},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_length = name === 1 || name === :length
    dy = wants_length ? NoTangent() : bitcast(Ptr{NoTangent}, tangent(x).ptr)
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:Memory,<:Memory},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_length = name === 1 || name === :length
    dy = wants_length ? NoFData() : bitcast(Ptr{NoTangent}, x.dx.ptr)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:MemoryRef,<:MemoryRef},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_offset = name === 1 || name === :ptr_or_offset
    dy = wants_offset ? bitcast(Ptr{NoTangent}, tangent(x).ptr_or_offset) : tangent(x).mem
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:MemoryRef,<:MemoryRef},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_offset = name === 1 || name === :ptr_or_offset
    dy = wants_offset ? bitcast(Ptr{NoTangent}, x.dx.ptr_or_offset) : x.dx.mem
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{<:Array,<:Array},
    ::Dual{Val{name}},
    ::Dual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_size = name === 2 || name === :size
    dy = wants_size ? NoTangent() : tangent(x).ref
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:Array,<:Array},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_size = name === 2 || name === :size
    dy = wants_size ? NoFData() : x.dx.ref
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

const _MemTypes = Union{Memory,MemoryRef,Array}

function frule!!(
    f::Dual{typeof(lgetfield)}, x::Dual{<:_MemTypes,<:_MemTypes}, name::Dual{<:Val}
)
    return frule!!(f, x, name, zero_dual(Val(:not_atomic)))
end
function rrule!!(
    f::CoDual{typeof(lgetfield)}, x::CoDual{<:_MemTypes,<:_MemTypes}, name::CoDual{<:Val}
)
    y, adj = rrule!!(f, x, name, zero_fcodual(Val(:not_atomic)))
    ternary_lgetfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_lgetfield_adjoint
end

function frule!!(
    ::Dual{typeof(getfield)},
    x::Dual{<:_MemTypes,<:_MemTypes},
    name::Dual{<:Union{Int,Symbol}},
    order::Dual{Symbol},
)
    return frule!!(
        zero_dual(lgetfield), x, zero_dual(Val(primal(name))), zero_dual(Val(primal(order)))
    )
end
function rrule!!(
    ::CoDual{typeof(getfield)},
    x::CoDual{<:_MemTypes,<:_MemTypes},
    name::CoDual{<:Union{Int,Symbol}},
    order::CoDual{Symbol},
)
    y, adj = rrule!!(
        zero_fcodual(lgetfield),
        x,
        zero_fcodual(Val(primal(name))),
        zero_fcodual(Val(primal(order))),
    )
    getfield_adjoint(dy) = adj(dy)
    return y, getfield_adjoint
end

function frule!!(
    ::Dual{typeof(getfield)},
    x::Dual{<:_MemTypes,<:_MemTypes},
    name::Dual{<:Union{Int,Symbol}},
)
    return frule!!(zero_dual(lgetfield), x, zero_dual(Val(primal(name))))
end
function rrule!!(
    f::CoDual{typeof(getfield)},
    x::CoDual{<:_MemTypes,<:_MemTypes},
    name::CoDual{<:Union{Int,Symbol}},
)
    y, adj = rrule!!(f, x, name, zero_fcodual(:not_atomic))
    ternary_getfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_getfield_adjoint
end

@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::Dual{<:Array,<:Array}, ::Dual{Val{name}}, x::Dual
) where {name}
    setfield!(primal(value), name, primal(x))
    setfield!(tangent(value), name, (name === :size || name === 2) ? primal(x) : tangent(x))
    return x
end
@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)},
    value::CoDual{<:Array,<:Array},
    ::CoDual{Val{name}},
    x::CoDual,
) where {name}
    old_x = getfield(value.x, name)
    old_dx = getfield(value.dx, name)
    setfield!(value.x, name, x.x)
    setfield!(value.dx, name, (name === :size || name === 2) ? x.x : x.dx)
    function array_lsetfield!_adjoint(::NoRData)
        setfield!(value.x, name, old_x)
        setfield!(value.dx, name, old_dx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, array_lsetfield!_adjoint
end

# Misc. other rules which are required for correctness.

@is_primitive MinimalCtx Tuple{typeof(copy),Array}
frule!!(::Dual{typeof(copy)}, a::Dual{<:Array}) = Dual(copy(primal(a)), copy(tangent(a)))
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    dx = tangent(a)
    dy = copy(dx)
    y = CoDual(copy(primal(a)), dy)
    function copy_pullback!!(::NoRData)
        increment!!(dx, dy)
        return NoRData(), NoRData()
    end
    return y, copy_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(fill!),Array{<:Union{UInt8,Int8}},Integer}
@is_primitive MinimalCtx Tuple{typeof(fill!),Memory{<:Union{UInt8,Int8}},Integer}
function frule!!(
    ::Dual{typeof(fill!)}, a::Dual{T}, x::Dual{<:Integer}
) where {V<:Union{UInt8,Int8},T<:Union{Array{V},Memory{V}}}
    return Dual(fill!(primal(a), primal(x)), tangent(a))
end
function rrule!!(
    ::CoDual{typeof(fill!)}, a::CoDual{T}, x::CoDual{<:Integer}
) where {V<:Union{UInt8,Int8},T<:Union{Array{V},Memory{V}}}
    pa = primal(a)
    old_value = copy(pa)
    fill!(pa, primal(x))
    function fill!_pullback!!(::NoRData)
        pa .= old_value
        return NoRData(), NoRData(), NoRData()
    end
    return a, fill!_pullback!!
end

# Test cases

function _mems()

    # Set up memory with an undefined element.
    mem_with_single_undef = Memory{Memory{Int}}(undef, 2)
    mem_with_single_undef[1] = fill!(Memory{Int}(undef, 4), 2)

    # Return a collection of test cases.
    mems = [
        (fill!(Memory{Float64}(undef, 10), 0.0)),
        (fill!(Memory{Int}(undef, 5), 1)),
        (Memory{Vector{Float64}}([randn(1), randn(3)])),
        (Memory{Vector{Float64}}(undef, 3)),
        (Memory{Any}(randn(3))),
        (mem_with_single_undef),
        (Memory{Any}(undef, 0)),
    ]
    sample_values = [1.0, 3, randn(2), randn(2), 5.0, Memory{Int}(undef, 5), nothing]
    return mems, sample_values
end

function _mem_refs()

    # Generate test cases of arbitrary length.
    mems_1, sample_values_1 = _mems()

    # Restrict to minimum length of 2.
    _mems_2, _sample_values_2 = _mems()
    inds = findall(x -> length(x) >= 2, _mems_2)
    mems_2 = _mems_2[inds]
    sample_values_2 = _sample_values_2[inds]

    # Construct memoryref test cases.
    mem_refs = vcat([memoryref(m) for m in mems_1], [memoryref(m, 2) for m in mems_2])
    return mem_refs, vcat(sample_values_1, sample_values_2)
end

function generate_data_test_cases(rng_ctor, ::Val{:memory})
    return vcat(_mems()[1], _mem_refs()[1], [randn(2), Any[]])
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    rng = rng_ctor(123)
    mems, _ = _mems()
    mem_refs, sample_mem_ref_values = _mem_refs()

    assignable_refs = Iterators.filter(
        x -> length(x[1].mem) >= Core.memoryrefoffset(x[1]),
        zip(mem_refs, sample_mem_ref_values),
    )
    test_cases = vcat(

        # Rules for `Memory`
        (true, :stability, nothing, Memory{Float64}, undef, 5),
        (true, :stability, nothing, Memory{Memory{Float64}}, undef, 5),
        [(false, :stability_and_allocs, nothing, lgetfield, m, Val(:length)) for m in mems],
        [(false, :stability_and_allocs, nothing, lgetfield, m, Val(1)) for m in mems],
        [(false, :none, nothing, getfield, m, :length) for m in mems],
        [(false, :none, nothing, getfield, m, 1) for m in mems],

        # Rules for `MemoryRef`
        [
            (false, :none, nothing, memoryref_isassigned, mem_ref, :not_atomic, bc) for
            mem_ref in mem_refs for bc in [false, true]
        ],
        [
            (false, :none, nothing, memoryrefget, mem_ref, :not_atomic, bc) for
            mem_ref in filter(isassigned, mem_refs) for bc in [false, true]
        ],
        [(false, :none, nothing, memoryrefnew, mem) for mem in mems],
        [
            (false, :none, nothing, memoryrefnew, mem, 1) for
            mem in filter(x -> length(x.mem) > Core.memoryrefoffset(x), mem_refs)
        ],
        [
            (false, :none, nothing, memoryrefnew, mem, 1, bc) for
            mem in filter(x -> length(x.mem) > Core.memoryrefoffset(x), mem_refs) for
            bc in [false, true]
        ],
        [(false, :none, nothing, memoryrefoffset, mem_ref) for mem_ref in mem_refs],
        [
            (
                false,
                :none,
                nothing,
                lmemoryrefset!,
                mem_ref,
                sample_value,
                Val(:not_atomic),
                bc,
            ) for (mem_ref, sample_value) in assignable_refs for
            bc in [Val(false), Val(true)]
        ],
        [
            (false, :none, nothing, memoryrefset!, mem_ref, sample_value, :not_atomic, bc)
            for (mem_ref, sample_value) in assignable_refs for bc in [false, true]
        ],
        (
            false,
            :stability,
            nothing,
            unsafe_copyto!,
            randn(rng, 10).ref,
            randn(rng, 8).ref,
            5,
        ),
        (
            false,
            :stability,
            nothing,
            unsafe_copyto!,
            memoryref(randn(rng, 10).ref, 2),
            memoryref(randn(rng, 8).ref, 3),
            4,
        ),
        (
            false,
            :stability,
            nothing,
            unsafe_copyto!,
            [randn(rng, 10), randn(rng, 5)].ref,
            [randn(rng, 10), randn(rng, 3)].ref,
            2,
        ),
        (
            false,
            :none,
            nothing,
            unsafe_copyto!,
            memoryref(fill!(Memory{Any}(undef, 3), 4.0), 1),
            memoryref(Memory{Any}(undef, 2)),
            2,
        ),

        # Rules for `Array`
        (false, :stability, nothing, _new_, Vector{Float64}, randn(rng, 10).ref, (10,)),
        (
            false,
            :stability,
            nothing,
            _new_,
            Vector{Vector{Float64}},
            [randn(rng, 10), randn(rng, 5)].ref,
            (2,),
        ),
        (false, :none, nothing, _new_, Vector{Any}, [1, randn(rng, 5)].ref, (2,)),
        (false, :stability, nothing, _new_, Matrix{Float64}, randn(rng, 12).ref, (4, 3)),
        (
            false,
            :stability,
            nothing,
            _new_,
            Array{Float64,3},
            randn(rng, 12).ref,
            (4, 1, 3),
        ),
        [
            (false, :stability, nothing, lgetfield, randn(rng, 10), f) for
            f in [Val(:ref), Val(:size), Val(1), Val(2)]
        ],
        [(false, :none, nothing, getfield, randn(rng, 10), f) for f in [:ref, :size, 1, 2]],
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            randn(rng, 10),
            Val(:ref),
            randn(rng, 10).ref,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            randn(rng, 10),
            Val(1),
            randn(rng, 10).ref,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            randn(rng, 10),
            Val(:size),
            (10,),
        ),
        (false, :stability_and_allocs, nothing, lsetfield!, randn(rng, 10), Val(2), (10,)),
        (false, :none, nothing, setfield!, randn(rng, 10), :ref, randn(rng, 10).ref),
        (false, :none, nothing, setfield!, randn(rng, 10), 1, randn(rng, 10).ref),
        (false, :none, nothing, setfield!, randn(rng, 10), :size, (10,)),
        (false, :none, nothing, setfield!, randn(rng, 10), 2, (10,)),
        (false, :stability, nothing, copy, randn(10)),
        (false, :stability, nothing, fill!, fill!(Memory{Int8}(undef, 5), 0), Int8(1)),
        (false, :stability, nothing, fill!, fill!(Memory{UInt8}(undef, 5), 0), UInt8(1)),
    )
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    rng = rng_ctor(123)
    x = Memory{Float64}(randn(rng, 10))
    test_cases = Any[
        (true, :none, nothing, Array{Float64,0}, undef),
        (true, :none, nothing, Array{Float64,1}, undef, 5),
        (true, :none, nothing, Array{Float64,2}, undef, 5, 4),
        (true, :none, nothing, Array{Float64,3}, undef, 5, 4, 3),
        (true, :none, nothing, Array{Float64,4}, undef, 5, 4, 3, 2),
        (true, :none, nothing, Array{Float64,5}, undef, 5, 4, 3, 2, 1),
        (true, :none, nothing, Array{Float64,0}, undef, ()),
        (true, :none, nothing, Array{Float64,4}, undef, (2, 3, 4, 5)),
        (true, :none, nothing, Array{Float64,5}, undef, (2, 3, 4, 5, 6)),
        (false, :none, nothing, copy, Memory{Float64}(randn(5))),
        (false, :none, nothing, copy, Memory{Any}([randn(5), 5.0])),
        (false, :none, nothing, copy, randn(5, 4)),
        (false, :none, nothing, Base._deletebeg!, randn(5), 0),
        (false, :none, nothing, Base._deletebeg!, randn(5), 2),
        (false, :none, nothing, Base._deletebeg!, randn(5), 5),
        (false, :none, nothing, Base._deleteend!, randn(5), 2),
        (false, :none, nothing, Base._deleteend!, randn(5), 5),
        (false, :none, nothing, Base._deleteend!, randn(5), 0),
        (false, :none, nothing, Base._deleteat!, randn(5), 2, 2),
        (false, :none, nothing, Base._deleteat!, randn(5), 1, 5),
        (false, :none, nothing, Base._deleteat!, randn(5), 5, 1),
        (false, :none, nothing, fill!, rand(Int8, 5), Int8(2)),
        (false, :none, nothing, fill!, rand(UInt8, 5), UInt8(2)),
        (false, :none, nothing, fill!, Memory{Int8}(rand(Int8, 5)), Int8(3)),
        (false, :none, nothing, fill!, Memory{UInt8}(rand(UInt8, 5)), UInt8(5)),
        (true, :none, nothing, Base._growbeg!, randn(5), 3),
        (true, :none, nothing, Base._growend!, randn(5), 3),
        (true, :none, nothing, Base._growat!, randn(5), 2, 2),
        (false, :none, nothing, sizehint!, randn(5), 10),
        (false, :none, nothing, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
        (
            false,
            :none,
            nothing,
            unsafe_copyto!,
            [rand(3) for _ in 1:5],
            2,
            [rand(4) for _ in 1:4],
            1,
            3,
        ),
        (
            false,
            :none,
            nothing,
            unsafe_copyto!,
            Vector{Any}(undef, 5),
            2,
            Any[rand() for _ in 1:4],
            1,
            3,
        ),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x, 1), memoryref(x), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x, 2), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x, 4), 3), x),
    ]
    memory = Any[]
    return test_cases, memory
end
