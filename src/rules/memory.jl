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

# MemoryRef's element-wise V mirrors the referenced primal; unassigned slots have no value.
function TestUtils._chunked_v_invariant(p::MemoryRef, v::MemoryRef, c::IdDict)
    haskey(c, v) && return true
    c[v] = nothing
    (isassigned(p) && isassigned(v)) || return true
    return TestUtils._chunked_v_invariant(p[], v[], c)
end

function TestUtils.has_equal_data_internal(
    x::Memory{P}, y::Memory{P}, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P}
    length(x) == length(y) || return false
    id_pair = (x, y)
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

# Only differing extents reach this allocating path; callers handle first/full coverage inline.
function _increment_todo!(
    c::IdDict{Any,Any},
    buf,
    want::UnitRange{Int},
    prev::Union{UnitRange{Int},Vector{UnitRange{Int}}},
)
    covered = prev isa UnitRange{Int} ? [prev] : prev
    todo = _uncovered(covered, want)
    push!(covered, want)
    c[buf] = covered
    return todo
end

function increment_internal!!(c::IncCache, x::Memory{P}, y::Memory{P}) where {P}
    x === y && return x
    # Keyed on the BUFFER, so an `Array` and the `Memory` backing it agree however they are spelt.
    # A `Memory` always spans itself, so it claims the whole buffer.
    full() = _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
    c isa NoCache && return full()
    prev = get(c, x, nothing)
    prev === true && return x
    if prev === nothing
        c[x] = true
        return full()
    end
    todo = _increment_todo!(
        c, x, 1:length(x), prev::Union{UnitRange{Int},Vector{UnitRange{Int}}}
    )
    for piece in todo, i in piece
        if isbitstype(P) || (isassigned(x, i) && isassigned(y, i))
            x[i] = increment_internal!!(c, x[i], y[i])
        end
    end
    return x
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

function _scale_internal(c::MaybeCache, a::Float64, t::Memory{T}) where {T}
    haskey(c, t) && return c[t]::Memory{T}
    t′ = Memory{T}(undef, length(t))
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

function tangent_to_primal_internal!!(
    x::Memory{P}, t::Memory{<:Any}, c::MaybeCache
) where {P}
    haskey(c, x) && return c[x]::Memory{P}
    c[x] = x
    return _map_if_assigned!(x, x, t) do xn, tn
        return tangent_to_primal_internal!!(xn, tn, c)
    end
end
function primal_to_tangent_internal!!(
    t::Memory{T}, x::Memory{<:Any}, c::MaybeCache
) where {T}
    haskey(c, x) && return c[x]::Memory{T}
    c[x] = t
    return _map_if_assigned!(t, t, x) do tn, xn
        return primal_to_tangent_internal!!(tn, xn, c)
    end
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
    x === y && return x
    # Key on target storage so reshapes accumulate once. Including the source would double
    # contributions when a shared target receives independently stored forward gradients.
    full() = (_map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y); x)
    c isa NoCache && return full()
    xr = getfield(x, :ref)
    buf = xr.mem
    prev = get(c, buf, nothing)
    prev === true && return x
    off = Core.memoryrefoffset(xr)
    want = off:(off + length(x) - 1)
    if prev === nothing
        # Full buffers use an interned sentinel; partial extents leave a complement to fill.
        c[buf] = _spans_memory(x, xr) ? true : want
        return full()
    end
    todo = _increment_todo!(
        c, buf, want, prev::Union{UnitRange{Int},Vector{UnitRange{Int}}}
    )
    # Buffer position `p` is array index `p - off + 1`.
    for piece in todo, p in piece
        i = p - off + 1
        if isbitstype(eltype(T)) || (isassigned(x, i) && isassigned(y, i))
            x[i] = increment_internal!!(c, x[i], y[i])
        end
    end
    return x
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Array)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:Array}
    haskey(c, t) && return c[t]::T
    # Preserve backing-storage sharing before finite differences call `_add_to_primal`.
    tr = getfield(t, :ref)
    if _spans_memory(t, tr)
        t′ = Base.wrap(Array, construct_ref(tr, _scale_internal(c, a, tr.mem)), size(t))::T
        c[t] = t′
        return t′
    end
    t′ = T(undef, size(t)...)
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

# Deduplicate backing storage shared by distinct arrays (e.g. reshapes). The cache must use
# identity: equality would conflate unrelated zeroed buffers and undercount dimensions.
@inline function _dot_storage(x::Array)
    r = getfield(x, :ref)
    return (r.mem, Core.memoryrefoffset(r), length(x))
end
@inline _dot_storage(x::Memory) = (x, 1, length(x))

# Record covered positions per buffer pair to deduplicate overlapping extents. Positions share
# an index space only at equal offsets; differing offsets must retain their element pairing.
function _uncovered(covered::Vector{UnitRange{Int}}, r::UnitRange{Int})
    pieces = [r]
    for cr in covered
        isempty(pieces) && break
        next = UnitRange{Int}[]
        for p in pieces
            lo, hi = max(first(p), first(cr)), min(last(p), last(cr))
            if lo > hi
                push!(next, p)                       # disjoint
            else
                first(p) < lo && push!(next, first(p):(lo - 1))
                hi < last(p) && push!(next, (hi + 1):last(p))
            end
        end
        pieces = next
    end
    return pieces
end

for A in (Array, Memory)
    @eval function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:$A}
        bitstype = Val(isbitstype(eltype(T)))
        tb, to, tl = _dot_storage(t)
        sb, so, _ = _dot_storage(s)
        full() =
            sum(eachindex(t, s); init=0.0) do i
                if bitstype isa Val{true} || (isassigned(t, i) && isassigned(s, i))
                    _dot_internal(c, t[i], s[i])::Float64
                else
                    0.0
                end
            end
        # Different offsets have no shared index space for coverage.
        (c isa NoCache || to != so) && return full()
        k = (:dot_positions, tb, sb)
        prev = get(c, k, nothing)
        want = to:(to + tl - 1)
        # Store a bare range on first sight; allocate a vector only for repeated buffer pairs.
        if prev === nothing
            c[k] = want
            return full()
        end
        covered = prev isa UnitRange{Int} ? [prev] : prev::Vector{UnitRange{Int}}
        pieces = _uncovered(covered, want)
        push!(covered, want)
        c[k] = covered
        return sum(pieces; init=0.0) do piece
            sum(piece; init=0.0) do pos
                i = eachindex(t, s)[pos - to + 1]
                if bitstype isa Val{true} || (isassigned(t, i) && isassigned(s, i))
                    _dot_internal(c, t[i], s[i])::Float64
                else
                    0.0
                end
            end
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, x::Array{P,N}, t::Array{<:Any,N}, unsafe::Bool
) where {P,N}
    key = (x, t, unsafe)
    haskey(c, key) && return c[key]::Array{P,N}
    # Reuse perturbed backing Memory to preserve sharing between arrays in finite differences.
    # Both arrays must span their buffers: spare capacity may differ between primal and tangent.
    # The fallback handles spare capacity but does not preserve sharing.
    xr, tr = getfield(x, :ref), getfield(t, :ref)
    if _spans_memory(x, xr) && _spans_memory(t, tr)
        mem = _add_to_primal_internal(c, xr.mem, tr.mem, unsafe)
        x′ = Base.wrap(Array, construct_ref(xr, mem), size(x))::Array{P,N}
        c[key] = x′
        return x′
    end
    x′ = Array{P,N}(undef, size(x)...)
    c[key] = x′
    return _map_if_assigned!((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x′, x, t)
end

# Both callers recurse before caching the new array; reference eltypes could form cycles.
@inline function _spans_memory(x::Array, r::MemoryRef)
    isbitstype(eltype(x)) || return false
    return Core.memoryrefoffset(r) == 1 && length(r.mem) == length(x)
end

function tangent_to_primal_internal!!(
    x::Array{P,N}, t::Array{<:Any,N}, c::MaybeCache
) where {P,N}
    haskey(c, x) && return c[x]::Array{P,N}
    c[x] = x
    return _map_if_assigned!(x, x, t) do xn, tn
        return tangent_to_primal_internal!!(xn, tn, c)
    end
end
function primal_to_tangent_internal!!(
    t::Array{T,N}, x::Array{<:Any,N}, c::MaybeCache
) where {T,N}
    haskey(c, x) && return c[x]::Array{T,N}
    c[x] = t
    return _map_if_assigned!(t, t, x) do tn, xn
        return primal_to_tangent_internal!!(tn, xn, c)
    end
end

# Rules

@is_primitive(
    MinimalCtx, Tuple{typeof(unsafe_copyto!),MemoryRef{P},MemoryRef{P},Int} where {P}
)
# Copy adjacent lane columns as one overlap-safe memmove, including complex elements.
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    src::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    n::Lifted,
) where {Nw,P<:NDualEltype}
    _n = primal(n)
    unsafe_copyto!(primal(dest), primal(src), _n)
    if _n > 0
        dv, sv = tangent(dest), tangent(src)
        unsafe_copyto!(
            Nfwd._block_column_ref(getfield(dv, :partials_ref), getfield(dv, :col), Nw),
            Nfwd._block_column_ref(getfield(sv, :partials_ref), getfield(sv, :col), Nw),
            Nw * _n,
        )
    end
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
    x::MemoryRef{P}, y::MemoryRef{P}, equal_undefs::Bool, d::IdDict{Any,Bool}
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

function tangent_to_primal_internal!!(x::MemoryRef, tx, c::MaybeCache)
    return construct_ref(x, tangent_to_primal_internal!!(x.mem, tx.mem, c))
end
function primal_to_tangent_internal!!(tx, x::MemoryRef, c::MaybeCache)
    return construct_ref(x, primal_to_tangent_internal!!(tx.mem, x.mem, c))
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
    ::Lifted{typeof(lmemoryrefget),Nw},
    x::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    _ordering::Lifted{<:Val},
    _boundscheck::Lifted{<:Val},
) where {Nw,P<:NDualEltype}
    ordering = primal(_ordering)
    bc = primal(_boundscheck)
    y = memoryrefget(primal(x), _val(ordering), _val(bc))
    v = tangent(x)
    colref = Nfwd._block_column_ref(getfield(v, :partials_ref), getfield(v, :col), Nw)
    dy_partials = Nfwd._read_lanes(colref, Val(Nw))
    return Lifted{P,Nw}(y, _scalar_ndual(y, dy_partials))
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
    ::Lifted{typeof(memoryrefget),Nw},
    x::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    _ordering::Lifted{Symbol},
    _boundscheck::Lifted{Bool},
) where {Nw,P<:NDualEltype}
    ordering = primal(_ordering)
    bc = primal(_boundscheck)
    y = memoryrefget(primal(x), ordering, bc)
    v = tangent(x)
    colref = Nfwd._block_column_ref(getfield(v, :partials_ref), getfield(v, :col), Nw)
    dy_partials = Nfwd._read_lanes(colref, Val(Nw))
    return Lifted{P,Nw}(y, _scalar_ndual(y, dy_partials))
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

@inline function frule!!(
    ::Lifted{typeof(memoryrefnew),Nw},
    x::Lifted{Memory{P},Nw,<:NDualArray{P,Nw,1,Memory{P}}},
) where {Nw,P<:NDualEltype}
    # Share the Memory V's block (column j ↔ mem slot j); the fresh ref is at slot 1 = column 1.
    y = memoryrefnew(primal(x))
    block = getfield(tangent(x), :partials_block)
    return Lifted{MemoryRef{P},Nw}(y, NDualMemoryRef{P,Nw,Memory{P}}(y, block, 1))
end
@inline function rrule!!(f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:Memory})
    return CoDual(memoryrefnew(x.x), memoryrefnew(x.dx)), NoPullback(f, x)
end

# JuliaLang/julia#58768 adds indexed Memory construction in 1.13; slot ii is block column ii.
@inline function frule!!(
    ::Lifted{typeof(memoryrefnew),Nw},
    x::Lifted{Memory{P},Nw,<:NDualArray{P,Nw,1,Memory{P}}},
    ii::Lifted,
    rest::Vararg{Lifted,K},
) where {Nw,P<:NDualEltype,K}
    y = memoryrefnew(primal(x), primal(ii), map(primal, rest)...)
    block = getfield(tangent(x), :partials_block)
    return Lifted{MemoryRef{P},Nw}(y, NDualMemoryRef{P,Nw,Memory{P}}(y, block, primal(ii)))
end

@inline function frule!!(
    ::Lifted{typeof(memoryrefnew),Nw},
    x::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    ii::Lifted,
    rest::Vararg{Lifted,K},
) where {Nw,P<:NDualEltype,K}
    a = (primal(ii), map(primal, rest)...)
    y = memoryrefnew(primal(x), a...)
    # Same block, column advanced in lockstep with the primal ref's offset.
    v = tangent(x)
    newcol = getfield(v, :col) + primal(ii) - 1
    return Lifted{MemoryRef{P},Nw}(
        y,
        NDualMemoryRef{P,Nw,Memory{P}}(
            y, getfield(v, :partials_ref), getfield(v, :ncols), newcol
        ),
    )
end
@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:Union{Memory,MemoryRef}}, ii::CoDual{Int}
)
    return CoDual(memoryrefnew(x.x, ii.x), memoryrefnew(x.dx, ii.x)), NoPullback(f, x, ii)
end

@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)},
    x::CoDual{<:Union{Memory,MemoryRef}},
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
    ::Lifted{typeof(lmemoryrefset!),Nw},
    x::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    value::Lifted{P,Nw},
    ::Lifted{Val{ordering},Nw},
    ::Lifted{Val{boundscheck},Nw},
) where {Nw,P<:NDualEltype,ordering,boundscheck}
    memoryrefset!(primal(x), primal(value), ordering, boundscheck)
    v = tangent(x)
    colref = Nfwd._block_column_ref(getfield(v, :partials_ref), getfield(v, :col), Nw)
    vals = ntuple(lane -> _nfwd_dual_partial(tangent(value), lane), Val(Nw))
    Nfwd._write_lanes!(colref, vals, Val(Nw))
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
    memoryrefset!(x.dx, zero_tangent(value.x, value.dx), _val(ordering), _val(bc))

    function isbits_lmemoryrefset!_adjoint(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(x.dx, _val(ordering), _val(bc))))
        memoryrefset!(x.x, old_x[1], _val(ordering), _val(bc))
        memoryrefset!(x.dx, old_x[2], _val(ordering), _val(bc))
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, isbits_lmemoryrefset!_adjoint
end

@inline function frule!!(
    ::Lifted{typeof(memoryrefset!),Nw},
    x::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    value::Lifted{P,Nw},
    ordering::Lifted{Symbol},
    boundscheck::Lifted{Bool},
) where {Nw,P<:NDualEltype}
    ord = primal(ordering)
    bc = primal(boundscheck)
    memoryrefset!(primal(x), primal(value), ord, bc)
    v = tangent(x)
    colref = Nfwd._block_column_ref(getfield(v, :partials_ref), getfield(v, :col), Nw)
    vals = ntuple(lane -> _nfwd_dual_partial(tangent(value), lane), Val(Nw))
    Nfwd._write_lanes!(colref, vals, Val(Nw))
    return value
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

# Element-wise V memory operations thread primal and V refs together, preserving recursive
# coherence. Plain MemoryRef V dispatch is disjoint from NDualMemoryRef; forward-over-reverse
# uses it for Vector{Tuple{pullback}} storage.
@static if VERSION >= v"1.11-rc4"
    @inline function frule!!(
        ::Lifted{typeof(memoryrefnew),Nw},
        x::Lifted{<:Union{Memory,MemoryRef},Nw,<:Union{Memory,MemoryRef}},
        args::Vararg{Lifted,K},
    ) where {Nw,K}
        a = map(primal, args)
        yp = memoryrefnew(primal(x), a...)
        return Lifted{typeof(yp),Nw}(yp, memoryrefnew(tangent(x), a...))
    end
    @inline function frule!!(
        ::Lifted{typeof(memoryrefget),Nw},
        x::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        ordering::Lifted,
        boundscheck::Lifted,
    ) where {Nw}
        ord = primal(ordering)
        bc = primal(boundscheck)
        y = memoryrefget(primal(x), ord, bc)
        return Lifted{typeof(y),Nw}(y, memoryrefget(tangent(x), ord, bc))
    end
    @inline function frule!!(
        ::Lifted{typeof(lmemoryrefget),Nw},
        x::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        ::Lifted{Val{ordering}},
        ::Lifted{Val{boundscheck}},
    ) where {Nw,ordering,boundscheck}
        y = lmemoryrefget(primal(x), Val(ordering), Val(boundscheck))
        return Lifted{typeof(y),Nw}(
            y, lmemoryrefget(tangent(x), Val(ordering), Val(boundscheck))
        )
    end
    @inline function frule!!(
        ::Lifted{typeof(lmemoryrefset!),Nw},
        x::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        value::Lifted,
        ::Lifted{Val{ordering}},
        ::Lifted{Val{boundscheck}},
    ) where {Nw,ordering,boundscheck}
        lmemoryrefset!(primal(x), primal(value), Val(ordering), Val(boundscheck))
        lmemoryrefset!(tangent(x), tangent(value), Val(ordering), Val(boundscheck))
        return value
    end
    @inline function frule!!(
        ::Lifted{typeof(memoryrefset!),Nw},
        x::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        value::Lifted,
        ordering::Lifted{Symbol},
        boundscheck::Lifted{Bool},
    ) where {Nw}
        ord = primal(ordering)
        bc = primal(boundscheck)
        memoryrefset!(primal(x), primal(value), ord, bc)
        memoryrefset!(tangent(x), tangent(value), ord, bc)
        return value
    end
    # Array growth must copy element-wise V refs alongside primal refs.
    @inline function frule!!(
        ::Lifted{typeof(unsafe_copyto!),Nw},
        dest::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        src::Lifted{<:MemoryRef,Nw,<:MemoryRef},
        n::Lifted,
    ) where {Nw}
        _n = primal(n)
        unsafe_copyto!(primal(dest), primal(src), _n)
        unsafe_copyto!(tangent(dest), tangent(src), _n)
        return dest
    end
    # Size is primal metadata; ref is V storage. Include integer field aliases.
    @inline function frule!!(
        ::Lifted{typeof(lsetfield!),Nw},
        value::Lifted{<:Array,Nw,<:Array},
        ::Lifted{Val{name}},
        x::Lifted,
    ) where {Nw,name}
        lsetfield!(primal(value), Val(name), primal(x))
        lsetfield!(
            tangent(value),
            Val(name),
            (name === :size || name === 2) ? primal(x) : tangent(x),
        )
        return x
    end
    # Whole-buffer NoDual slots arise in forward-over-reverse storage (e.g. Stack Int32 blocks).
    # Canonical standalone seeds always produce wrappers, so HVP/Hessian tests cover these
    # methods rather than the per-rule registry.
    @inline function frule!!(
        ::Lifted{typeof(memoryrefnew),Nw},
        x::Lifted{<:Union{Memory,MemoryRef},Nw,NoDual},
        args::Vararg{Lifted,K},
    ) where {Nw,K}
        yp = memoryrefnew(primal(x), map(primal, args)...)
        return Lifted{typeof(yp),Nw}(yp, NoDual())
    end
    @inline function frule!!(
        ::Lifted{typeof(memoryrefget),Nw},
        x::Lifted{<:MemoryRef,Nw,NoDual},
        ordering::Lifted,
        boundscheck::Lifted,
    ) where {Nw}
        yp = memoryrefget(primal(x), primal(ordering), primal(boundscheck))
        return Lifted{typeof(yp),Nw}(yp, NoDual())
    end
    @inline function frule!!(
        ::Lifted{typeof(lmemoryrefget),Nw},
        x::Lifted{<:MemoryRef,Nw,NoDual},
        ::Lifted{Val{ordering}},
        ::Lifted{Val{boundscheck}},
    ) where {Nw,ordering,boundscheck}
        yp = lmemoryrefget(primal(x), Val(ordering), Val(boundscheck))
        return Lifted{typeof(yp),Nw}(yp, NoDual())
    end
    @inline function frule!!(
        ::Lifted{typeof(lmemoryrefset!),Nw},
        x::Lifted{<:MemoryRef,Nw,NoDual},
        value::Lifted,
        ::Lifted{Val{ordering}},
        ::Lifted{Val{boundscheck}},
    ) where {Nw,ordering,boundscheck}
        lmemoryrefset!(primal(x), primal(value), Val(ordering), Val(boundscheck))
        return value
    end
    @inline function frule!!(
        ::Lifted{typeof(memoryrefset!),Nw},
        x::Lifted{<:MemoryRef,Nw,NoDual},
        value::Lifted,
        ordering::Lifted{Symbol},
        boundscheck::Lifted{Bool},
    ) where {Nw}
        memoryrefset!(primal(x), primal(value), primal(ordering), primal(boundscheck))
        return value
    end
    @inline function frule!!(
        ::Lifted{typeof(unsafe_copyto!),Nw},
        dest::Lifted{<:MemoryRef,Nw,NoDual},
        src::Lifted{<:MemoryRef,Nw,NoDual},
        n::Lifted,
    ) where {Nw}
        unsafe_copyto!(primal(dest), primal(src), primal(n))
        return dest
    end
end

# Core.memoryrefsetonce!
# Core.memoryrefswap!
# Core.set_binding_type!

# _new_ and _new_-adjacent rules for Memory, MemoryRef, and Array.

@static if VERSION >= v"1.12-"
    @is_primitive MinimalCtx Tuple{typeof(Core.memorynew),Type{<:Memory},Int}
    # Delegate to constructor dispatch so both allocation spellings select the same V.
    function frule!!(
        ::Lifted{typeof(Core.memorynew),Nw}, ::Lifted{Type{Memory{P}},Nw}, n::Lifted
    ) where {Nw,P}
        return frule!!(zero_lifted(Val(Nw), Memory{P}), zero_lifted(Val(Nw), undef), n)
    end
    function rrule!!(
        ::CoDual{typeof(Core.memorynew)}, ::CoDual{Type{Memory{P}}}, n::CoDual{Int}
    ) where {P}
        return rrule!!(zero_fcodual(Memory{P}), zero_fcodual(undef), n)
    end
end

@is_primitive MinimalCtx Tuple{Type{<:Memory},UndefInitializer,Int}
function frule!!(
    ::Lifted{Type{Memory{P}},Nw}, ::Lifted{UndefInitializer,Nw}, n::Lifted
) where {Nw,P<:NDualEltype}
    x = Memory{P}(undef, primal(n))
    # Zero-initialized partials block covering the fresh memory (column j ↔ mem slot j).
    return Lifted{Memory{P},Nw}(x, NDualArray{P,Nw,1,Memory{P}}(x))
end
function rrule!!(
    ::CoDual{Type{Memory{P}}}, ::CoDual{UndefInitializer}, n::CoDual{Int}
) where {P}
    x = Memory{P}(undef, primal(n))
    # Fresh tangents must be zero even when the primal allocation contains stale data.
    dx = zero_tangent_internal(x, NoCache())
    return CoDual(x, dx), NoPullback((NoRData(), NoRData(), NoRData()))
end

# The NDualEltype overload handles floats/complex; other elements use element-wise V.
# Readable isbits slots need coherent zero duals to avoid copying garbage partials; reference
# slots stay #undef until the parallel memoryrefset! writes them.
@generated function frule!!(
    ::Lifted{Type{Memory{P}},Nw}, ::Lifted{UndefInitializer,Nw}, n::Lifted
) where {Nw,P}
    # isbitstype is structural, but dual_type must resolve in the returned expression at the
    # call world so later extension overloads apply (e.g. pullback closures capturing CuArrays).
    # Computing it in the generator would bake in generic recursion into CuPtr{Nothing}.
    fill_expr = if isbitstype(P)
        :(@inbounds for i in eachindex(dv)
            dv[i] = zero_dual(Val($Nw), x[i])
        end)
    else
        nothing
    end
    return quote
        x = Memory{$P}(undef, primal(n))
        MemV = dual_type(Val($Nw), Memory{$P})
        MemV === NoDual && return Lifted{Memory{$P},$Nw}(x, NoDual())
        dv = MemV(undef, primal(n))
        $fill_expr
        return Lifted{Memory{$P},$Nw}(x, dv)
    end
end
function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{MemoryRef{P}}},
    ptr_or_offset::CoDual{Ptr{Nothing}},
    mem::CoDual{Memory{P}},
) where {P}
    y = _new_(MemoryRef{P}, ptr_or_offset.x, mem.x)
    # `ptr_or_offset.dx` is an `VoidPtrTangent`: the address plus what it is laid out in. The
    # tangent `MemoryRef` wants the address alone.
    dy = _new_(MemoryRef{tangent_type(P)}, _void_ptr_addr(ptr_or_offset.dx), mem.dx)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function frule!!(
    ::Lifted{typeof(_new_),Nw},
    ::Lifted{Type{Array{P,D}},Nw},
    ref::Lifted{MemoryRef{P},Nw,NDualMemoryRef{P,Nw,Memory{P}}},
    sz::Lifted,
) where {Nw,P<:NDualEltype,D}
    _sz = primal(sz)
    y = _new_(Array{P,D}, primal(ref), _sz)
    # Share the ref's backing from its column onward so array/ref V mutations alias.
    v = tangent(ref)
    flat = _new_(
        Vector{P},
        Nfwd._block_column_ref(getfield(v, :partials_ref), getfield(v, :col), Nw),
        (Nw * prod(_sz),),
    )
    block = NDualBlock{P,D + 1}(flat, (Nw, _sz...))
    return Lifted{Array{P,D},Nw}(y, NDualArray{P,Nw,D,Array{P,D}}(y, block))
end
# Build the element-wise V array over its V ref, mirroring reverse tangent storage.
@inline function frule!!(
    ::Lifted{typeof(_new_),Nw},
    ::Lifted{Type{Array{P,D}},Nw},
    ref::Lifted{<:MemoryRef,Nw,<:MemoryRef},
    sz::Lifted,
) where {Nw,P,D}
    _sz = primal(sz)
    y = _new_(Array{P,D}, primal(ref), _sz)
    yv = _new_(Array{dual_type(Val(Nw), P),D}, tangent(ref), _sz)
    return Lifted{Array{P,D},Nw}(y, yv)
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
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_genericmemory_copy},Nw},
    ::Lifted,
    ::Lifted{Tuple{Val{Any}},Nw},
    ::Lifted{Val{0},Nw},
    ::Lifted{Val{:ccall},Nw},
    x::Lifted{Memory{P},Nw,<:NDualArray{P,Nw,1,Memory{P}}},
) where {Nw,P<:NDualEltype}
    new_primal = copy(primal(x))
    new_block = copy(getfield(tangent(x), :partials_block))
    return Lifted{Memory{P},Nw}(
        new_primal, NDualArray{P,Nw,1,Memory{P}}(new_primal, new_block)
    )
end
# Shallow-copy element-wise V, matching primal semantics (including UInt8 metadata and Any).
function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_genericmemory_copy},Nw},
    ::Lifted,
    ::Lifted{Tuple{Val{Any}},Nw},
    ::Lifted{Val{0},Nw},
    ::Lifted{Val{:ccall},Nw},
    x::Lifted{<:Memory,Nw,<:Memory},
) where {Nw}
    new_primal = copy(primal(x))
    return Lifted{typeof(new_primal),Nw}(new_primal, copy(tangent(x)))
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

# Forward field projection uses generic lgetfield and _get_lifted_field in misc.jl.
function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:Memory,<:Memory},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_length = name === 1 || name === :length
    # Tag pointer layout so retyped loads/stores cannot dereference zero-byte tangent buffers.
    dy = if wants_length
        NoFData()
    elseif eltype(x.dx) === NoTangent
        VoidPtrTangent(Ptr{Nothing}(0), NoTangent)
    else
        VoidPtrTangent(bitcast(Ptr{Nothing}, x.dx.ptr), eltype(x.dx))
    end
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:MemoryRef,<:MemoryRef},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name,order}
    y = getfield(primal(x), name, order)
    wants_offset = name === 1 || name === :ptr_or_offset
    # Stride-tagged rather than an unbacked address for a zero-size element type, as in `Memory`.
    dy = if wants_offset
        if eltype(x.dx) === NoTangent
            VoidPtrTangent(Ptr{Nothing}(0), NoTangent)
        else
            VoidPtrTangent(bitcast(Ptr{Nothing}, x.dx.ptr_or_offset), eltype(x.dx))
        end
    else
        x.dx.mem
    end
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
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

const _MemTypes = Union{Memory,MemoryRef,DenseArray,Array}

function rrule!!(
    f::CoDual{typeof(lgetfield)}, x::CoDual{<:_MemTypes,<:_MemTypes}, name::CoDual{<:Val}
)
    y, adj = rrule!!(f, x, name, zero_fcodual(Val(:not_atomic)))
    ternary_lgetfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_lgetfield_adjoint
end

# Runtime-name forward getfield in builtins.jl avoids an unstable Val round-trip.
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

# Forward getfield lives in builtins.jl so it is available on Julia 1.10 too.
function rrule!!(
    f::CoDual{typeof(getfield)},
    x::CoDual{<:_MemTypes,<:_MemTypes},
    name::CoDual{<:Union{Int,Symbol}},
)
    y, adj = rrule!!(f, x, name, zero_fcodual(:not_atomic))
    ternary_getfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_getfield_adjoint
end

# Retarget the block in place to preserve V aliases. Size sets its flat length; ref points at
# the incoming column. Array growth writes them separately, allowing transient inconsistency
# only while nothing reads the block.
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{<:Array,Nw,<:NDualArray},
    ::Lifted{Val{name},Nw},
    x::Lifted,
) where {Nw,name}
    setfield!(primal(value), name, primal(x))
    block = getfield(tangent(value), :partials_block)
    parent = getfield(block, :parent)
    if name === :size || name === 2
        setfield!(parent, :size, (Nw * prod(primal(x)),))
    else
        xv = tangent(x)
        setfield!(
            parent,
            :ref,
            Nfwd._block_column_ref(getfield(xv, :partials_ref), getfield(xv, :col), Nw),
        )
    end
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
# Copy primal and V together; the four-parameter NDualArray prefix also matches complex V.
function frule!!(
    ::Lifted{typeof(copy),N}, a::Lifted{Array{T,D},N,<:NDualArray{T,N,D,Array{T,D}}}
) where {N,T<:NDualEltype,D}
    yp = copy(primal(a))
    block = copy(getfield(tangent(a), :partials_block))
    return Lifted{Array{T,D},N}(yp, NDualArray{T,N,D,Array{T,D}}(yp, block))
end
@inline function frule!!(::Lifted{typeof(copy),N}, a::Lifted{<:Array,N,<:Array}) where {N}
    return Lifted{typeof(primal(a)),N}(copy(primal(a)), copy(tangent(a)))
end

@is_primitive MinimalCtx Tuple{typeof(fill!),Array{<:Union{UInt8,Int8}},Integer}
@is_primitive MinimalCtx Tuple{typeof(fill!),Memory{<:Union{UInt8,Int8}},Integer}
function frule!!(
    ::Lifted{typeof(fill!),Nw}, a::Lifted{<:Union{Array{V},Memory{V}},Nw}, x::Lifted
) where {Nw,V<:Union{UInt8,Int8}}
    fill!(primal(a), primal(x))
    return a
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

function hand_written_rule_test_cases(rng_ctor, ::Val{:memory})
    rng = rng_ctor(123)
    mems, _ = _mems()
    mem_refs, sample_mem_ref_values = _mem_refs()

    assignable_refs = Iterators.filter(
        x -> length(x[1].mem) >= Core.memoryrefoffset(x[1]),
        zip(mem_refs, sample_mem_ref_values),
    )
    test_cases = vcat(
        @static(
            if VERSION >= v"1.12-"
                [
                    (true, :stability, nothing, Core.memorynew, Memory{Float64}, 5),
                    (true, :stability, nothing, Core.memorynew, Memory{Float64}, 10),
                    (true, :stability, nothing, Core.memorynew, Memory{Int}, 5),
                ]
            else
                []
            end
        ),

        # Rules for `Memory`
        (true, :stability, nothing, Memory{Float64}, undef, 5),
        (true, :stability, nothing, Memory{Memory{Float64}}, undef, 5),
        # Non-scalar isbits element: exercises the generic `Memory{P}(undef, n)` constructor rule
        # for a struct/tuple eltype (the `bitstype` branch of `_dot_internal`).
        (true, :stability, nothing, Memory{Tuple{Float64,Int}}, undef, 4),
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
        [
            (false, :none, nothing, lmemoryrefget, mem_ref, Val(:not_atomic), bc) for
            mem_ref in filter(isassigned, mem_refs) for bc in [Val(false), Val(true)]
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
        if VERSION >= v"1.13-"
            [
                (false, :none, nothing, memoryrefnew, mem, length(mem), bc...) for
                mem in filter(!isempty, mems) for bc in ((), (false,), (true,))
            ]
        else
            []
        end,
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
        # Element-wise V parent (non-NDualEltype elements): the Symbol AND Int field forms
        # must both project through `_get_lifted_field(::Array, ...)`.
        [
            (false, :none, nothing, getfield, [randn(rng, 2) for _ in 1:3], f) for
            f in [:ref, :size, 1, 2]
        ],
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
        # Element-wise V must thread the tangent ref for integer field aliases too.
        (
            false,
            :none,
            nothing,
            lsetfield!,
            [randn(rng, 2) for _ in 1:3],
            Val(1),
            [randn(rng, 2) for _ in 1:3].ref,
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

function derived_rule_test_cases(rng_ctor, ::Val{:memory})
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
        # Complex MemoryRef V must be NDualMemoryRef during array growth; reverse is the oracle.
        (
            false,
            :none,
            nothing,
            x -> (v=ComplexF64[]; push!(v, x); push!(v, 2x); sum(abs2, v)),
            ComplexF64(1.0, 2.0),
        ),
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
    # A `memoryref` reaching past the partials block must refuse rather than read slack.
    slack_v = Float64[]
    sizehint!(slack_v, 16)
    for i in 1:3
        push!(slack_v, Float64(i))
    end
    push!(
        test_cases,
        (
            false,
            :none,
            (throws=(ArgumentError, "past the 3 partials columns"), mode=ForwardMode),
            memoryref_into_capacity_slack,
            slack_v,
        ),
    )
    push!(
        test_cases,
        (false, :none, (mode=ForwardMode,), memoryref_across_realloc, collect(1.0:4.0)),
    )
    push!(
        test_cases,
        (false, :none, (mode=ForwardMode,), memoryref_mem_across_realloc, collect(1.0:4.0)),
    )
    push!(
        test_cases,
        (
            false,
            :none,
            (mode=ForwardMode,),
            memoryref_mem_projected_then_realloc,
            collect(1.0:4.0),
        ),
    )
    push!(
        test_cases,
        (false, :allocs, (mode=ForwardMode,), memoryref_mem_sum, collect(1.0:4.0)),
    )
    memory = Any[slack_v]
    return test_cases, memory
end

@static if VERSION >= v"1.11-"
    # A `Vector` grown under `sizehint!` has `length(array) < length(backing Memory)`, so an
    # offset validated against the Memory can land in capacity slack with no partials column.
    # The primal read is legal there (uninitialised capacity); the block read would not be.
    function memoryref_into_capacity_slack(v)
        return Core.memoryrefget(Core.memoryrefnew(getfield(v, :ref), 5), :not_atomic, true)
    end

    # A ref held across reallocation must retain OLD derivative storage (JVP 4, not 1).
    function memoryref_across_realloc(v)
        r = getfield(v, :ref)
        push!(v, 0.0)
        pop!(v)
        return 3.0 * v[1] + Core.memoryrefget(Core.memoryrefnew(r, 2), :not_atomic, false)
    end

    # The resize happens AFTER the projection, so no check at projection time can see it: the
    # block must already be pinned to storage a later reallocation cannot retarget.
    function memoryref_mem_projected_then_realloc(v)
        r = getfield(v, :ref)
        m = getfield(r, :mem)
        push!(v, 0.0)
        pop!(v)
        v[1] = 7.0 * v[1]
        return Core.memoryrefget(Core.memoryrefnew(m), :not_atomic, false)
    end

    # Projecting .mem must not allocate: its temporary array header must not escape, since
    # length and bounds checks can project once per element.
    function memoryref_mem_sum(v)
        m = getfield(getfield(v, :ref), :mem)
        r = Core.memoryrefnew(m)
        y = 0.0
        for i in 1:length(m)
            y += Core.memoryrefget(Core.memoryrefnew(r, i, false), :not_atomic, false)
        end
        return y
    end

    # Projecting .mem after resizing must retain old storage (derivative 1, not 7).
    function memoryref_mem_across_realloc(v)
        r = getfield(v, :ref)
        push!(v, 0.0)
        pop!(v)
        v[1] = 7.0 * v[1]
        m = getfield(r, :mem)
        return Core.memoryrefget(Core.memoryrefnew(m), :not_atomic, false)
    end
end
