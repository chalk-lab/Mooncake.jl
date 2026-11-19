# Loaded after nfwd/Nfwd.jl for the NDual carriers. Forward V types encode differentiability,
# wrappers and layout, so frules may need separate methods for shapes one rrule handles.
# When porting a reverse rule, preserve its conventions as well as its formula.

"""
    ImmutableDual{T<:NamedTuple}

Forward representation of an immutable struct `P` under the structural lift. Its
`fields::T` is a `NamedTuple{fieldnames(P)}` whose `i`th entry has type
`V_i = dual_type(Val(N), fieldtype(P, i))`, or `PossiblyUninitTangent{V_i}` where the field
may be uninitialised, as in `tangent_type`.
"""
struct ImmutableDual{T<:NamedTuple}
    fields::T
end

Base.:(==)(x::ImmutableDual, y::ImmutableDual) = x.fields == y.fields

"""
    MutableDual{T<:NamedTuple}

Mutable counterpart to `ImmutableDual`.
"""
mutable struct MutableDual{T<:NamedTuple}
    fields::T
    # Unfilled shell: caches must register it before its fields, which may cycle back to it.
    MutableDual{T}() where {T<:NamedTuple} = new{T}()
    MutableDual{T}(fields) where {T<:NamedTuple} = new{T}(fields)
end
@inline MutableDual(fields::T) where {T<:NamedTuple} = MutableDual{T}(fields)

Base.:(==)(x::MutableDual, y::MutableDual) = x.fields == y.fields

# Type caches keep the source type alive. Its aligned address leaves the tag's low bit
# free to mark a source whose primal and partial layouts match.
struct PointerDual{N}
    lanes::NTuple{N,Ptr{Cvoid}}
    source::UInt
    function PointerDual{N}(lanes::NTuple{N,Ptr{Cvoid}}, ::Type{T}) where {N,T}
        if T !== Union{} && T <: PointerDual
            T === PointerDual{N} && all(p -> p == lanes[1], lanes) || throw(
                ArgumentError(
                    "Pointer metadata must have matching width and one shared buffer."
                ),
            )
        end
        return new{N}(lanes, _pointer_source(T))
    end
    function PointerDual(lanes::NTuple{N,Ptr{Cvoid}}, v::PointerDual{N}) where {N}
        v.source === _pointer_source(PointerDual{N}) &&
            return PointerDual{N}(lanes, PointerDual{N})
        return new{N}(lanes, v.source)
    end
end

@inline function _pointer_source(::Type{T}) where {T}
    T === NoDual ||
        T === Nothing ||
        T === NoTangent ||
        _carries_partials(T) ||
        return _pointer_source(NoTangent)
    source = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), T))
    return source | UInt(isbitstype(T) && T === tangent_type(T) && sizeof(T) > 0)
end

@inline PointerDual(lanes::Tuple{Ptr{T},Vararg{Ptr{T},N}}) where {N,T} = PointerDual{N+1}(
    map(p -> bitcast(Ptr{Cvoid}, p), lanes), T
)
Base.:(==)(x::PointerDual, y::PointerDual) = x.source === y.source && x.lanes == y.lanes

"""
    Lifted{P, N, V}

Forward-mode slot holding a primal of type `P` and its width-`N` forward representation `V`.

- `primal::P`: the user's primal value; a mutable primal is aliased, not copied. Where `rep`
  holds the primal itself (`NDualArray`, `NDualMemoryRef`), the constructor takes it from
  `rep`, so the two cannot disagree.
- `rep::V`: the forward representation. A caller of `Lifted{P, N, V}` must ensure that `V` is
  `dual_type(Val(N), P)` when that is concrete, and a subtype of it otherwise. Only a `Ptr`
  slot is checked: its `rep` must be a `PointerDual{N}`.

Rules dispatch on `Lifted{P, N}` and read a slot with `primal`, `tangent` and, for lane `k`,
`tangent(slot, k)`. They build outputs with `Lifted{P, N}(primal, rep)`, which infers `V`;
given a type `X` as its primal, it sets `P` to `Type{X}`, whatever `P` the caller wrote.

Width `N == 1` is ordinary forward mode; `N >= 2` is chunked forward mode.
"""
struct Lifted{P,N,V}
    primal::P
    rep::V
    # Inner, so every construction path takes the `Ptr` check and `_slot_primal`.
    function Lifted{P,N,V}(primal, rep) where {P,N,V}
        P <: Ptr &&
            !(rep isa PointerDual{N}) &&
            throw(
                ArgumentError(
                    "Pointer lanes must have width $N and one source element type."
                ),
            )
        return new{P,N,V}(_slot_primal(primal, rep), rep)
    end
end

# NDualArray / NDualMemoryRef name their primal storage; take the slot's primal from there.
@unstable @inline _slot_primal(primal, rep) = primal
@inline _slot_primal(::Any, rep::NDualArray) = getfield(rep, :primal)
@static if VERSION >= v"1.11-rc4"
    @inline _slot_primal(::Any, rep::NDualMemoryRef) = getfield(rep, :primal)
end

"""
    NoDual

Forward-mode sentinel for "this slot has no derivative". Used as the
`V` of `Lifted{P, N, V}` and as the return of `dual_type(Val(N), P)` for
primal types without a meaningful tangent space (integers, booleans,
symbols, modules, types, …).

Parallels reverse-mode's `NoTangent` but lives in the forward-mode V
layer so the dual_type return shape is self-documenting. The two are
intentionally distinct types so a code path's mode (forward vs reverse)
is visible from its argument types.
"""
struct NoDual end

@inline function Lifted{P,N}(primal::P, rep::V) where {P,N,V}
    return Lifted{P,N,V}(primal, rep)
end

@inline function Lifted{P,N}(p::P, lanes::Tuple{Ptr{T},Vararg{Ptr{T}}}) where {P<:Ptr,N,T}
    return Lifted{P,N}(p, PointerDual(lanes))
end
@inline function Lifted{P,N}(p::P, ::NoDual) where {P<:Ptr,N}
    return Lifted{P,N}(p, PointerDual{N}(ntuple(_ -> Ptr{Cvoid}(C_NULL), Val(N)), NoDual))
end

# Sharpen `P` for a `Type{X}` primal: type-constructor rules dispatch on the type slot, and
# `Lifted` is invariant, so a wide `Lifted{DataType,N}` slot would miss them. `P_wide` is what the
# caller type-applied and is ignored.
@inline function Lifted{P_wide,N}(primal::Type{P_sharp}, rep::V) where {P_wide,P_sharp,N,V}
    # A phantom `TypeVar` leaves `P_sharp` unbound, as in the `CoDual(::Type{P})` constructor.
    return Lifted{(@isdefined(P_sharp) ? Type{P_sharp} : typeof(primal)),N,V}(primal, rep)
end

# Accessors — mirror the `CoDual` API.
primal(d::Lifted) = d.primal
tangent(d::Lifted) = d.rep

function verify_lifted_type(::Lifted{P,N,V}) where {P,N,V}
    !isconcretetype(P) || V === dual_type(Val(N), P)
end

extract(d::Lifted) = (primal(d), tangent(d))

function Base.:(==)(a::Lifted, b::Lifted)
    return primal(a) == primal(b) && tangent(a) == tangent(b)
end

# Whole-array accessors — O(1) by aliasing.
@inline primal(a::NDualArray) = a.primal
@inline tangent(a::NDualArray) = Nfwd._lane_views(a)

# `tangent(slot, lane)` materialises lane `lane` as a `tangent_type(P)`; `tangent_view` is
# an array's writable lane. Leaf methods key on V, leaving `P` free for abstract slots.
@inline function tangent(x::Lifted{P,N,NDual{T,N}}, lane::Integer) where {P,T<:IEEEFloat,N}
    return tangent(x).partials[lane]
end
@inline function tangent(x::Lifted{P,N,<:NDualArray}, lane::Integer) where {P,N}
    v = tangent(x)
    return copyto!(similar(getfield(v, :primal)), tangent_view(v, lane))
end
@inline function Nfwd.tangent_view(x::Lifted{P,N,<:NDualArray}, lane::Integer) where {P,N}
    return Nfwd.tangent_view(tangent(x), lane)
end
# A `Ref` is a mutable struct, so its reverse tangent is a
# `MutableTangent{@NamedTuple{x::PossiblyUninitTangent{P}}}`, NOT a bare scalar.
@inline function tangent(
    x::Lifted{<:Base.RefValue{P},N,<:NDualRef}, lane::Integer
) where {P<:FloatOrComplex,N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
# `NDualMemoryRef` / `MemoryRef` / `Core.memoryref` are 1.11+; gate to avoid an
# `UndefVarError` at precompile on 1.10.
@static if VERSION >= v"1.11-rc4"
    @inline function tangent(x::Lifted{P,N,<:NDualMemoryRef}, lane::Integer) where {P,N}
        # A copy into a same-length `Memory` at the same offset. Block column `j` pairs with
        # slot `o - c + j`; slots the block does not cover stay zero.
        v = tangent(x)
        p = getfield(v, :primal)
        block = Nfwd._reconstruct_block(v)
        c = getfield(v, :col)
        o = Core.memoryrefoffset(p)
        T = eltype(block)
        mem = fill!(Memory{T}(undef, length(p.mem)), zero(T))
        @inbounds for j in 1:size(block, 2)
            s = o - c + j
            1 <= s <= length(mem) && (mem[s] = block[lane, j])
        end
        return _memoryref_at(mem, o)
    end
    # `Core.memoryref(mem, offset)` is out-of-bounds for an empty `mem` (length 0); the 1-arg
    # `Core.memoryref(mem)` gives the canonical empty ref. Empty `Memory` is a valid primal
    # (empty arrays/vectors), so re-build a `MemoryRef` V through this guard.
    @inline _memoryref_at(mem, offset) =
        length(mem) == 0 ? Core.memoryref(mem) : Core.memoryref(mem, offset)
    # Element-wise `MemoryRef` V: project the lane through the `.mem`, then re-`memoryref`.
    @inline function tangent(
        x::Lifted{P,N,V}, lane::Integer
    ) where {P<:MemoryRef,N,V<:MemoryRef}
        return _materialise_lane(x, lane, IdDict{Any,Any}())
    end
end
@inline function tangent(
    x::Lifted{P,N,Complex{NDual{R,N}}}, lane::Integer
) where {P,R<:IEEEFloat,N}
    v = tangent(x)
    return Complex(real(v).partials[lane], imag(v).partials[lane])
end
# NoDual means no forward partial, not necessarily NoTangent in reverse (Ptr and CUDA's
# DataRef differ). Rebuild from the primal, but refuse a noncanonical NoDual first.
@inline function tangent(x::Lifted{P,N,NoDual}, ::Integer) where {P,N}
    verify_lifted_type(x) || _throw_noncanonical_nodual(Val(N), P)
    return uninit_tangent(primal(x))
end
@noinline function _throw_noncanonical_nodual(::Val{N}, ::Type{P}) where {N,P}
    throw(
        ArgumentError(
            "a slot over the primal `$P` carries the forward value `NoDual`, but its " *
            "canonical forward value is `$(dual_type(Val(N), P))` and its reverse " *
            "tangent `$(tangent_type(P))`. `NoDual` is valid only where `dual_type` " *
            "declares it.",
        ),
    )
end
# Rebuild the container shape even when its elements have no forward partials.
@inline function tangent(x::Lifted{P,N,<:AbstractArray{NoDual}}, ::Integer) where {P,N}
    return uninit_tangent(primal(x))
end
# Element-wise container V: the same-shape array of each element's lane, mirroring reverse
# `tangent_type(Array{T}) === Array{tangent_type(T)}`; undefined slots stay undefined.
@inline function tangent(x::Lifted{P,N,V}, lane::Integer) where {P,N,V<:AbstractArray}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
@inline function tangent(x::Lifted{P,N,PointerDual{N}}, lane::Integer) where {P<:Ptr,N}
    return bitcast(tangent_type(P), tangent(x).lanes[lane])
end
@inline function tangent(x::Lifted{P,N,<:ImmutableDual}, lane::Integer) where {P,N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
@inline function tangent(x::Lifted{P,N,<:MutableDual}, lane::Integer) where {P,N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
@inline function tangent(x::Lifted{P,N,<:Tuple}, lane::Integer) where {P<:Tuple,N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end
@inline function tangent(x::Lifted{<:NamedTuple,N,<:NamedTuple}, lane::Integer) where {N}
    return _materialise_lane(x, lane, IdDict{Any,Any}())
end

# Public 2-tuple unpack at the slot boundary. Width-1 only — chunked slots
# carry per-lane derivatives in their V and have no single native-tangent
# unpack; use per-lane access (`tangent(x, lane)`) for width N > 1.
@inline unlift(x::Lifted{P,1}) where {P} = (primal(x), tangent(x, 1))

# Materialise reverse tangent_type(P), never a writable view. Register mutable shells
# before recursion so cycles terminate and aliased children share one reverse tangent.
# Only leaf Vs may fall back to the lane accessor. Diagnose unsupported aggregates here
# instead of passing a wrong-shaped tangent to reverse arithmetic.
@inline function _materialise_lane(x::Lifted{P,N}, lane::Integer, cache::IdDict) where {P,N}
    t = tangent(x, lane)
    t isa tangent_type(P) || _throw_not_a_leaf_v(P, t)
    return _register_leaf(primal(x), t, cache)
end
# Registering a fresh leaf keeps two aliased leaves under one slot on one reverse tangent.
# `t` is built first so the cache hit can be asserted to its concrete type.
@inline function _register_leaf(p, t, cache::IdDict)
    ismutable(p) || return t
    haskey(cache, p) && return cache[p]::typeof(t)
    cache[p] = t
    return t
end
@noinline function _throw_not_a_leaf_v(::Type{P}, t) where {P}
    throw(
        ArgumentError(
            "there is no method materialising a reverse tangent from the forward value " *
            "of `$P`, so it fell back to the lane accessor, which gave a `$(typeof(t))` " *
            "where the reverse tangent is `$(tangent_type(P))`. That fallback is right " *
            "only for a leaf; an aggregate needs its own `_materialise_lane` " *
            "rebuilding a reverse tangent from its components.",
        ),
    )
end
function _materialise_lane(
    x::Lifted{P,N,<:MutableDual}, lane::Integer, cache::IdDict
) where {P,N}
    p = primal(x)
    haskey(cache, p) && return cache[p]
    Tt = tangent_type(P)
    shell = Tt()
    cache[p] = shell
    nt = tangent(x).fields
    field_tangents = map(keys(nt)) do name
        return _materialise_field_lane(Val(N), P, p, name, getfield(nt, name), lane, cache)
    end
    # Coerce into the declared reverse backing `tangent_type(P)` (an abstract field is
    # stored widened, e.g. `a::Any`, matching reverse mode).
    shell.fields = fieldtype(Tt, :fields)(field_tangents)
    return shell
end
function _materialise_lane(
    x::Lifted{P,N,<:ImmutableDual}, lane::Integer, cache::IdDict
) where {P,N}
    nt = tangent(x).fields
    p = primal(x)
    field_tangents = map(keys(nt)) do name
        return _materialise_field_lane(Val(N), P, p, name, getfield(nt, name), lane, cache)
    end
    return Tangent(fieldtype(tangent_type(P), :fields)(field_tangents))
end
# Restrict element-wise materialisation to tuple primals. Child slots take the held
# value's type, as at `_materialise_field_lane`.
function _materialise_lane(
    x::Lifted{P,N,<:Tuple}, lane::Integer, cache::IdDict
) where {P<:Tuple,N}
    p = primal(x)
    v = tangent(x)
    return ntuple(length(v)) do i
        pe = p[i]
        return _materialise_lane(Lifted{typeof(pe),N}(pe, v[i]), lane, cache)
    end
end
function _materialise_lane(
    x::Lifted{P,N,<:NamedTuple}, lane::Integer, cache::IdDict
) where {P,N}
    p = primal(x)
    v = tangent(x)
    names = keys(v)
    return NamedTuple{names}(
        map(names) do name
            pf = getfield(p, name)
            return _materialise_lane(
                Lifted{typeof(pf),N}(pf, getfield(v, name)), lane, cache
            )
        end,
    )
end
# A block-backed array IS a leaf: it cannot reference back into itself, so it needs no
# shell. It does need registering, being storage two aliased positions must share.
function _materialise_lane(
    x::Lifted{P,N,<:NDualArray}, lane::Integer, cache::IdDict
) where {P,N}
    _register_leaf(primal(x), tangent(x, lane), cache)
end
@static if VERSION >= v"1.11-rc4"
    # ... except that reverse derives an `Array`'s tangent from its `Memory`'s, so an
    # `Array` beside its `Memory` must share one buffer, as `_derived_array_dual` seeds it.
    function _materialise_lane(
        x::Lifted{P,N,<:NDualArray}, lane::Integer, cache::IdDict
    ) where {P<:Array,N}
        p = primal(x)
        haskey(cache, p) && return cache[p]::tangent_type(P)
        E = eltype(P)
        r = getfield(p, :ref)
        mem = r.mem
        buf = getfield(getfield(getfield(tangent(x), :partials_block), :parent), :ref).mem
        # Only a `_window_block` into the `Memory`'s `N`-scaled block has one to tie.
        len = N * length(mem)
        length(buf) == len || return _register_leaf(p, tangent(x, lane), cache)
        block = Nfwd.PartialsBlock{E,2}(
            Base.wrap(Array, memoryref(buf), (len,))::Vector{E}, (N, length(mem))
        )
        memv = NDualArray{E,N,1,Memory{E}}(mem, block)
        tmem = _materialise_lane(Lifted{Memory{E},N}(mem, memv), lane, cache)
        t = Base.wrap(Array, construct_ref(r, tmem), size(p))::tangent_type(P)
        cache[p] = t
        return t
    end

    # The `MemoryRef` half: register the materialised `Memory` when the block spans it all.
    function _materialise_lane(
        x::Lifted{P,N,<:NDualMemoryRef}, lane::Integer, cache::IdDict
    ) where {P<:MemoryRef,N}
        p = primal(x)
        v = tangent(x)
        t = tangent(x, lane)
        off = Core.memoryrefoffset(p)
        (getfield(v, :ncols) == length(p.mem) && getfield(v, :col) == off) || return t
        return _memoryref_at(_register_leaf(p.mem, t.mem, cache), off)
    end
end
@static if VERSION >= v"1.11-rc4"
    # Thread the cache, so the ref's `Memory` shares a lane with other headers over it.
    function _materialise_lane(
        x::Lifted{P,N,V}, lane::Integer, cache::IdDict
    ) where {P<:MemoryRef,N,V<:MemoryRef}
        p = primal(x)
        mem = _materialise_lane(Lifted{typeof(p.mem),N}(p.mem, tangent(x).mem), lane, cache)
        return _memoryref_at(mem, Core.memoryrefoffset(p))
    end
end
# An all-NoDual representation is a leaf only if its elements have NoTangent too.
@inline function _materialise_lane(
    x::Lifted{P,N,<:AbstractArray{NoDual}}, lane::Integer, cache::IdDict
) where {P,N}
    tangent_type(eltype(P)) === NoTangent &&
        return _register_leaf(primal(x), tangent(x, lane), cache)
    return _materialise_lane_elementwise(x, lane, cache)
end
function _materialise_lane(
    x::Lifted{P,N,V}, lane::Integer, cache::IdDict
) where {P,N,V<:AbstractArray}
    return _materialise_lane_elementwise(x, lane, cache)
end
function _materialise_lane_elementwise(
    x::Lifted{P,N}, lane::Integer, cache::IdDict
) where {P,N}
    p = primal(x)
    if p isa Union{_STRUCTURAL_CONTAINERS...}
        # Share storage between headers, as the lift and seed caches do.
        key = _structural_key(p)
        key, first = _extent_key(p, key, cache)
        if first !== nothing
            first isa tangent_type(P) && _same_window(first, p) && return first
            hkey = _header_key(p, key, first, tangent_type(eltype(P)))
            haskey(cache, hkey) && return cache[hkey]
            t = _shared_lane(p, first, tangent_type(eltype(P)))::tangent_type(P)
            cache[hkey] = t
        else
            t = _new_lane(p, tangent_type(eltype(P)))::tangent_type(P)
            cache[key] = t
        end
    else
        haskey(cache, p) && return cache[p]
        t = similar(p, tangent_type(eltype(P)))
        cache[p] = t
    end
    # Use the concrete `typeof(pe)`: an abstract `eltype(P)` has no fields to lift.
    _map_if_assigned!(
        (pe, ve) -> _materialise_lane(Lifted{typeof(pe),N}(pe, ve), lane, cache),
        t,
        p,
        tangent(x),
    )
    return t
end
# `Ref{P<:FloatOrComplex}` (V `NDualRef`): build the `MutableTangent` reverse mode gives it
# (a `Ref` is a mutable struct, registered in the alias cache before returning). Its `:x`
# field is non-always-init, so reverse wraps it in `PossiblyUninitTangent`.
function _materialise_lane(
    x::Lifted{<:Base.RefValue{P},N,<:NDualRef}, lane::Integer, cache::IdDict
) where {P<:FloatOrComplex,N}
    p = primal(x)
    haskey(cache, p) && return cache[p]
    Tt = tangent_type(typeof(p))
    shell = Tt()
    cache[p] = shell
    Ft = fieldtype(fieldtype(Tt, :fields), :x)
    part = tangent(x).partials[][lane]
    shell.fields = fieldtype(Tt, :fields)((Ft <: PossiblyUninitTangent ? Ft(part) : part,))
    return shell
end
# Undefined primal fields produce uninitialised reverse PUTs. Child slots use the held
# value's concrete type: a declared Any would break struct lookup or treat a tuple as lanes.
# Rt remains the declared reverse backing type, which admits the concrete result.
@inline _materialise_field_lane(
    ::Val{N}, ::Type{P}, p, name, vfield, lane, cache::IdDict
) where {N,P} =
    let pf = getfield(p, name)
        _materialise_lane(Lifted{typeof(pf),N}(pf, vfield), lane, cache)
    end
@inline function _materialise_field_lane(
    ::Val{N}, ::Type{P}, p, name, vfield::PossiblyUninitTangent, lane, cache::IdDict
) where {N,P}
    Rt = tangent_type(fieldtype(P, name))
    (is_init(vfield) && isdefined(p, name)) || return PossiblyUninitTangent{Rt}()
    pf = getfield(p, name)
    return PossiblyUninitTangent{Rt}(
        _materialise_lane(Lifted{typeof(pf),N}(pf, val(vfield)), lane, cache)
    )
end
@noinline function unlift(x::Lifted{P,N,V}) where {P,N,V}
    throw(
        ArgumentError(
            "unlift only supports width-1 Lifted slots; got Lifted{$P, $N, $V}. " *
            "Use `tangent(x, lane)` for per-lane access at width > 1.",
        ),
    )
end

# `NDualMemoryRef` (and its constructor) lives in `src/nfwd/Nfwd.jl`.
# Mooncake-namespace method extensions follow.

@static if VERSION >= v"1.11-rc4"
    @inline primal(a::NDualMemoryRef) = a.primal
    # Bulk/interface access reconstructs the shared (N, ncols) block from its backing ref.
    @inline tangent(a::NDualMemoryRef) = Nfwd._reconstruct_block(a)
end

struct NumericStorage end
struct StructuralStorage end

# Split numeric storage requires a concrete element type; unions (including bottom)
# use structural storage so each runtime element keeps its own forward representation.
@foldable @inline function _storage_kind(::Type{T}) where {T}
    return isconcretetype(T) && T <: FloatOrComplex ? NumericStorage() : StructuralStorage()
end

"""
    dual_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `N`-width forward-mode value type for a primal of
type `P`. This is the type of the inner `V` carried by a
`Lifted{P, N, V}` slot.

Shapes:

- `NDual{P,N}` for a float `P`; `Complex{NDual{R,N}}` for a complex float `Complex{R}`;
- for a concrete float or complex element type: `NDualArray` for an `Array` (and `Memory` on
  1.11+), whose primal aliases user storage and whose partials are a slot-local
  `(N, size...)` block; `NDualMemoryRef` for a `MemoryRef`; `NDualRef` for a `Base.RefValue`;
- element-wise for other `Array`, `Memory` and `MemoryRef` types, tuples and named tuples;
- `PointerDual{N}` for a `Ptr`;
- `ImmutableDual`/`MutableDual` over the per-field `NamedTuple` for other structs, with
  `PossiblyUninitTangent` around a field that may be uninitialised;
- `NTuple{N,tangent_type(P)}`, one tangent per lane, for a fieldless `P` whose tangent is not
  `NoTangent`, such as `Core.BFloat16`, which has no `NDual`;
- `NoDual` for a concrete `P` whose tangent is `NoTangent`.

A non-concrete `P` gives an upper bound, often `Any`.
"""
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat}
    return isconcretetype(P) ? NDual{P,N} : Any
end
# Each member mirrors a `tangent_type` method returning `NoTangent`. Other `Integer` subtypes
# can hold differentiable fields, so they take the structural method.
const NoDualPrimal = Union{
    Bool,
    Base.BitInteger,
    BigInt,
    Char,
    Symbol,
    Nothing,
    Type,
    TypeVar,
    Module,
    Expr,
    Cstring,
    Cwstring,
}
@foldable @inline dual_type(::Val{N}, ::Type{<:NoDualPrimal}) where {N} = NoDual
# Bottom type, mirroring `tangent_type(Union{}) === Union{}` and
# `lifted_type(Val(N), Union{}) === Union{}` (e.g. a Bottom-typed IR node after a
# guaranteed-throw call). Without this, `dual_type(Val(N), Union{})` MethodErrors.
@foldable @inline dual_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
@foldable @inline function dual_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}
end
@foldable @inline function dual_type(w::Val, ::Type{Array{T,D}}) where {T,D}
    return _dual_type(_storage_kind(T), w, Array{T,D})
end
@foldable @inline function _dual_type(
    ::NumericStorage, w::Val, ::Type{Array{T,D}}
) where {T,D}
    return Nfwd._ndual_array_V(Array{T,D}, w)
end
@foldable @inline function _dual_type(
    ::StructuralStorage, w::Val, ::Type{Array{T,D}}
) where {T,D}
    return Array{dual_type(w, T),D}
end
@foldable @inline function dual_type(w::Val, ::Type{Base.RefValue{P}}) where {P}
    return _dual_type(_storage_kind(P), w, Base.RefValue{P})
end
@foldable @inline function _dual_type(
    ::NumericStorage, ::Val{N}, ::Type{Base.RefValue{P}}
) where {N,P}
    return NDualRef{P,N}
end
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    # A phantom free-TypeVar `Tuple` leaves `P` unbound.
    @isdefined(P) || return Any
    # First, so a non-concrete but non-differentiable tuple collapses as in reverse.
    tangent_type(P) === NoTangent && return NoDual
    # A union of tuples has no single element list; `fieldcount` reports one member's.
    Base.unwrap_unionall(P) isa Union && return Any
    # A `Vararg` tuple has no finite element list.
    Base.isvatuple(P) && return Any
    V = _dual_tuple_v(Val(N), P)
    # A non-concrete tuple can have wholly non-differentiable concretisations, which collapse to
    # `NoDual`, so its V must admit `NoDual` too, as `tangent_type` does. A concrete one cannot.
    isconcretetype(P) && return V
    return Tuple{Vararg{NoDual,fieldcount(P)}} <: V ? Union{V,NoDual} : V
end
# Element-wise tuple V without the collapse gate. Generated, as `tangent_type(::Type{<:Tuple})` is:
# head/tail recursion overflows inference on long tuples, which can SIGSEGV (julia#17109).
@foldable @generated function _dual_tuple_v(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    Vs = map(i -> :(dual_type(Val(N), fieldtype(P, $i))), 1:fieldcount(P))
    return :(Tuple{$(Vs...)})
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{NamedTuple{names,T}}
) where {N,names,T<:Tuple}
    # Non-differentiable NamedTuples collapse to NoDual, matching reverse.
    tangent_type(NamedTuple{names,T}) === NoTangent && return NoDual
    # As in `tangent_type`: `Any` unless `DT` is concrete, which an abstract field prevents.
    DT = dual_type(Val(N), T)
    return isconcretetype(DT) ? NamedTuple{names,DT} : Any
end
@foldable @inline dual_type(::Val{N}, ::Type{Ptr{T}}) where {N,T} = PointerDual{N}
# MemoryRef canonical V (Julia 1.11+); `NDualMemoryRef` is defined in `src/nfwd/Nfwd.jl`.
# Memory itself is `<: AbstractArray{T, 1}` on 1.11+ — its canonical V is
# an NDualArray over `Memory{T}`: `Memory{T} → NDualArray{T, N, 1, Memory{T}}`.
@static if VERSION >= v"1.11-rc4"
    for C in (:Memory, :MemoryRef)
        @eval @foldable @inline function dual_type(w::Val, ::Type{$C{T}}) where {T}
            return _dual_type(_storage_kind(T), w, $C{T})
        end
        @eval @foldable @inline function _dual_type(
            ::StructuralStorage, w::Val, ::Type{$C{T}}
        ) where {T}
            return $C{dual_type(w, T)}
        end
    end
    @foldable @inline function _dual_type(
        ::NumericStorage, w::Val, ::Type{Memory{T}}
    ) where {T}
        return Nfwd._ndual_array_V(Memory{T}, w)
    end
    @foldable @inline function _dual_type(
        ::NumericStorage, ::Val{N}, ::Type{MemoryRef{T}}
    ) where {N,T}
        return NDualMemoryRef{T,N,Memory{T}}
    end
end

# Structural fallback: abstract P widens to Any; non-differentiable P gives NoDual.
@foldable @inline dual_type(w::Val, P::Type) = _dual_type(StructuralStorage(), w, P)
@foldable @generated function _dual_type(
    ::StructuralStorage, ::Val{N}, ::Type{P}
) where {N,P}
    isconcretetype(P) || return Any
    # Resolve extension-overloadable tangent_type in the returned expression at the call
    # world. Only the world-independent structural skeleton belongs in the generator.
    if fieldcount(P) == 0
        return :(tangent_type($P) === NoTangent ? NoDual : NTuple{$N,tangent_type($P)})
    end
    field_names = fieldnames(P)
    n_fields = fieldcount(P)
    inits = always_initialised(P)
    # Non-always-initialised fields are wrapped in `PossiblyUninitTangent`,
    # exactly as reverse-mode `tangent_type`, so the two stay coherent (e.g. a
    # lazily-built field like `LazyDerivedRule.rule`).
    field_dual_exprs = map(1:n_fields) do i
        base = :(dual_type(Val($N), $(fieldtype(P, i))))
        inits[i] ? base : :(PossiblyUninitTangent{$base})
    end
    inner_nt_type = :(NamedTuple{$field_names,Tuple{$(field_dual_exprs...)}})
    wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
    # Only a structural reverse tangent has a structural lift. Deferred, so a later extension
    # overload takes effect.
    Tt = ismutabletype(P) ? :MutableTangent : :Tangent
    msg =
        "`$P` has a custom reverse tangent type but no `dual_type` method, and the structural " *
        "forward lift applies only where the reverse tangent is a `$Tt`. Define " *
        "`dual_type(::Val{N}, ::Type{$P})` together with its seed factories and `lift`."
    return :(
        if tangent_type($P) === NoTangent
            NoDual
        elseif tangent_type($P) <: $Tt
            $wrapper{$inner_nt_type}
        else
            error($msg)
        end
    )
end

# Metatype kinds are concrete, but their runtime slots are `Lifted{Type{X}}`, which a bounded
# slot cannot hold.
@inline _is_metatype_kind(@nospecialize(T)) =
    T === DataType || T === UnionAll || T === Union || T === Core.TypeofBottom

# A widened `dual_type` is an upper bound, and `Lifted` is invariant in `V`, so the slot must
# leave `V` free rather than claim `Lifted{P,N,Any}`.
@inline function _concrete_lifted_type(::Val{N}, ::Type{P}) where {N,P}
    V = dual_type(Val(N), P)
    return isconcretetype(V) ? Lifted{P,N,V} : (Lifted{P,N,W} where {W})
end

@foldable @inline lifted_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
# An abstract tuple `P` widens to a UnionAll slot, since `Lifted` is invariant in `P`. `@foldable`
# keeps folding under coverage and exempts the `@isdefined(P)` guard from DispatchDoctor.
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    @isdefined(P) || return Lifted  # phantom free-TypeVar Tuple — broad `Lifted` slot, as the generic `lifted_type` guard below
    return if isconcretetype(P)
        _concrete_lifted_type(Val(N), P)
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# True when every member of `U` is non-differentiable (`tangent_type === NoTangent`), so
# `lifted_type` can split `U` into its members' slots.
function _all_nodual_union_members(@nospecialize(U))
    if U isa Union
        _all_nodual_union_members(U.a) && _all_nodual_union_members(U.b)
    else
        (U === Union{} || tangent_type(U) === NoTangent)
    end
end

# @foldable preserves folding under coverage and exempts the phantom-P guard from
# DispatchDoctor's unconditional static-parameter reads, as in the Tuple overload.
"""
    lifted_type(::Val{N}, ::Type{P}) -> Type

Return the slot type for a primal of type `P` at width `N`.

When `P` and `dual_type(Val(N), P)` are both concrete, this is
`Lifted{P, N, dual_type(Val(N), P)}`, except for the metatype kinds `DataType`, `UnionAll`,
`Union` and `Core.TypeofBottom`, whose runtime slots are `Lifted{Type{X}}`. `Type{X}` gives its
exact slot, and a union of non-differentiable non-tuple types the union of its members' slots.
Any other `P` except `Union{}` gives a `UnionAll` that leaves the unknown parameters free, since
`Lifted` is invariant in `P` and `V`.
"""
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    # A phantom `TypeVar` leaves `P` unbound.
    @isdefined(P) || return Lifted
    # A union of non-differentiable members splits into its members' slots, which do not box;
    # any other union keeps the UnionAll, since splitting it trips an OpaqueClosure assertion.
    P isa Union &&
        _all_nodual_union_members(P) &&
        return Union{lifted_type(Val(N), P.a),lifted_type(Val(N), P.b)}
    return if isconcretetype(P) && !_is_metatype_kind(P)
        _concrete_lifted_type(Val(N), P)
    elseif P <: Type
        # A bounded slot trips an OpaqueClosure return assertion on the sharpened `Lifted{Type{X}}`.
        Lifted{T,N,V} where {T,V}
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# `Type{X}` has one inhabitant, so its concrete slot is exact; the generic method would widen it
# (`isconcretetype(Type{X})` is false) and box it.
@foldable @inline function lifted_type(::Val{N}, ::Type{Type{X}}) where {N,X}
    return Lifted{Type{X},N,dual_type(Val(N), Type{X})}
end

let doc = """
        zero_dual(Val(N), x)
        uninit_dual(Val(N), x)
        randn_dual(Val(N), rng, x)

    Return the width-`N` representation `dual_type(Val(N), typeof(x))` of `x`, with zero,
    unspecified or `randn(rng)` partials in fresh storage. No cache is threaded, so parts of
    `x` that alias get separate partials, and a self-referential `x` is unsupported; use
    [`zero_lifted`](@ref) and its siblings for those.
    """
    for f in (:zero_dual, :uninit_dual)
        @eval @doc $doc $f(::Val, x)
    end
    @eval @doc $doc randn_dual(::Val, ::AbstractRNG, x)
end

@inline function zero_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

# Array and dict lifts index the seed by the primal's indices or keys; check it on entry.
@inline _check_seed_size(x, dx) = Nfwd._check_seed_axes(x, (dx,))
@inline _check_seed_size(::Base.RefValue, ::MutableTangent) = nothing
for C in (Array, Base.RefValue)
    D = C === Array ? Array : MutableTangent
    @eval @inline function lift(x::$C, dx::$D)
        _check_seed_size(x, dx)
        return _lift(_storage_kind(eltype(x)), x, dx)
    end
    @eval @inline function lift(x::$C{T}, dx::$D, c::Union{Nothing,IdDict}) where {T}
        _check_seed_size(x, dx)
        return _lift(_storage_kind(T), x, dx, c)
    end
end
@inline lift(x::P, dx::MutableTangent, c::Union{Nothing,IdDict}) where {P} = _lift(
    StructuralStorage(), x, dx, c
)
@static if VERSION >= v"1.11-rc4"
    # A seed ref's `[]` is at its own offset, so it must sit at the primal's.
    @inline function _check_seed_size(x::MemoryRef, dx::MemoryRef)
        _check_seed_size(x.mem, dx.mem)
        Core.memoryrefoffset(dx) == Core.memoryrefoffset(x) && return nothing
        throw(
            DimensionMismatch(
                "a seed ref at offset $(Core.memoryrefoffset(dx)) cannot carry the " *
                "partials of a primal ref at offset $(Core.memoryrefoffset(x))",
            ),
        )
    end
    for C in (Memory, MemoryRef)
        @eval @inline function lift(x::$C, dx::$C)
            _check_seed_size(x, dx)
            return _lift(_storage_kind(eltype(x)), x, dx)
        end
        @eval @inline function lift(x::$C, dx::$C, c::Union{Nothing,IdDict})
            _check_seed_size(x, dx)
            return _lift(_storage_kind(eltype(x)), x, dx, c)
        end
        @eval @inline _lift(k::StructuralStorage, x::$C, dx::$C) = _lift(k, x, dx, nothing)
    end
end
@inline _lift(k::StructuralStorage, x::Array, dx::Array) = _lift(k, x, dx, nothing)
@inline _lift(k, x::Base.RefValue, dx::MutableTangent) = _lift(k, x, dx, nothing)

"""
    lift(x, ẋ)

Return a width-1 [`Lifted`](@ref) slot for `x` whose direction is `ẋ`, a reverse-mode
tangent of `x` (`NoTangent()` is a zero direction). The primal aliases `x`; the partials
copy `ẋ`.

One cache spans the call, so within `x` one object reached twice, or two headers over one
buffer, share partials. Throws a `DimensionMismatch` if an array seed's size, or a
`MemoryRef` seed's offset, differs from its primal's, and an `ArgumentError` if seeds
reaching one storage differ.
"""
function lift end
@inline lift(x::T, ẋ::T) where {T<:IEEEFloat} = Lifted{T,1}(x, NDual{T,1}(x, (ẋ,)))
lift(x::FloatOrComplex, ẋ::FloatOrComplex) = _throw_tangent_leaf_type_error(x, ẋ)
@noinline function _throw_tangent_leaf_type_error(x, ẋ)
    throw(
        ArgumentError(
            "Tangent leaf type $(typeof(ẋ)) must match primal leaf type $(typeof(x))"
        ),
    )
end
@inline function _lift(
    ::NumericStorage, x::A, ẋ::A
) where {E<:FloatOrComplex,D,A<:Array{E,D}}
    return Lifted{A,1}(x, NDualArray{E,1,D,A}(x, (ẋ,)))
end
@inline function lift(x::Complex{R}, ẋ::Complex{R}) where {R<:IEEEFloat}
    re = NDual{R,1}(real(x), (real(ẋ),))
    im_ = NDual{R,1}(imag(x), (imag(ẋ),))
    return Lifted{Complex{R},1}(x, Complex{NDual{R,1}}(re, im_))
end
# `NoTangent` for a differentiable primal (a ChainRules result, say) is a zero derivative.
@inline lift(x, ::NoTangent) = zero_lifted(Val(1), x)
# Through a cache, a zero direction must still share storage with other visits to `x`.
@inline function lift(x, ẋ::NoTangent, c::IdDict)
    t = zero_tangent(x)
    return t isa NoTangent ? lift(x, ẋ) : lift(x, t, c)
end
@inline lift(x::Ptr, ::NoTangent, ::IdDict) = lift(x, NoTangent())
@inline lift(x::Ptr, ẋ::Ptr) = Lifted{typeof(x),1}(x, (ẋ,))
@static if VERSION >= v"1.11-rc4"
    # The seed is copied into a fresh block, as for `Array`. The eltype bound must match the
    # `Memory` lift's, or a ref falls through to the element-wise lift.
    @inline function _lift(
        ::NumericStorage, x::MemoryRef{T}, ẋ::MemoryRef{T}
    ) where {T<:FloatOrComplex}
        len = length(ẋ.mem)
        block = PartialsBlock{T,2}(undef, 1, len)
        copyto!(getfield(block, :parent), 1, ẋ.mem, 1, len)
        return Lifted{MemoryRef{T},1}(
            x, NDualMemoryRef{T,1,Memory{T}}(x, block, Core.memoryrefoffset(x))
        )
    end
    # The cache-threading form, which `_lift_backing` calls: aliased primals share one V.
    @inline function _lift(
        ::NumericStorage, x::MemoryRef{T}, ẋ::MemoryRef{T}, c::Union{Nothing,IdDict}
    ) where {T<:FloatOrComplex}
        c isa IdDict || return lift(x, ẋ)
        # Window the backing `Memory`'s V, which a ref shares storage with; lifting it first
        # also checks the seed against any earlier one for that storage.
        memv = tangent(lift(x.mem, ẋ.mem, c))
        haskey(c, x) && return c[x]::Lifted{MemoryRef{T},1}
        lifted = Lifted{MemoryRef{T},1}(
            x,
            NDualMemoryRef{T,1,Memory{T}}(
                x, getfield(memv, :partials_block), Core.memoryrefoffset(x)
            ),
        )
        c[x] = lifted
        return lifted
    end
    # `Memory{T}` (T<:IEEEFloat / Complex{<:IEEEFloat}) lifts to the NDualArray,
    # mirroring the `Array` overloads above; reached when a reverse rule's `Memory`
    # field is lifted under forward-over-reverse, or a Memory primal is seeded.
    @inline function _lift(
        ::NumericStorage, x::A, ẋ::A
    ) where {E<:FloatOrComplex,A<:Memory{E}}
        return Lifted{A,1}(x, NDualArray{E,1,1,A}(x, (ẋ,)))
    end
    @inline function _lift(
        ::NumericStorage, x::A, ẋ::A, c::Union{Nothing,IdDict}
    ) where {E<:FloatOrComplex,A<:Memory{E}}
        c isa IdDict || return lift(x, ẋ)
        owner = _storage_id(x)
        _check_seed!(c, owner, _storage_offset(x), ẋ)
        haskey(c, x) && return c[x]::lifted_type(Val(1), A)
        key = (owner, Val(:lift_storage))
        haskey(c, key) ||
            (c[key] = Lifted{typeof(owner),1}(owner, zero_dual(Val(1), owner)))
        block = Nfwd._window_block(
            getfield(tangent(c[key]::Lifted{typeof(owner),1}), :partials_block),
            Val(1),
            _storage_offset(x),
            (1, length(x)),
        )
        v = NDualArray{E,1,1,A}(x, block)
        # Backing capacities may differ; unused owner slots keep their zero seed.
        copyto!(tangent_view(v, 1), ẋ)
        lifted = Lifted{A,1}(x, v)
        c[x] = lifted
        return lifted
    end
    # General element-wise `MemoryRef` lift (non-float / nested / `Any` / `NoTangent`
    # element): lift the `.mem` via the Memory lift, then `memoryref` at the
    # offset. Numeric storage uses the `NDualMemoryRef` methods above.
    @inline function _lift(
        ::StructuralStorage, x::MemoryRef, ẋ::MemoryRef, c::Union{Nothing,IdDict}
    )
        mem_v = tangent(lift(x.mem, ẋ.mem, c))
        ref_v = _memoryref_at(mem_v, Core.memoryrefoffset(x))
        return Lifted{typeof(x),1,typeof(ref_v)}(x, ref_v)
    end
end
# Thread one aliasing cache through aggregate lifts, so aliased leaves share one V.
# Leaf/passthrough overloads may ignore the cache.
@inline lift(x, ẋ, ::Union{Nothing,IdDict}) = lift(x, ẋ)

# Coerce into the declared backing NamedTuple so abstract fields keep the canonical V.
# Guard possibly-uninitialised fields before reading the primal or reverse PUT.
@generated function _lift_backing(x, nt, ::Type{Backing}, c) where {Backing}
    names = Backing.parameters[1]
    Vfs = Backing.parameters[2].parameters
    exprs = map(enumerate(names)) do (i, name)
        Vf = Vfs[i]
        qn = QuoteNode(name)
        if Vf <: PossiblyUninitTangent
            return :(
                if isdefined(x, $qn)
                    $Vf(tangent(lift(getfield(x, $qn), val(getfield(nt, $qn)), c)))
                else
                    $Vf()
                end
            )
        else
            return :(tangent(lift(getfield(x, $qn), getfield(nt, $qn), c)))
        end
    end
    return :(Backing(($(exprs...),)))
end
# `Ref{P<:FloatOrComplex}` (V `NDualRef`): build the parallel partials buffer from the reverse
# tangent's scalar (its `:x` field is non-always-init, hence `PossiblyUninitTangent`-wrapped).
@inline function _lift(
    ::NumericStorage, x::Base.RefValue{P}, ẋ::MutableTangent, c::Union{Nothing,IdDict}
) where {P<:FloatOrComplex}
    # Register storage-owning leaves so repeated Refs share partials. Build here because
    # the two-argument entry delegates to this method with nothing.
    c isa IdDict &&
        (_check_seed!(c, x, 0, ẋ); haskey(c, x)) &&
        return c[x]::Lifted{Base.RefValue{P},1}
    lifted = Lifted{Base.RefValue{P},1}(
        x, NDualRef{P,1}(Base.RefValue{NTuple{1,P}}((val(ẋ.fields.x),)))
    )
    c isa IdDict && (c[x] = lifted)
    return lifted
end
@inline lift(x::P, ẋ::Tangent) where {P} = lift(x, ẋ, nothing)
@inline function lift(x::P, ẋ::Tangent, c::Union{Nothing,IdDict}) where {P}
    backing = fieldtype(dual_type(Val(1), P), 1)
    # One cache per top-level lift, so children aliased across fields share one V.
    d = c === nothing ? IdDict() : c
    return Lifted{P,1}(x, ImmutableDual(_lift_backing(x, ẋ.fields, backing, d)))
end
@inline lift(x::P, ẋ::MutableTangent) where {P} = lift(x, ẋ, nothing)
@inline function _lift(
    ::StructuralStorage, x::P, ẋ::MutableTangent, c::Union{Nothing,IdDict}
) where {P}
    backing = fieldtype(dual_type(Val(1), P), 1)
    LT = Lifted{P,1,MutableDual{backing}}
    # Register the shell before its fields, so a cycle back to `x` finds it.
    d = c === nothing ? IdDict() : c
    _check_seed!(d, x, 0, ẋ)
    haskey(d, x) && return d[x]::LT
    lifted = LT(x, MutableDual{backing}())
    d[x] = lifted
    lifted.rep.fields = _lift_backing(x, ẋ.fields, backing, d)
    return lifted
end
# Reached only for defined fields.
@inline lift(x, ẋ::PossiblyUninitTangent) = lift(x, val(ẋ))
@inline lift(x, ẋ::PossiblyUninitTangent, c::Union{Nothing,IdDict}) = lift(x, val(ẋ), c)
# Each lift copies into a fresh block, so aliased primals share one V only through the cache.
@inline function _lift(
    ::NumericStorage, x::A, ẋ::A, c::Union{Nothing,IdDict}
) where {E<:FloatOrComplex,D,A<:Array{E,D}}
    c isa IdDict || return lift(x, ẋ)
    # 1.10 has no backing `Memory`; its headers over one buffer have one length.
    sk = @static if VERSION >= v"1.11-rc4"
        _storage_id(getfield(x, :ref).mem)
    else
        (_storage_id(x), length(x))
    end
    _check_seed!(c, sk, _storage_offset(x), ẋ)
    haskey(c, x) && return c[x]::lifted_type(Val(1), A)
    # Share one owner block, then copy the caller's seed into the array window.
    lifted = @static if VERSION >= v"1.11-rc4"
        mem = sk
        # Reserving the owner's block does not mean its full seed has been copied.
        key = (mem, Val(:lift_storage))
        haskey(c, key) || (c[key] = Lifted{typeof(mem),1}(mem, zero_dual(Val(1), mem)))
        memv = tangent(c[key]::Lifted{typeof(mem),1})
        v = _derived_array_dual(Val(1), x, memv)
        copyto!(tangent_view(v, 1), ẋ)
        Lifted{A,1}(x, v)
    else
        # 1.10 has no backing `Memory` to window, so share through a STORAGE key and a block over
        # the cached block's parent — the same device the 1.10 seed path uses. Keyed separately
        # from the object key below, which cannot see that `a` and `reshape(a)` are one buffer.
        if haskey(c, sk)
            cached = tangent(c[sk]::Lifted)
            Lifted{A,1}(
                x,
                NDualArray{E,1,D,A}(
                    x, _share_block(getfield(cached, :partials_block), (1, size(x)...))
                ),
            )
        else
            owned = lift(x, ẋ)
            c[sk] = owned
            owned
        end
    end
    c[x] = lifted
    return lifted
end

# Seeds lifted into one storage so far, by offset and length in it. A later seed must agree
# with each where they overlap; a window never seeded before may hold anything. A repeat of
# a recorded seed, or of its window, adds nothing, so the cost stays linear in visits.
function _check_seed!(c::IdDict, key, off::Int, ẋ)
    T = Tuple{Int,Int,Any}
    seeds = get!(Vector{T}, c, (key, Val(:lift_seeds)))::Vector{T}
    n = ẋ isa AbstractArray ? length(ẋ) : 0
    any(((o, _, s),) -> o == off && s === ẋ, seeds) && return nothing
    if !isempty(seeds)
        seen = IdDict{Any,Nothing}()
        all(((o, _, s),) -> _seeds_agree(o, s, off, ẋ, seen), seeds) ||
            _throw_conflicting_seeds()
    end
    any(((o, m, _),) -> o == off && m == n, seeds) || push!(seeds, (off, n, ẋ))
    return nothing
end
function _seeds_agree(o::Int, a::AbstractArray, n::Int, b::AbstractArray, seen)
    return all(max(o, n):(min(o + length(a), n + length(b)) - 1)) do p
        i, j = p - o + 1, p - n + 1
        !(isassigned(a, i) && isassigned(b, j)) || _seed_eq(a[i], b[j], seen)
    end
end
_seeds_agree(::Int, a, ::Int, b, seen) = _seed_eq(a, b, seen)
# Exact, except that a NaN matches itself and signed zeros match: one seed never conflicts
# with an equal one.
_seed_eq(a::Number, b::Number, _) = isequal(a, b) || a == b
function _seed_eq(a::T, b::T, seen) where {T<:Union{Tuple,NamedTuple}}
    return all(map((x, y) -> _seed_eq(x, y, seen), a, b))
end
_seed_eq(a::Tangent, b::Tangent, seen) = _seed_eq(a.fields, b.fields, seen)
function _seed_eq(a::PossiblyUninitTangent, b::PossiblyUninitTangent, seen)
    return is_init(a) == is_init(b) && (!is_init(a) || _seed_eq(val(a), val(b), seen))
end
# Pairs already under comparison agree, so cyclic seeds terminate.
function _seed_eq(a::T, b::T, seen) where {T<:Union{MutableTangent,AbstractArray}}
    (a === b || haskey(seen, (a, b))) && return true
    seen[(a, b)] = nothing
    a isa MutableTangent && return _seed_eq(a.fields, b.fields, seen)
    return axes(a) == axes(b) && _seeds_agree(0, a, 0, b, seen)
end
@static if VERSION >= v"1.11-rc4"
    function _seed_eq(a::MemoryRef, b::MemoryRef, seen)
        Core.memoryrefoffset(a) == Core.memoryrefoffset(b) || return false
        return _seed_eq(a.mem, b.mem, seen)
    end
end
_seed_eq(a, b, _) = a === b || isequal(a, b) || a == b
@noinline _throw_conflicting_seeds() = throw(
    ArgumentError(
        "`lift` was given conflicting tangents for one storage, reached twice through " *
        "one object or through two headers over one buffer. Supply tangents that agree " *
        "wherever primals share storage.",
    ),
)

# One storage identity for primals and tangents: an `Array` and its backing `Memory` are
# distinct objects over one buffer, as are reshaped arrays.
@inline _storage_id(@nospecialize(x)) = x
@inline _storage_offset(@nospecialize(x)) = 0
@static if VERSION >= v"1.11-rc4"
    @inline function _storage_id(x::Array)
        owner = _storage_id(getfield(x, :ref).mem)
        # Only a zero-length owner cannot be shared: `empty!` can retain nonempty storage.
        return isempty(owner) ? x : owner
    end
    @inline function _storage_id(x::Memory)
        @static if VERSION < v"1.12-"
            owner = ccall(:jl_genericmemory_owner, Any, (Any,), x)
            return owner isa typeof(x) ? owner : x
        else
            return x
        end
    end
    @inline _storage_offset(x::Array) =
        _storage_offset(getfield(x, :ref).mem) + Core.memoryrefoffset(getfield(x, :ref)) - 1
    @inline function _storage_offset(x::Memory)
        iszero(Base.elsize(x)) && return 1
        return Int(UInt(pointer(x)) - UInt(pointer(_storage_id(x)))) ÷ Base.elsize(x) + 1
    end
    # A `Memory` header over a window of its owner (1.11 `reshape`) gets the same window of
    # the owner's tangent, which keeps ownership even for boxed elements.
    @inline function _tangent_memory_window(x::Memory, t::Memory{T}) where {T}
        msg =
            "Julia 1.11 cannot share derivative storage with part of a larger array " *
            "whose elements are a Union (for example, a reshape of a Vector that has " *
            "grown). Reshape a copy instead, or use Julia 1.12 or later."
        Base.isbitsunion(T) && length(x) != length(t) && throw(ArgumentError(msg))
        ref = memoryref(t, _storage_offset(x))
        return ccall(
            :jl_genericmemory_slice,
            Memory{T},
            (Any, Ptr{Cvoid}, Int),
            t,
            ref.ptr_or_offset,
            length(x),
        )
    end
else
    # `reshape` stores its root owner in the header (`how == 3`); `unsafe_wrap` stores none.
    @inline function _storage_id(x::Array)
        p = pointer_from_objref(x)
        flags = GC.@preserve x unsafe_load(Ptr{UInt16}(p + _JL_ARRAY_T_FLAGS_OFFSET))
        flags & 0x3 == 0x3 || return x
        offset = 4 * sizeof(Int) + 8 + sizeof(Int) * max(0, ndims(x) - 2)
        return GC.@preserve x unsafe_pointer_to_objref(
            unsafe_load(Ptr{Ptr{Cvoid}}(p + offset))
        )
    end
end

const _STRUCTURAL_CONTAINERS = @static if VERSION >= v"1.11-rc4"
    (Array, Memory)
else
    (Array,)
end

# Derived array growth branches on primal capacity and moves refs by primal offsets.
# Its element-wise V must therefore mirror the backing storage, not just size(x).
@inline _structural_dual_shell(x, ::Type{T}) where {T} = similar(x, T)
@static if VERSION >= v"1.11-rc4"
    @inline function _structural_dual_shell(x::Array{P,D}, ::Type{T}) where {P,D,T}
        ref = getfield(x, :ref)
        mem = Memory{T}(undef, length(ref.mem))
        return Base.wrap(Array, construct_ref(ref, mem), size(x))::Array{T,D}
    end
end

# Headers over one storage (`reshape`, an `Array` beside its `Memory`) must share one
# representation, or a write through one leaves the other's partials stale. Caches therefore
# key a structural container on its storage, holding the first header's representation there;
# any other header is built over that representation's storage and registered under itself.
@inline _same_window(v, x) = size(v) == size(x) && _storage_offset(v) == _storage_offset(x)
# Return the storage key and its cache entry (or `nothing`).
@inline function _extent_key(x, key, cache)
    cache isa NoCache && return key, nothing
    return haskey(cache, key) ? (key, cache[key]) : (key, nothing)
end
@static if VERSION >= v"1.11-rc4"
    @inline function _structural_key(x::Array{T}) where {T}
        mem = getfield(x, :ref).mem
        owner = _storage_id(mem)
        return isempty(owner) ? x : _structural_key(mem)
    end
    @inline function _structural_key(x::Memory)
        isempty(x) && return x
        @static if VERSION < v"1.12-"
            # Small owned Memory stores its data inline; it cannot have another owner.
            inline_data = GC.@preserve x pointer(x) == pointer_from_objref(x) + 16
            inline_data && return x
        end
        return _storage_id(x)::typeof(x)
    end
    # Caches hold `Any`; asserting the representation keeps the dispatch on it static.
    @inline function _owner_memory(first, ::Type{T}) where {T}
        m = (first isa Memory ? first : getfield(getfield(first, :ref), :mem))::Memory{T}
        return _storage_id(m)::Memory{T}
    end
    # The primal owner is already the storage key; register its header under the tangent owner.
    @inline _header_key(x, key, first, ::Type{T}) where {T} =
        x === key ? (key isa Array ? key : _owner_memory(first, T)) : x
    # `x`'s representation over `m`, the representation of its owning storage. A backing as
    # long as the owner covers all of it, so `m` itself serves.
    @inline function _structural_dual_shell(x::Array{P,D}, m::Memory{T}) where {P,D,T}
        ref = getfield(x, :ref)
        w = length(ref.mem) == length(m) ? m : _tangent_memory_window(ref.mem, m)
        return Base.wrap(Array, construct_ref(ref, w), size(x))::Array{T,D}
    end
    @inline _structural_dual_shell(x::Memory, m::Memory) =
        length(x) == length(m) ? m : _tangent_memory_window(x, m)
    @inline _new_shared_shell(x, ::Type{T}, key) where {T} = _structural_dual_shell(
        x, Memory{T}(undef, length(key))
    )
    @inline _shared_shell(x, first, ::Type{T}) where {T} = _structural_dual_shell(
        x, _owner_memory(first, T)
    )
    # A lane allocates only up to the end of its header's window, not the owner's capacity,
    # but at the header's offset, which locates later headers. One reaching past that end
    # gets storage of its own. An empty `Array`'s ref still needs a slot past offset 1.
    @inline _lane_end(p::Array) =
        _storage_offset(p) + max(length(p), _storage_offset(p) > 1) - 1
    @inline _lane_end(p::Memory) = _storage_offset(p) + length(p) - 1
    @inline _lane_over(p::Array, m) = Base.wrap(
        Array, _memoryref_at(m, _storage_offset(p)), size(p)
    )
    @inline _lane_over(p::Memory, m) = _structural_dual_shell(p, m)
    @inline _new_lane(p, ::Type{T}) where {T} = _lane_over(
        p, Memory{T}(undef, _lane_end(p))
    )
    @inline function _shared_lane(p, first, ::Type{T}) where {T}
        m = _owner_memory(first, T)
        _lane_end(p) <= length(m) || return _new_lane(p, T)
        return _lane_over(p, m)
    end
else
    const _JL_ARRAY_T_FLAGS_OFFSET = 2 * sizeof(Int)
    const _JL_ARRAY_T_FLAGS_ISSHARED = UInt16(0x4000)
    # A shared header keys on its owner, wrapped so the key never equals a header.
    @inline function _structural_key(x::Array)
        flags = GC.@preserve x unsafe_load(
            Ptr{UInt16}(pointer_from_objref(x) + _JL_ARRAY_T_FLAGS_OFFSET)
        )
        return iszero(flags & _JL_ARRAY_T_FLAGS_ISSHARED) ? x : (_storage_id(x),)
    end
    @inline _header_key(x, key, first, ::Type) = x === key ? key : x
    @inline function _new_shared_shell(x::Array{P,D}, ::Type{T}, key) where {P,D,T}
        key === x && return _structural_dual_shell(x, T)
        # A shared V must not own its buffer: Julia 1.10 cannot resize a shared owner.
        owner = Vector{T}(undef, length(x))
        return ccall(
            :jl_reshape_array, Array{T,D}, (Any, Any, Any), Array{T,D}, owner, size(x)
        )
    end
    # `reshape`'s `Array` case: the first header's unknown rank would make `reshape` dispatch
    # at runtime.
    @inline function _shared_shell(x::Array{P,D}, first, ::Type{T}) where {P,D,T}
        return ccall(
            :jl_reshape_array, Array{T,D}, (Any, Any, Any), Array{T,D}, first, size(x)
        )
    end
    @inline _new_lane(p::Array, ::Type{T}) where {T} = _new_shared_shell(
        p, T, _structural_key(p)
    )
    @inline _shared_lane(p, first, ::Type{T}) where {T} = _shared_shell(p, first, T)
end

# Register before filling, as `zero_tangent_internal(::Array)` does, so cycles terminate and
# repeated children share one V.
for C in _STRUCTURAL_CONTAINERS
    @eval @inline function _lift(
        ::StructuralStorage, x::$C, ẋ::$C, c::Union{Nothing,IdDict}
    )
        V = dual_type(Val(1), typeof(x))
        # `NoDual` elements hold nothing to lift, and bottom storage nothing to read.
        if ẋ isa $C{NoTangent} && eltype(V) === NoDual
            return Lifted{typeof(x),1,V}(
                x, fill!(_structural_dual_shell(x, NoDual), NoDual())
            )
        end
        d = c === nothing ? IdDict() : c
        key = _structural_key(x)
        _check_seed!(d, key, _storage_offset(x), ẋ)
        key, first = _extent_key(x, key, c === nothing ? NoCache() : d)
        # A fresh cache holds nothing; skipping its lookup also keeps it off the heap.
        if first === nothing
            hkey = key
            v = _new_shared_shell(x, eltype(V), key)::V
        else
            first = first::Lifted
            primal(first) === x && return first::Lifted{typeof(x),1,V}
            hkey = _header_key(x, key, tangent(first), eltype(V))
            haskey(d, hkey) && return d[hkey]::Lifted{typeof(x),1,V}
            v = _shared_shell(x, tangent(first), eltype(V))::V
        end
        lifted = Lifted{typeof(x),1,V}(x, v)
        d[hkey] = lifted
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = tangent(lift(x[i], ẋ[i], d)))
        end
        return lifted
    end
end
@inline lift(x::Tuple, ẋ::Tuple) = lift(x, ẋ, nothing)
@inline function lift(x::Tuple, ẋ::Tuple, c::Union{Nothing,IdDict})
    # Thread a shared cache through the elements so aliased mutable elements dedup to one V
    # (reverse aliasing invariant); upgrade a top-level `nothing` like the aggregate lifts above.
    d = c === nothing ? IdDict() : c
    field_Vs = map((xi, vi) -> tangent(lift(xi, vi, d)), x, ẋ)
    return Lifted{typeof(x),1}(x, field_Vs)
end
@inline lift(x::NamedTuple, ẋ::NamedTuple) = lift(x, ẋ, nothing)
@inline function lift(
    x::NamedTuple{names}, ẋ::NamedTuple, c::Union{Nothing,IdDict}
) where {names}
    d = c === nothing ? IdDict() : c
    field_Vs = map((xi, vi) -> tangent(lift(xi, vi, d)), values(x), values(ẋ))
    return Lifted{typeof(x),1}(x, NamedTuple{names}(field_Vs))
end

# A scalar float's uninit seed is the zero seed (a bits partial can't be read as garbage).
@inline uninit_dual(w::Val{N}, x::T) where {N,T<:IEEEFloat} = zero_dual(w, x)

@inline function randn_dual(::Val{N}, rng::AbstractRNG, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> randn(rng, T), Val(N)))
end

# Primitive seeds bypass the structural fallback.
@inline zero_dual(::Val{N}, ::NoDualPrimal) where {N} = NoDual()
@inline uninit_dual(::Val{N}, ::NoDualPrimal) where {N} = NoDual()
@inline randn_dual(::Val{N}, ::AbstractRNG, ::NoDualPrimal) where {N} = NoDual()

# Array seeds alias the user's primal and own a slot-local partials block.

@inline function _zero_dual(
    ::NumericStorage, ::Val{N}, x::A
) where {N,E<:FloatOrComplex,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x)
end
@inline function _uninit_dual(
    ::NumericStorage, ::Val{N}, x::A
) where {N,E<:FloatOrComplex,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x, Nfwd._allocate_block(Val(N), x))
end

# `Ref{P<:FloatOrComplex}` → `NDualRef` (scalar analogue of the `Array` factories above): fresh
# slot-local parallel partials. Zero and uninit coincide (the partials are bits scalars).
@inline function _zero_dual(
    ::NumericStorage, ::Val{N}, ::Base.RefValue{P}
) where {N,P<:FloatOrComplex}
    return NDualRef{P,N}()
end
@inline _uninit_dual(::NumericStorage, w::Val{N}, r::Base.RefValue{P}) where {N,P<:FloatOrComplex} = zero_dual(
    w, r
)

# A seed has no tangent buffer to point at: its lanes repeat a non-differentiable pointee's
# address, as `uninit_tangent` does, and are null for a differentiable one.
@inline function uninit_dual(::Val{N}, x::Ptr) where {N}
    p = _carries_partials(typeof(x)) ? oftype(x, C_NULL) : x
    return PointerDual{N}(ntuple(_ -> bitcast(Ptr{Cvoid}, p), Val(N)), NoDual)
end
@inline zero_dual(w::Val{N}, x::Ptr) where {N} = uninit_dual(w, x)
@inline randn_dual(w::Val{N}, ::AbstractRNG, x::Ptr) where {N} = uninit_dual(w, x)

@inline function _randn_dual(
    ::NumericStorage, ::Val{N}, rng::AbstractRNG, x::A
) where {N,E<:FloatOrComplex,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x, ntuple(_ -> randn(rng, E, size(x)), Val(N)))
end
@inline function _randn_dual(
    ::NumericStorage, ::Val{N}, rng::AbstractRNG, ::Base.RefValue{P}
) where {N,P<:FloatOrComplex}
    return NDualRef{P,N}(Base.RefValue{NTuple{N,P}}(ntuple(_ -> randn(rng, P), Val(N))))
end

# Element-wise array seeds skip undefined slots; numeric elements use NDualArray above.
# Keep dual_type in the caller's world so extension overloads remain visible.
for factory in (:zero_dual, :uninit_dual, :randn_dual)
    storage_factory = Symbol(:_, factory)
    rng_args = factory === :randn_dual ? (:(rng::AbstractRNG),) : ()
    rng_vals = factory === :randn_dual ? (:rng,) : ()
    containers = @static if VERSION >= v"1.11-rc4"
        (Array, Base.RefValue, Memory, MemoryRef)
    else
        (Array, Base.RefValue)
    end
    for C in containers
        @eval @inline function $factory(w::Val, $(rng_args...), x::$C{T}) where {T}
            return $storage_factory(_storage_kind(T), w, $(rng_vals...), x)
        end
    end
    for C in _STRUCTURAL_CONTAINERS
        @eval @inline function $storage_factory(
            ::StructuralStorage, w::Val{N}, $(rng_args...), x::$C{T}
        ) where {N,T}
            v = _structural_dual_shell(x, dual_type(w, T))
            @inbounds for i in eachindex(x)
                isassigned(x, i) && (v[i] = $factory(w, $(rng_vals...), x[i]))
            end
            return v
        end
    end
    # Gate tuple seeds on dual_type: even the empty tuple seeds to NoDual, while the
    # private _dual_tuple_v recursion base remains Tuple{}.
    @eval @inline function $factory(w::Val{N}, $(rng_args...), x::Tuple) where {N}
        dual_type(w, typeof(x)) === NoDual && return NoDual()
        return map(xi -> $factory(w, $(rng_vals...), xi), x)
    end

    # map preserves NamedTuple names.
    # All-non-differentiable NamedTuples seed to whole `NoDual`, matching their
    # `dual_type`; otherwise build element-wise.
    @eval @inline function $factory(w::Val{N}, $(rng_args...), x::NamedTuple) where {N}
        tangent_type(typeof(x)) === NoTangent && return NoDual()
        return map(xi -> $factory(w, $(rng_vals...), xi), x)
    end

    # Complex floats use Complex{NDual}, not the generic structural lift.
    @eval @inline function $factory(
        w::Val{N}, $(rng_args...), z::Complex{R}
    ) where {N,R<:IEEEFloat}
        return Complex{NDual{R,N}}(
            $factory(w, $(rng_vals...), real(z)), $factory(w, $(rng_vals...), imag(z))
        )
    end
end

# Structural seeds wrap per-field Vs according to mutability. Sub-function calls stay
# in the returned expression, with deferred errors for unsupported P.

@inline _zero_dual_zero_field(::Val{N}, x) where {N} = ntuple(_ -> zero_tangent(x), Val(N))
@inline _uninit_dual_zero_field(::Val{N}, x) where {N} = ntuple(
    _ -> uninit_tangent(x), Val(N)
)
@inline _randn_dual_zero_field(::Val{N}, rng, x) where {N} = ntuple(
    _ -> randn_tangent(rng, x), Val(N)
)

# A possibly-uninit field's seed is a `PossiblyUninitTangent` guarded by `isdefined`.
function _seed_field_expr(N, P, i, callexpr)
    always_initialised(P)[i] && return callexpr
    name = QuoteNode(fieldnames(P)[i])
    Vt = :(dual_type(Val($N), $(fieldtype(P, i))))
    return :(
        if isdefined(x, $name)
            PossiblyUninitTangent{$Vt}($callexpr)
        else
            PossiblyUninitTangent{$Vt}()
        end
    )
end

# Coerce into the declared backing NamedTuple, so `typeof(seed) === dual_type(Val(N), P)`.
for (f, helper) in (
    (:zero_dual, :_zero_dual_zero_field),
    (:uninit_dual, :_uninit_dual_zero_field),
    (:randn_dual, :_randn_dual_zero_field),
)
    rng_args = f === :randn_dual ? (:(rng::AbstractRNG),) : ()
    rng_vals = f === :randn_dual ? (:rng,) : ()
    storage_factory = Symbol(:_, f)
    @eval @inline $f(w::Val, $(rng_args...), x) = $storage_factory(
        StructuralStorage(), w, $(rng_vals...), x
    )
    @eval @generated function $storage_factory(
        ::StructuralStorage, ::Val{N}, $(rng_args...), x::P
    ) where {N,P}
        isconcretetype(P) || return :(error($("$($f): P=$P is not concrete")))
        # NoDual has no backing to seed. Resolve dual_type at the call world.
        if fieldcount(P) == 0
            return :(
                if dual_type(Val($N), typeof(x)) === NoDual
                    NoDual()
                else
                    $($(QuoteNode(helper)))(Val($N), $($(QuoteNode(rng_vals))...), x)
                end
            )
        end
        seeds = map(1:fieldcount(P)) do i
            nm = QuoteNode(fieldnames(P)[i])
            return _seed_field_expr(
                N,
                P,
                i,
                :($($f)(Val($N), $($(QuoteNode(rng_vals))...), getfield(x, $nm))),
            )
        end
        wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
        msg =
            "$($f): $P declares a `dual_type` that is not the structural lift, so this fallback " *
            "cannot seed it. A type with its own V needs BOTH seed entry points: define " *
            "`zero_dual`/`uninit_dual`/`randn_dual` for it alongside its `_*_dual_internal` " *
            "overloads."
        return quote
            V = dual_type(Val($N), typeof(x))
            V === NoDual && return NoDual()
            V <: $wrapper || error($msg)
            $wrapper(fieldtype(V, 1)(($(seeds...),)))
        end
    end
end

# Cache-aware seeds register mutable shells before recursion to preserve cycles and aliases.
# Custom Vs need BOTH cache-free and cache-threading factories; direct callers use the former.
@static if VERSION >= v"1.11-rc4"
    # Window the backing Memory's cached block so Array/Memory aliases share partials.
    @inline function _derived_array_dual(
        ::Val{N}, x::Array{E,D}, memv
    ) where {N,E<:FloatOrComplex,D}
        dims = (N, size(x)...)
        block = Nfwd._window_block(
            getfield(memv, :partials_block), Val(N), _storage_offset(x), dims
        )
        return NDualArray{E,N,D,typeof(x)}(x, block)
    end
end

# Julia 1.10 refuses to resize an array that `reshape` gave a second header; mark a
# shared block's parent the same way.
@inline function _share_block(b::PartialsBlock, dims)
    @static if VERSION < v"1.11-rc4"
        p = getfield(b, :parent)
        flags = GC.@preserve p unsafe_load(
            Ptr{UInt16}(pointer_from_objref(p) + _JL_ARRAY_T_FLAGS_OFFSET)
        )
        iszero(flags & _JL_ARRAY_T_FLAGS_ISSHARED) && reshape(p, (1, length(p)))
    end
    return reshape(b, dims)
end

# Julia 1.10 has no Memory: key on (_storage_id, length) and share the cached block's
# parent across reshapes. Every factory uses this; owned runs only on a cache miss.
@static if VERSION < v"1.11-rc4"
    @inline function _cached_array_dual(
        w::Val{N}, x::Array, d::MaybeCache, owned::F
    ) where {N,F}
        d isa NoCache && return owned()
        sk = (_storage_id(x), length(x))
        if haskey(d, sk)
            # The cached entry may have another shape; share its flat parent, asserted so the
            # constructor dispatches statically.
            shared = getfield(getfield(d[sk], :partials_block), :parent)::Vector{eltype(x)}
            return NDualArray{eltype(x),N,ndims(x),typeof(x)}(
                x, _share_block(Nfwd.PartialsBlock(shared, (N, length(x))), (N, size(x)...))
            )
        end
        v = owned()
        d[sk] = v
        return v
    end
end

for (factory, internal) in (
    (:zero_dual, :_zero_dual_internal),
    (:uninit_dual, :_uninit_dual_internal),
    (:randn_dual, :_randn_dual_internal),
)
    rng_args = factory === :randn_dual ? (:(rng::AbstractRNG),) : ()
    rng_vals = factory === :randn_dual ? (:rng,) : ()
    containers = @static if VERSION >= v"1.11-rc4"
        (Array, Base.RefValue, Memory, MemoryRef)
    else
        (Array, Base.RefValue)
    end
    for C in containers
        @eval @inline function $internal(
            w::Val, $(rng_args...), x::$C{T}, d::MaybeCache
        ) where {T}
            return $internal(_storage_kind(T), w, $(rng_vals...), x, d)
        end
    end
    @eval @inline $internal(w::Val, $(rng_args...), x, d::MaybeCache) = $internal(
        StructuralStorage(), w, $(rng_vals...), x, d
    )
    @eval begin
        @generated function $internal(
            ::StructuralStorage, w::Val{N}, $(rng_args...), x::P, d::MaybeCache
        ) where {N,P}
            # `fieldcount(P) == 0` is world-independent (gen-time); the
            # `dual_type(...) === NoDual` test goes in the returned expression, not the body.
            fieldcount(P) == 0 &&
                return :($$(QuoteNode(factory))(w, $($(QuoteNode(rng_vals))...), x))
            seeds = map(1:fieldcount(P)) do i
                nm = QuoteNode(fieldnames(P)[i])
                _seed_field_expr(
                    N,
                    P,
                    i,
                    :($$(QuoteNode(internal))(
                        w, $($(QuoteNode(rng_vals))...), getfield(x, $nm), d
                    )),
                )
            end
            if ismutabletype(P)
                return quote
                    V = dual_type(Val(N), P)
                    V === NoDual && return NoDual()
                    backing = fieldtype(V, 1)
                    haskey(d, x) && return d[x]::MutableDual{backing}
                    shell = MutableDual{backing}()
                    d[x] = shell
                    shell.fields = backing(($(seeds...),))
                    return shell
                end
            else
                return quote
                    V = dual_type(Val(N), P)
                    V === NoDual && return NoDual()
                    ImmutableDual(fieldtype(V, 1)(($(seeds...),)))
                end
            end
        end
        # Assert cache hits to the concrete V or IdDict{Any,Any} poisons inference.
        # Numeric elements cannot cycle; their blocks still share the backing Memory's V.
        function $internal(
            ::NumericStorage,
            w::Val{N},
            $(rng_args...),
            x::Array{<:FloatOrComplex},
            d::MaybeCache,
        ) where {N}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            # Derive the block from the backing `Memory`'s (see `_derived_array_dual`). Only with a
            # cache: without one there is no other V to share with, so an owned block is both
            # correct and cheaper.
            @static if VERSION >= v"1.11-rc4"
                v = if d isa NoCache
                    $factory(w, $(rng_vals...), x)
                else
                    _derived_array_dual(
                        w,
                        x,
                        $internal(w, $(rng_vals...), _storage_id(getfield(x, :ref).mem), d),
                    )
                end
            else
                v = _cached_array_dual(w, x, d, () -> $factory(w, $(rng_vals...), x))
            end
            d[x] = v
            return v
        end
        # `Ref{P<:FloatOrComplex}` → `NDualRef` (scalar analogue of the `Array` branch): build the
        # wrapper directly and register by identity, so the generic struct recursion never re-lifts it.
        function $internal(
            ::NumericStorage, w::Val{N}, $(rng_args...), x::Base.RefValue{P}, d::MaybeCache
        ) where {N,P<:FloatOrComplex}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            v = $factory(w, $(rng_vals...), x)
            d[x] = v
            return v
        end
        function $internal(w::Val{N}, $(rng_args...), x::Tuple, d::MaybeCache) where {N}
            dual_type(w, typeof(x)) === NoDual && return NoDual()
            return map(xi -> $internal(w, $(rng_vals...), xi, d), x)
        end
        function $internal(
            w::Val{N}, $(rng_args...), x::NamedTuple{names}, d::MaybeCache
        ) where {N,names}
            tangent_type(typeof(x)) === NoTangent && return NoDual()
            return NamedTuple{names}(
                map(xi -> $internal(w, $(rng_vals...), xi, d), values(x))
            )
        end
        # `Complex` has fields but a non-structural V, and holds no storage to share.
        $internal(w::Val{N}, $(rng_args...), z::Complex, ::MaybeCache) where {N} = $factory(
            w, $(rng_vals...), z
        )
    end
    # Register before filling: nested containers may alias or cycle. Thread the same
    # cache through children, unlike the deliberately cache-free factories, `Memory`
    # included: a `Dict`'s `vals` can hold one array in two elements.
    for C in _STRUCTURAL_CONTAINERS
        @eval @inline function $internal(
            ::StructuralStorage, w::Val{N}, $(rng_args...), x::$C, d::MaybeCache
        ) where {N}
            V = dual_type(Val(N), typeof(x))
            key = _structural_key(x)
            key, first = _extent_key(x, key, d)
            if first !== nothing
                # A seed has no primal field, so register each header explicitly. Distinct
                # headers over one window get distinct V wrappers over the shared partials.
                hkey = _header_key(x, key, first, eltype(V))
                haskey(d, hkey) && return d[hkey]::V
                shell = _shared_shell(x, first, eltype(V))::V
                d[hkey] = shell
            else
                shell = _new_shared_shell(x, eltype(V), key)::V
                d[key] = shell
                hkey = _header_key(x, key, shell, eltype(V))
                hkey === key || (d[hkey] = shell)
            end
            @inbounds for i in eachindex(x)
                isassigned(x, i) && (shell[i] = $internal(w, $(rng_vals...), x[i], d))
            end
            return shell
        end
    end
    @static if VERSION >= v"1.11-rc4"
        # With a cache, MemoryRef must share the backing Memory's partials. Leaf elements
        # window its block; aggregates reference its element-wise shell. NoCache owns a block.
        @eval function $internal(
            ::NumericStorage, w::Val{N}, $(rng_args...), x::MemoryRef{E}, d::MaybeCache
        ) where {N,E<:FloatOrComplex}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            v = if d isa NoCache
                $factory(w, $(rng_vals...), x)
            else
                NDualMemoryRef{E,N,Memory{E}}(
                    x,
                    getfield($internal(w, $(rng_vals...), x.mem, d), :partials_block),
                    Core.memoryrefoffset(x),
                )
            end
            d[x] = v
            return v
        end
        @eval function $internal(
            ::StructuralStorage, w::Val{N}, $(rng_args...), x::MemoryRef, d::MaybeCache
        ) where {N}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            v = if d isa NoCache
                $factory(w, $(rng_vals...), x)
            else
                _memoryref_at($internal(w, $(rng_vals...), x.mem, d), Core.memoryrefoffset(x))
            end
            d[x] = v
            return v
        end
        @eval function $internal(
            ::NumericStorage, w::Val{N}, $(rng_args...), x::Memory, d::MaybeCache
        ) where {N}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            owner = _storage_id(x)
            v = if owner === x || d isa NoCache
                $factory(w, $(rng_vals...), x)
            else
                ov = $internal(w, $(rng_vals...), owner, d)
                block = Nfwd._window_block(
                    getfield(ov, :partials_block), w, _storage_offset(x), (N, length(x))
                )
                NDualArray{eltype(x),N,1,typeof(x)}(x, block)
            end
            d[x] = v
            return v
        end
    end
end

for (factory, internal) in (
    (:zero_lifted, :_zero_dual_internal),
    (:uninit_lifted, :_uninit_dual_internal),
    (:randn_lifted, :_randn_dual_internal),
)
    rng_args = factory === :randn_lifted ? (:(rng::AbstractRNG),) : ()
    rng_vals = factory === :randn_lifted ? (:rng,) : ()
    @eval @inline function $factory(w::Val{N}, $(rng_args...), x::P) where {N,P}
        return Lifted{P,N}(
            x,
            $internal(w, $(rng_vals...), x, isbitstype(P) ? NoCache() : IdDict{Any,Any}()),
        )
    end
    @eval @doc """
            zero_lifted(Val(N), x)
            uninit_lifted(Val(N), x)
            randn_lifted(Val(N), rng, x)

        Return a width-`N` [`Lifted`](@ref) slot for `x` with zero, unspecified or
        `randn(rng)` partials. The primal aliases `x`; the partials are fresh. One cache
        spans the call, so within `x` one object reached twice, or two headers over one
        buffer, share partials.
        """ $factory
end

# ── Memory and MemoryRef seed factories (Julia 1.11+) ───────────────────────

@static if VERSION >= v"1.11-rc4"
    # Storage dispatch keeps numeric seeds separate from element-wise factories.
    # MemoryRef blocks cover the whole backing Memory, with the ref's column at its offset.
    @inline function _zero_dual(
        ::NumericStorage, ::Val{N}, p::MemoryRef{E}
    ) where {N,E<:FloatOrComplex}
        return NDualMemoryRef{E,N,Memory{E}}(p)
    end
    @inline function _uninit_dual(
        ::NumericStorage, ::Val{N}, p::MemoryRef{E}
    ) where {N,E<:FloatOrComplex}
        return NDualMemoryRef{E,N,Memory{E}}(
            p, PartialsBlock{E,2}(undef, N, length(p.mem)), Core.memoryrefoffset(p)
        )
    end
    @inline function _randn_dual(
        ::NumericStorage, ::Val{N}, rng::AbstractRNG, p::MemoryRef{E}
    ) where {N,E<:FloatOrComplex}
        len = length(p.mem)
        block = PartialsBlock{E,2}(randn(rng, E, N * len), (N, len))
        return NDualMemoryRef{E,N,Memory{E}}(p, block, Core.memoryrefoffset(p))
    end
    @inline function _zero_dual(
        ::NumericStorage, ::Val{N}, m::Memory{E}
    ) where {N,E<:FloatOrComplex}
        return NDualArray{E,N,1,Memory{E}}(m)
    end
    @inline function _uninit_dual(
        ::NumericStorage, ::Val{N}, m::Memory{E}
    ) where {N,E<:FloatOrComplex}
        return NDualArray{E,N,1,Memory{E}}(m, PartialsBlock{E,2}(undef, N, length(m)))
    end
    @inline function _randn_dual(
        ::NumericStorage, ::Val{N}, rng::AbstractRNG, m::Memory{E}
    ) where {N,E<:FloatOrComplex}
        return NDualArray{E,N,1,Memory{E}}(
            m, ntuple(_ -> Memory{E}(randn(rng, E, length(m))), Val(N))
        )
    end
    @inline function _zero_dual(::StructuralStorage, ::Val{N}, p::MemoryRef{T}) where {N,T}
        return _memoryref_at(zero_dual(Val(N), p.mem), Core.memoryrefoffset(p))
    end
    # `MemoryRef`'s V is built via `memoryref` over the `.mem`'s V (a plain
    # `MemoryRef` can't be constructed field-wise from a raw `Ptr`), so mirror
    # `zero_dual` rather than fall through to the generic struct seed.
    @inline function _uninit_dual(
        ::StructuralStorage, ::Val{N}, p::MemoryRef{T}
    ) where {N,T}
        return _memoryref_at(uninit_dual(Val(N), p.mem), Core.memoryrefoffset(p))
    end
    @inline function _randn_dual(
        ::StructuralStorage, ::Val{N}, rng::AbstractRNG, p::MemoryRef{T}
    ) where {N,T}
        return _memoryref_at(randn_dual(Val(N), rng, p.mem), Core.memoryrefoffset(p))
    end
end
