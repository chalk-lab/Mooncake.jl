# Loaded after nfwd/Nfwd.jl for the NDual carriers. Forward V types encode differentiability,
# wrappers and layout, so frules may need separate methods for shapes one rrule handles.
# When porting a reverse rule, preserve its conventions as well as its formula.

"""
    ImmutableDual{T<:NamedTuple}

Single-field immutable wrapper used as the canonical V for *immutable struct*
primals under the forward-mode structural lift. Its `fields::T` holds
the recursive `NamedTuple{fieldnames(P), Tuple{V_i...}}` of canonical field
Vs, where each `V_i = dual_type(Val(N), fieldtype(P, i))`.
"""
struct ImmutableDual{T<:NamedTuple}
    fields::T
end

Base.:(==)(x::ImmutableDual, y::ImmutableDual) = x.fields == y.fields

"""
    MutableDual{T<:NamedTuple}

Mutable counterpart to `ImmutableDual`. Must be mutable: the `MutableDualTangentView`
proxy writes back to `fields` via `setfield!`.
"""
mutable struct MutableDual{T<:NamedTuple}
    fields::T
    # Uninitialised form, used by the cyclic-struct `lift` to register a shell
    # in the aliasing cache before its fields (which may reference back to it)
    # are built. Mirrors reverse-mode `MutableTangent()`.
    MutableDual{T}() where {T<:NamedTuple} = new{T}()
    MutableDual{T}(fields) where {T<:NamedTuple} = new{T}(fields)
end
@inline MutableDual(fields::T) where {T<:NamedTuple} = MutableDual{T}(fields)

Base.:(==)(x::MutableDual, y::MutableDual) = x.fields == y.fields

"""
    Lifted{P, N, V}

Forward-mode slot wrapper for a primal value of type `P` and its canonical
`N`-width forward representation `V`. Two fields:

- `primal::P` — slot-level back-reference to the user's primal value. For
  mutable struct primals this aliases user storage; for immutable primals
  it carries the same value. Where `rep` names the primal itself
  (`NDualArray`, `NDualMemoryRef`) the constructor takes it from there, so the
  two names cannot disagree.
- `rep::V` — the canonical `N`-width forward representation. For
  concrete runtime wrappers `V === dual_type(Val(N), P)`.

Rules dispatch on `Lifted{P, N}` (V left abstract) and use `primal`,
`tangent`, and per-lane extractors to access tangent data. Outputs are
constructed via `Lifted{P_out, N}(primal_out, rep_out)`.

Width `N == 1` is ordinary forward mode; `N >= 2` is chunked forward
mode. `Lifted` never nests inside another `Lifted`'s `V`.
"""
struct Lifted{P,N,V}
    primal::P
    rep::V
    # Inner, so every construction path routes through `_slot_primal`.
    function Lifted{P,N,V}(primal, rep) where {P,N,V}
        return new{P,N,V}(_slot_primal(primal, rep), rep)
    end
end

# NDualArray / NDualMemoryRef already name their primal storage; use that identity.
# Wrapper primals (SubArray, Diagonal, …) cannot be rebuilt from their field Vs because
# metadata such as indices lifts to NoDual, so mismatched wrapper slots remain constructible.
# Type-valued primals (e.g. Type{Union{...}}) need not infer a concrete return type.
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

# Sharpen `P` when constructing from a `Type{X}` primal: the type-constructor frules (`_new_`,
# `Complex`, `TwicePrecision`) dispatch on the type slot, and `Lifted` is invariant, so a type-valued
# primal left in the wide `Lifted{DataType,N}` slot carries the value but misses the rule. Centralized
# because the interpreter and rule bodies build such slots in many places, not just the seed factories.
# Two params because neither alone serves: `P_wide` is the slot the caller type-applied (broad, since
# callers compute `typeof(x)`) and is ignored by the body; `P_sharp` is the identity dispatch binds.
@inline function Lifted{P_wide,N}(primal::Type{P_sharp}, rep::V) where {P_wide,P_sharp,N,V}
    # Fall back to the broad `typeof(primal)` when `P_sharp` can't bind — a phantom `TypeVar` (e.g. an
    # over-sharpened `UnionAll`), where touching `Type{P_sharp}` throws `UndefVarError`. Identical to the
    # `CoDual(::Type{P})` ctor's `@isdefined(P) ? Type{P} : typeof(x)` fallback (chalk-lab/Mooncake.jl#1191).
    return Lifted{(@isdefined(P_sharp) ? Type{P_sharp} : typeof(primal)),N,V}(primal, rep)
end

# Accessors — mirror the `CoDual` API.
primal(d::Lifted) = d.primal
tangent(d::Lifted) = d.rep
# `_primal` extracts a primal value from a forward-mode slot; the generic fallback returns
# the value unchanged. The interpreter's dual IR calls `_primal` on each operand, which is
# either a `Lifted` slot or a lifted constant.
_primal(x) = x
_primal(x::Lifted) = primal(x)

# Forward-mode slot-type check, used by the test framework and by the `NoDual` lane accessor below:
# a well-formed `Lifted{P,N,V}` slot has `V === dual_type(Val(N), P)` for concrete `P` (the
# coherence invariant). Abstract-`P` slots are sharpened to a concrete subtype at runtime, so the
# static V cannot be asserted — accept those.
# `Ptr` primals are exempt too: they have no ownable derivative storage, so a non-differentiable
# `Ptr` result legitimately carries `NoDual` rather than the per-lane `NTuple{N,Ptr}` (mirrors the
# `Ptr` exemption in `DebugFRule`'s `verify_canonical_dual_type`; see the `pointerref`/`cglobal` rules).
function verify_lifted_type(::Lifted{P,N,V}) where {P,N,V}
    (!isconcretetype(P) || P <: Ptr) || V === dual_type(Val(N), P)
end

"""
    extract(d::Lifted) -> (primal, value)

Helper that returns the `(primal(d), tangent(d))` tuple. Mirrors
`extract(::CoDual)` for symmetric ergonomics.
"""
extract(d::Lifted) = (primal(d), tangent(d))

# Lifted slots are safe to share by reference within a forward pass —
# slot-local tangent storage rules out cross-slot aliasing hazards. This is the only copy the
# pipeline takes: a `Base.copy(::Lifted)` cannot be right in general, because `Base.copy` is
# undefined or wrong for several supported primals and for every aggregate V (copying an
# `NDualArray` element-wise yields a `Vector{NDual}`, a shape `dual_type` never returns).
_copy(d::Lifted) = d

function Base.:(==)(a::Lifted, b::Lifted)
    return primal(a) == primal(b) && tangent(a) == tangent(b)
end

# Whole-array accessors — O(1) by aliasing.
@inline primal(a::NDualArray) = a.primal
@inline tangent(a::NDualArray) = Nfwd._lane_views(a)
@inline unpack_ndual(a::NDualArray) = (a.primal, Nfwd._lane_views(a))

# Per-lane accessor: lane `lane`'s derivative MATERIALISED as a reverse tangent of type
# `tangent_type(P)`, for every V shape. Writable per-lane access is `tangent_view(slot, lane)`.
# Leaf V accessors key on the V shape and leave `P` free: the inner V uniquely
# determines extraction, and an abstract slot (e.g. `Lifted{Real, N, NDual{Float64, N}}`,
# where the static primal type is abstract but the runtime V is concrete) must
# still resolve here.
@inline function tangent(x::Lifted{P,N,NDual{T,N}}, lane::Integer) where {P,T<:IEEEFloat,N}
    return tangent(x).partials[lane]
end
# tangent(x, lane) materialises a dense tangent_type(P) copy for reverse-shaped field
# conversion. tangent_view(x, lane) is a write-through stride-N view of the same partials.
@inline function tangent(x::Lifted{P,N,<:NDualArray}, lane::Integer) where {P,N}
    v = tangent(x)
    return copyto!(similar(getfield(v, :primal)), tangent_view(v, lane))
end
@inline Nfwd.tangent_view(x::Lifted{P,N,<:NDualArray}, lane::Integer) where {P,N} = Nfwd.tangent_view(
    tangent(x), lane
)
# A `Ref` is a mutable struct, so its reverse tangent is a
# `MutableTangent{@NamedTuple{x::PossiblyUninitTangent{P}}}`, NOT a bare scalar.
@inline tangent(x::Lifted{<:Base.RefValue{P},N,<:NDualRef}, lane::Integer) where {P<:NDualEltype,N} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)
# `NDualMemoryRef` / `MemoryRef` / `Core.memoryref` are 1.11+; gate to avoid an
# `UndefVarError` at precompile on 1.10.
@static if VERSION >= v"1.11-rc4"
    @inline function tangent(x::Lifted{P,N,<:NDualMemoryRef}, lane::Integer) where {P,N}
        # Materialize the lane into the reverse tangent shape (`tangent_type(MemoryRef{T})` is
        # `MemoryRef{T}` into a same-length `Memory` at the same offset). Copy semantics, like the
        # `NDualArray` lane accessor above: the block has no ownable per-lane memory to alias.
        # Block column `j` pairs with mem slot `o - c + j`; slots the block does not cover
        # (e.g. an `Array`-projected ref over a longer-capacity `Memory`) stay zero.
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
    # Element-wise `MemoryRef` V (a plain `MemoryRef` into an element-wise V `Memory`, e.g.
    # `MemoryRef{NoDual}` for `MemoryRef{Int}`, or `MemoryRef{NDualArray}` for `MemoryRef{Vector}`
    # comms). Project the lane through the `.mem`'s element-wise array lane accessor, then
    # re-`memoryref` at the offset.
    @inline function tangent(
        x::Lifted{P,N,V}, lane::Integer
    ) where {P<:MemoryRef,N,V<:MemoryRef}
        p = primal(x)
        lane_mem = tangent(Lifted{Memory{eltype(P)},N}(p.mem, tangent(x).mem), lane)
        return _memoryref_at(lane_mem, Core.memoryrefoffset(p))
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
            "a slot over the primal `$P` carries the forward value `NoDual`, but its canonical " *
            "forward value is `$(dual_type(Val(N), P))` and its reverse tangent " *
            "`$(tangent_type(P))`. `NoDual` is valid only where `dual_type` declares it.",
        ),
    )
end
# Rebuild from the primal: Vector{Ptr{Int}} has NoDual elements in forward but needs
# Ptr{NoTangent} addresses in reverse. Mapping NoDual to NoTangent would lose them.
@inline function tangent(x::Lifted{P,N,<:AbstractArray{NoDual}}, ::Integer) where {P,N}
    return uninit_tangent(primal(x))
end
# General element-wise container V — a plain `Array`/`Memory` of per-element forward Vs (e.g.
# `Array{NDualArray}` from a nested array, or `Memory{…}` forward-over-reverse comms).
# Project lane `k` element-wise to the same-shape array of each element's lane-`k`
# tangent, mirroring reverse `tangent_type(Array{T}) === Array{tangent_type(T)}`. The
# `NDualArray` (also `<:AbstractArray`) and the `AbstractArray{NoDual}` case have more
# specific overloads above; undefined slots stay undefined (reverse-PUT semantics).
@inline tangent(x::Lifted{P,N,V}, lane::Integer) where {P,N,V<:AbstractArray} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)
@inline function tangent(x::Lifted{P,N,<:Tuple}, lane::Integer) where {P,N}
    return tangent(x)[lane]
end
# `Ptr{Nothing}`'s reverse tangent is a `VoidPtrTangent`, not the raw address the per-lane V holds,
# so it is the second member of the `NoDual` family above: rebuild from the primal.
@inline tangent(x::Lifted{Ptr{Nothing},N,<:Tuple}, ::Integer) where {N} = uninit_tangent(
    primal(x)
)
@inline tangent(x::Lifted{P,N,<:ImmutableDual}, lane::Integer) where {P,N} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)
@inline tangent(x::Lifted{P,N,<:Tuple}, lane::Integer) where {P<:Tuple,N} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)
@inline tangent(x::Lifted{P,N,<:NamedTuple}, lane::Integer) where {P<:NamedTuple,N} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)

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
# A leaf mints its tangent fresh, so registering it is what keeps two aliased leaves under one slot
# on one reverse tangent — the aggregate methods register their shells, and without this the leaves
# they recurse into do not. An immutable primal has no identity to share. The value is built before
# the lookup so the hit has a concrete type to assert: unasserted, `cache[p]` out of the
# `IdDict{Any,Any}` widens a `Tuple{Vector{Float64},Vector{Float64}}` lane read to `Tuple{Any,Any}`.
@inline function _register_leaf(p, t, cache::IdDict)
    ismutable(p) || return t
    haskey(cache, p) && return cache[p]::typeof(t)
    cache[p] = t
    return t
end
@noinline function _throw_not_a_leaf_v(::Type{P}, t) where {P}
    throw(
        ArgumentError(
            "there is no method materialising a reverse tangent from the forward value of `$P`, " *
            "so it fell back to the lane accessor, which gave a `$(typeof(t))` where the reverse " *
            "tangent is `$(tangent_type(P))`. That fallback is right only for a leaf; an " *
            "aggregate needs its own `_materialise_lane` rebuilding a reverse tangent from its " *
            "components.",
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
# `P<:Tuple` only: a Tuple V is element-wise for a tuple primal, but per-LANE for `Ptr` and
# `TwicePrecision`, whose V is `NTuple{N,·}` of parallel copies of one leaf. Those take the
# generic terminal above; recursing element-wise here would index the leaf primal by lane and ask
# for `fieldtype(Ptr{Float64}, 1)`.
# Both container methods annotate the child with `typeof` of the held value, not the declared
# `fieldtype(P, ·)`, for the reason given at `_materialise_field_lane`.
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
# A block-backed array IS a leaf: it cannot reference back into itself, so it needs no shell — but
# it does need registering, being storage two aliased positions must share.
function _materialise_lane(
    x::Lifted{P,N,<:NDualArray}, lane::Integer, cache::IdDict
) where {P,N}
    _register_leaf(primal(x), tangent(x, lane), cache)
end
@static if VERSION >= v"1.11-rc4"
    # ... except that reverse derives an `Array`'s tangent from its backing `Memory`'s
    # (`zero_tangent_internal(::Array)` recurses through `.ref.mem`), so a primal holding both
    # comes back over ONE buffer. Mirror that, or the pair round-trips through `unlift` into two
    # independent tangents and a later accumulation lands in two buffers where the primal has one.
    # The seed path already ties them this way (`_derived_array_dual`).
    function _materialise_lane(
        x::Lifted{P,N,<:NDualArray}, lane::Integer, cache::IdDict
    ) where {P<:Array,N}
        p = primal(x)
        haskey(cache, p) && return cache[p]::tangent_type(P)
        E = eltype(P)
        r = getfield(p, :ref)
        mem = r.mem
        buf = getfield(getfield(getfield(tangent(x), :partials_block), :parent), :ref).mem
        # A V holding a block of its OWN rather than a window into the `Memory`'s — a rule's
        # freshly allocated array — shares no partial storage with it, so there is none to tie.
        # `_window_block` makes a window's parent the `Memory` block's storage, scaled by `N`,
        # which is what the length reads.
        len = N * length(mem)
        length(buf) == len || return _register_leaf(p, tangent(x, lane), cache)
        block = Nfwd.NDualBlock{E,2}(
            Base.wrap(Array, memoryref(buf), (len,))::Vector{E}, (N, length(mem))
        )
        memv = NDualArray{E,N,1,Memory{E}}(mem, block)
        tmem = _materialise_lane(Lifted{Memory{E},N}(mem, memv), lane, cache)
        t = Base.wrap(Array, construct_ref(r, tmem), size(p))::tangent_type(P)
        cache[p] = t
        return t
    end

    # The `MemoryRef` half of the same mirror: reverse's tangent is `construct_ref` over the
    # `Memory`'s. The lane accessor already materialises a full-length `Memory`, so register that
    # one and take the ref into it — but only where the V's block spans the whole `Memory`, which
    # is what makes the partials it carries the `Memory`'s own rather than a projected array's.
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
# An all-`NoDual` V is a leaf only when the ELEMENT's reverse tangent is `NoTangent` too. A `Ptr`
# to a non-differentiable element breaks that: `dual_type(Vector{Ptr{Int}})` is `Vector{NoDual}`
# while `tangent_type` is `Vector{Ptr{NoTangent}}`, so the accessor returns the wrong shape and
# only the element-wise path rebuilds it (through the scalar `Ptr`/`NoDual` method above).
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
    haskey(cache, p) && return cache[p]
    t = similar(p, tangent_type(eltype(P)))
    cache[p] = t
    # Build each element's slot from the CONCRETE `typeof(pe)`, not the static `eltype(P)`: for an
    # abstract-eltype array (e.g. `Vector{Distribution}` holding `Normal`s) the abstract type has
    # no fields, so `eltype(P)` would make the struct-lift do `fieldtype(Distribution, :μ)` and
    # throw. Undefined slots stay undefined (reverse-PUT semantics).
    _map_if_assigned!(
        (pe, ve) -> _materialise_lane(Lifted{typeof(pe),N}(pe, ve), lane, cache),
        t,
        p,
        tangent(x),
    )
    return t
end
# `Ref{P<:NDualEltype}` (V `NDualRef`): build the reverse `MutableTangent` it would have in reverse
# mode (a `Ref` is a mutable struct, registered in the alias cache before returning). Its `:x` field
# is non-always-init, so reverse wraps it in `PossiblyUninitTangent`.
function _materialise_lane(
    x::Lifted{<:Base.RefValue{P},N,<:NDualRef}, lane::Integer, cache::IdDict
) where {P<:NDualEltype,N}
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

# Raw forward Vs reach tangent arithmetic through test_frule_reuse.
_dot_internal(::MaybeCache, ::NoDual, ::NoDual) = 0.0
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:ImmutableDual}
    return _dot_internal(c, t.fields, s.fields)::Float64
end
# A `MutableDual` may be self-referential; cache the pair to break the cycle
# (mirrors reverse-mode `_dot_internal(::MaybeCache, ::MutableTangent, …)`).
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MutableDual}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return _dot_internal(c, t.fields, s.fields)::Float64
end
# Scalar NDual (forward-mode width-1 V for IEEEFloat) — sum the partials' dot.
function _dot_internal(::MaybeCache, t::NDual{T,N}, s::NDual{T,N}) where {T<:IEEEFloat,N}
    return Float64(sum(map(*, t.partials, s.partials); init=zero(T)))
end

_scale_internal(::MaybeCache, ::Float64, ::NoDual) = NoDual()
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:ImmutableDual}
    return T(_scale_internal(c, a, t.fields))
end
# Register an uninitialised result before recursing so a self-referential
# `MutableDual` terminates (mirrors reverse-mode `_scale_internal` for `MutableTangent`).
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MutableDual}
    haskey(c, t) && return c[t]::T
    y = T()
    c[t] = y
    y.fields = _scale_internal(c, a, t.fields)
    return y
end
# Scalar NDual scale — scale each lane only; `.value` is the primal it shadows (inner-value
# invariant) and stays.
function _scale_internal(::MaybeCache, a::Float64, t::NDual{T,N}) where {T<:IEEEFloat,N}
    aT = T(a)
    return NDual{T,N}(t.value, map(p -> aT * p, t.partials))
end

_add_to_primal_internal(::MaybeCache, x, ::NoDual, ::Bool) = x
# Scalar NDual: the perturbation content is the partials; `.value` is the primal the NDual shadows
# (inner-value invariant), which is already `x`, so adding it would double-count. Add only the
# partials so a zero-partials V is the identity: `_add_to_primal(x, NDual(x, zeros)) == x`.
function _add_to_primal_internal(
    ::MaybeCache, x::T, t::NDual{T,N}, ::Bool
) where {T<:IEEEFloat,N}
    return x + sum(t.partials; init=zero(T))
end
@unstable function _add_to_primal_internal(
    c::MaybeCache, x::P, t::ImmutableDual, unsafe::Bool
) where {P}
    # Match reverse Tangent reconstruction: unwrap PUTs, preserve undefined fields, and use
    # __construct_type to honour unsafe, inner constructors and AddToPrimalException.
    nt = t.fields
    isempty(propertynames(nt)) && return x
    fields = map(fieldnames(P)) do name
        tf = getfield(nt, name)
        isdefined(x, name) &&
            is_init(tf) &&
            return _add_to_primal_internal(c, getfield(x, name), val(tf), unsafe)
        !isdefined(x, name) && !is_init(tf) && return FieldUndefined()
        throw(error("unable to handle undefined-ness"))
    end
    return __construct_type(P, unsafe, fields...)::P
end
# Mutable structs may be self-referential, so use the two-pass scheme of the
# reverse-mode `MutableTangent` overload: const fields (which cannot cycle back)
# are perturbed up front, the placeholder is registered, then non-const fields are
# perturbed in place so a cycle resolves to the registered result.
function _add_to_primal_internal(
    c::MaybeCache, x::P, t::MutableDual, unsafe::Bool
) where {P}
    key = (x, t, unsafe)
    haskey(c, key) && return c[key]::P
    nt = t.fields
    # Mirror reverse-mode `_add_to_primal_internal(::MutableTangent)`: unwrap
    # `PossiblyUninitTangent` fields via `is_init`/`val`, map undefined fields to
    # `FieldUndefined()`, and build through `__construct_type` so `unsafe` is honoured.
    init_fields = map(fieldnames(P)) do name
        tf = getfield(nt, name)
        if isdefined(x, name) && is_init(tf) && isconst(P, name)
            return _add_to_primal_internal(c, getfield(x, name), val(tf), unsafe)
        elseif isdefined(x, name) && is_init(tf) && !isconst(P, name)
            return getfield(x, name)
        elseif !isdefined(x, name) && !is_init(tf)
            return FieldUndefined()
        else
            throw(error("unable to handle undefined-ness"))
        end
    end
    p′ = __construct_type(P, unsafe, init_fields...)::P
    c[key] = p′
    for name in fieldnames(P)
        tf = getfield(nt, name)
        isdefined(x, name) &&
            is_init(tf) &&
            !isconst(P, name) &&
            setfield!(
                p′, name, _add_to_primal_internal(c, getfield(x, name), val(tf), unsafe)
            )
    end
    return p′
end
# `NDualMemoryRef` (and its constructor) lives in `src/nfwd/Nfwd.jl`.
# Mooncake-namespace method extensions follow.

@static if VERSION >= v"1.11-rc4"
    @inline primal(a::NDualMemoryRef) = a.primal
    # Bulk/interface access reconstructs the shared (N, ncols) block from its backing ref.
    @inline tangent(a::NDualMemoryRef) = Nfwd._reconstruct_block(a)
    @inline unpack_ndual(a::NDualMemoryRef) = (a.primal, Nfwd._reconstruct_block(a))
end

# ──────────────────────────────────────────────────────────────────────────
# `MutableDualTangentView{SD, P}` — an immutable per-lane proxy over a mutable struct slot,
# returned by `tangent_view(slot, lane)`, so `view.field = x` from a rule body writes the lane
# back into the parent `MutableDual`.
#
# Reads are total over the shapes `dual_type` produces; writes cover the leaf shapes reverse
# `set_tangent_field!` accepts (scalar, complex, array, tuple, named-tuple and non-differentiable
# fields). A nested `MutableDual` has no writable lane — its would-be V is another view — and
# raises a clear error naming both the field's V and the value's type.
# ──────────────────────────────────────────────────────────────────────────

# Internal fields are underscore-prefixed so they cannot collide with a user struct field literally
# named `parent`/`primal`/`lane` (plausible for graph/tree nodes). `getproperty` never resolves to
# them: every internal read goes through `getfield`, so `v.name` means the user's field for EVERY
# name, matching `setproperty!` — a short-circuit on the underscored names would make a field
# genuinely called `_parent` readable as the view's parent but writable as itself.
struct MutableDualTangentView{N,SD<:MutableDual,P}
    _parent::SD
    _primal::P
    _lane::Int
end

# Lane-extraction (read) and lane-replacement (write) for individual V_i shapes. A read has a
# bespoke method only where the field owns storage the write has to reach; everything else
# materialises.
@inline _lane_tangent(::Val, ::Type, _p, _name, v::NDual, lane::Int) = v.partials[lane]

@inline function _replace_lane_tangent(v::NDual{T,N}, lane::Int, x::T) where {T,N}
    new_partials = ntuple(k -> k == lane ? x : v.partials[k], Val(N))
    return NDual{T,N}(v.value, new_partials)
end

# An array field reads as `Nfwd.tangent_view`, the WRITE-THROUGH lane view, not as
# `tangent(::Lifted, lane)`: that returns a dense reverse-shaped copy, so `view.field[i] = x` in a
# rule body would silently update nothing. The view already addresses the block, so a write is a
# `copyto!` and the V_i object is returned unchanged.
@inline _lane_tangent(::Val, ::Type, _p, _name, v::Nfwd.NDualArray, lane::Int) = Nfwd.tangent_view(
    v, lane
)

@inline function _replace_lane_tangent(v::Nfwd.NDualArray, lane::Int, x)
    copyto!(Nfwd.tangent_view(v, lane), x)
    return v
end

@inline _lane_tangent(::Val, ::Type, _p, _name, v::Complex{<:NDual}, lane::Int) = complex(
    v.re.partials[lane], v.im.partials[lane]
)

@inline function _replace_lane_tangent(
    v::Complex{NDual{T,N}}, lane::Int, x::Complex
) where {T,N}
    return complex(
        _replace_lane_tangent(v.re, lane, T(real(x))),
        _replace_lane_tangent(v.im, lane, T(imag(x))),
    )
end

# A nested mutable field reads as a nested WRITE-THROUGH view, for the same reason an array
# field does: it owns storage inside the parent block, so `view.next.w = x` from a rule body has
# to land there rather than in a throwaway copy.
@inline _lane_tangent(::Val{N}, ::Type{P}, p, name, v::MutableDual, lane::Int) where {N,P} =
    let pf = getfield(p, name)
        Nfwd.tangent_view(Lifted{typeof(pf),N}(pf, v), lane)
    end

# A non-always-initialised field's V is `PossiblyUninitTangent`-wrapped, as reverse's backing is;
# read and write through the wrapper as `get_tangent_field`/`set_tangent_field!` do. `val` throws
# on an uninitialised one: the primal field is undefined, so no lane has a value to read or shadow.
@inline _lane_tangent(w::Val{N}, ::Type{P}, p, name, v::PossiblyUninitTangent, lane::Int) where {N,P} = _lane_tangent(
    w, P, p, name, val(v), lane
)
@inline _replace_lane_tangent(v::T, lane::Int, x) where {T<:PossiblyUninitTangent} = T(
    _replace_lane_tangent(val(v), lane, x)
)

# A field with no storage of its own reads as its materialised reverse tangent, which is total
# over what `dual_type` produces and matches what reverse gives for the same field.
@inline _lane_tangent(::Val{N}, ::Type{P}, p, name, v, lane::Int) where {N,P} = _materialise_field_lane(
    Val(N), P, p, name, v, lane, IdDict{Any,Any}()
)

# A non-differentiable field has nothing to write, exactly as reverse `set_tangent_field!` accepts
# `NoTangent()` there. A tuple / named-tuple of leaf Vs recurses, so the write covers the shapes
# the read returns.
@inline _replace_lane_tangent(v::NoDual, ::Int, ::NoTangent) = v
@inline _replace_lane_tangent(v::Tuple, lane::Int, x::Tuple) = map(
    (vi, xi) -> _replace_lane_tangent(vi, lane, xi), v, x
)
@inline _replace_lane_tangent(v::NamedTuple{names}, lane::Int, x::NamedTuple{names}) where {names} = NamedTuple{
    names
}(
    map((vi, xi) -> _replace_lane_tangent(vi, lane, xi), values(v), values(x))
)

# The message names the VALUE type too: the common miss is a supported scalar V written with a
# number of another type (the repo does not promote implicitly), which is not a V-shape problem.
@inline function _replace_lane_tangent(v, lane::Int, @nospecialize(x))
    msg =
        "Cannot write a `$(typeof(x))` into lane $lane of a mutable struct field whose forward " *
        "V is $(typeof(v)): no lane write is defined for that combination."
    throw(ArgumentError(msg))
end

function Base.getproperty(v::MutableDualTangentView{N,SD,P}, name::Symbol) where {N,SD,P}
    nt = getfield(v, :_parent).fields
    return _lane_tangent(
        Val(N), P, getfield(v, :_primal), name, getfield(nt, name), getfield(v, :_lane)
    )
end

function Base.setproperty!(v::MutableDualTangentView, name::Symbol, x)
    parent = getfield(v, :_parent)
    lane = getfield(v, :_lane)
    nt = parent.fields
    new_V_i = _replace_lane_tangent(getfield(nt, name), lane, x)
    # `convert` to the stored NamedTuple type: `setfield!` is strict (no implicit convert) and
    # `NamedTuple` is invariant in its `Tuple` parameter, so for a mutable struct with an abstract
    # field (e.g. `x::Real` -> dual field NamedTuple `@NamedTuple{x}`, x::Any) the `merge` narrows to
    # `@NamedTuple{x::NDual}`, which is NOT `isa @NamedTuple{x}` — a bare `setfield!` throws. Mirrors
    # the writeback in `_setfield_tangent!(::MutableDual)`.
    setfield!(
        parent, :fields, convert(typeof(nt), merge(nt, NamedTuple{(name,)}((new_V_i,))))
    )
    return x
end

# Writable per-lane access on a `Lifted{MutS, N, <:MutableDual}` slot. `tangent(d, lane)`
# materialises a reverse `MutableTangent` instead, as it does for every other V shape — a proxy
# there is not of type `tangent_type(MutS)`, so any container composing per-field or per-element
# lane reads (an immutable struct's backing, a `Vector{MutS}`'s storage) could not hold it.
@inline function Nfwd.tangent_view(
    d::Lifted{MutS,N,<:MutableDual}, lane::Integer
) where {MutS,N}
    return MutableDualTangentView{N,typeof(d.rep),MutS}(d.rep, d.primal, Int(lane))
end
@inline tangent(d::Lifted{MutS,N,<:MutableDual}, lane::Integer) where {MutS,N} = _materialise_lane(
    d, lane, IdDict{Any,Any}()
)

"""
    dual_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `N`-width forward-mode value type for a primal of
type `P`. This is the type of the inner `V` carried by a
`Lifted{P, N, V}` slot.

Shapes defined so far:

- `P <: IEEEFloat`: `NDual{P, N}` — the packed scalar forward value.
- `Complex{R}` with `R <: IEEEFloat`: `Complex{NDual{R, N}}` — element-wise
  recursion through the complex real/imag parts.
- `Array{T, D}` with `T <: IEEEFloat`: `NDualArray{T, N, D, Array{T, D}, NDual{T, N}, NDualBlock{T, D+1}}`
  — the element-major canonical V wrapper (`primal` aliases user storage; the lane partials
  are a slot-local `(N, size...)` block).
- `Array{Complex{R}, D}` with `R <: IEEEFloat`: `NDualArray{Complex{R}, N, D, Array{Complex{R}, D}, Complex{NDual{R, N}}, NDualBlock{Complex{R}, D+1}}`
  — complex-eltype `NDualArray` variant.
- `Tuple{T1, T2, …}` (concrete tuple): `Tuple{dual_type(Val(N), T1), …}` —
  element-wise recursion via head/tail type-cons.
- `NamedTuple{names, T}` with `T <: Tuple`: `NamedTuple{names, dual_type(Val(N), T)}`
  — same names, fields recursively lifted via the tuple-type path.
- Concrete struct `P` (immutable): `ImmutableDual{NamedTuple{fieldnames(P), Tuple{V_i...}}}`
  where each `V_i = dual_type(Val(N), fieldtype(P, i))`.
- Concrete struct `P` (mutable): `MutableDual{NamedTuple{...}}` — mutable
  counterpart for in-place tangent updates.
- `MemoryRef{T}` with `T <: IEEEFloat` (Julia 1.11+):
  `NDualMemoryRef{T, N, Memory{T}}` — primal ref plus the shared `(N, ncols)` element-major
  partials block (one contiguous column per referenced element).
"""
# No `isconcretetype(P)` guard (unlike the `lifted_type(::Type{P<:IEEEFloat})` sibling, which
# widens a non-concrete `P` to a UnionAll): `dual_type` is the inner-V query and is only ever
# invoked with concrete `P` — the slot-level `lifted_type` performs the non-concrete widening and
# only calls `dual_type` in its `isconcretetype` branch, and the seed factories feed `typeof(x)`.
@foldable @inline dual_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat} = NDual{P,N}
# Non-differentiable primitives — mirrors `tangent_type(T) === NoTangent`
# in reverse mode, returning the forward-mode V sentinel `NoDual`.
@foldable @inline dual_type(
    ::Val{N},
    ::Type{<:Union{Integer,Char,Symbol,Nothing,Type,TypeVar,Module,Expr,Cstring,Cwstring}},
) where {N} = NoDual
# Bottom type, mirroring `tangent_type(Union{}) === Union{}` and
# `lifted_type(Val(N), Union{}) === Union{}` (e.g. a Bottom-typed IR node after a
# guaranteed-throw call). Without this, `dual_type(Val(N), Union{})` MethodErrors.
@foldable @inline dual_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
@foldable @inline function dual_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}
end
@foldable @inline function dual_type(::Val{N}, ::Type{Array{T,D}}) where {N,T<:IEEEFloat,D}
    return Nfwd._ndual_array_V(Array{T,D}, Val(N))
end
# `Base.RefValue{P<:NDualEltype}`: the `NDualRef` parallel-partials V (scalar analogue of `NDualArray`). A *distinct*
# wrapper, so the generic struct recursion never re-lifts it (a bare `RefValue` shadow would be).
# The generic struct rule below covers `Ref` of non-float / aggregate element types as usual.
@foldable @inline function dual_type(
    ::Val{N}, ::Type{<:Base.RefValue{P}}
) where {N,P<:NDualEltype}
    return NDualRef{P,N}
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{Array{Complex{R},D}}
) where {N,R<:IEEEFloat,D}
    return Nfwd._ndual_array_V(Array{Complex{R},D}, Val(N))
end
# Arrays never collapse to whole NoDual: reverse always gives an array tangent.
# Float/complex elements use NDualArray; all others recurse element-wise, including NoDual.
@foldable @generated function dual_type(::Val{N}, ::Type{Array{T,D}}) where {N,T,D}
    return :(Array{dual_type(Val($N), $T),$D})
end
# Tuple recursion: head/tail cons (via `_dual_tuple_v`). Specialized per concrete tuple type by
# Julia's normal dispatch, so concrete tuples resolve at compile time without an @generated function.
# A *standalone* empty tuple is non-differentiable (`tangent_type(Tuple{}) === NoTangent`), so it
# collapses to `NoDual` via the whole-tuple gate below — coherent with reverse, and distinct from
# the cons *base case* `Tuple{}` in `_dual_tuple_v` below, which must stay `Tuple{}` to terminate.
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    # Phantom free-TypeVar `Tuple` (e.g. `Tuple{T, A}`, where `UnionAll(A, Tuple{T,A})` normalises
    # to a `DataType` with dangling typevars): the static parameter `P` is unbound, so referencing
    # it below — or in the `tangent_type(P)` call — would throw `UndefVarError`. Widen to `Any`,
    # mirroring the `@isdefined(P)` guard the `CoDual` constructor uses for the same case.
    @isdefined(P) || return Any
    # Whole-tuple collapse (mirror reverse `tangent_type(P) === NoTangent`; the invariant
    # `dual_type(P) === NoDual` iff `tangent_type(P) === NoTangent`). Checked FIRST so a
    # non-concrete-but-non-differentiable tuple — e.g. `Tuple{Type{Float64},Type{Float64}}`,
    # whose elements are non-diff `Type`s — collapses to `NoDual`, matching `tangent_type`.
    tangent_type(P) === NoTangent && return NoDual
    # Only a `Vararg` tuple must widen to `Any`: its `tuple_type_tail` is a fixed point, so the
    # head/tail `_dual_tuple_v` recursion would not terminate. A fixed-length tuple — even one
    # with abstract elements — recurses finitely and builds a per-element V (each element's own
    # `dual_type` collapses non-diff elements to `NoDual`, which conses fine as a head element).
    Base.isvatuple(P) && return Any
    V = _dual_tuple_v(Val(N), P)
    # A NON-CONCRETE tuple has wholly non-differentiable concretisations, which collapse to
    # `NoDual` at the gate above, so the declared type must admit `NoDual` too: otherwise storing
    # one into a container typed by the abstract tuple — a `Vector{Tuple{NoPullback}}` of reverse
    # pullbacks under forward-over-reverse — is a `TypeError`. `tangent_type` unions here for the
    # same reason. A concrete tuple has exactly one concretisation, itself, and the gate above has
    # already ruled out `NoDual` for it, so the extra member would just name a `Lifted{P,N,V}` slot
    # type no runtime value inhabits (`Lifted` is invariant in `V`) — e.g. `Tuple{Ptr{UInt8},Int}`,
    # whose elements are all `NoDual` while `tangent_type` is not `NoTangent`.
    isconcretetype(P) && return V
    return Tuple{Vararg{NoDual,fieldcount(P)}} <: V ? Union{V,NoDual} : V
end
# Element-wise tuple V, WITHOUT the whole-tuple collapse gate so tails stay `Tuple`. Generated
# rather than head/tail recursive, for the same reason `tangent_type(::Type{<:Tuple})` is: each
# recursive step costs an inference frame, so an `NTuple{1000}` overflows the compiler's stack,
# and Julia does not reliably survive that overflow -- it can take a fatal SIGSEGV instead of
# raising `StackOverflowError` (julia#17109). The top `dual_type(Tuple)` rejects `Vararg` tuples
# before reaching here, so `fieldcount` is finite.
@foldable @generated function _dual_tuple_v(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    Vs = map(i -> :(dual_type(Val(N), fieldtype(P, $i))), 1:fieldcount(P))
    return :(Tuple{$(Vs...)})
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{NamedTuple{names,T}}
) where {N,names,T<:Tuple}
    # Non-differentiable NamedTuples collapse to NoDual, matching reverse and kwargs seeds.
    tangent_type(NamedTuple{names,T}) === NoTangent && return NoDual
    # Mirror `tangent_type(NamedTuple)`: an abstract field (e.g. `parts::Any` in a reverse
    # `MutableTangent` NamedTuple flowing through the forward-over-reverse HVP path) makes
    # `dual_type(Val(N), T)` non-concrete (`Any`), and `NamedTuple{names, Any}` is invalid
    # (the 2nd param must be `<:Tuple`). Widen to `Any` in that case, matching `tangent_type`.
    DT = dual_type(Val(N), T)
    return isconcretetype(DT) ? NamedTuple{names,DT} : Any
end
# `Ptr{T}` canonical V — `NTuple{N, Ptr{T}}`: N parallel partial pointers, one per lane.
# Matches reverse-mode `tangent_type(Ptr{T}) === Ptr{tangent_type(T)}` at the per-lane level.
@foldable @inline function dual_type(::Val{N}, ::Type{Ptr{T}}) where {N,T<:NDualEltype}
    return NTuple{N,Ptr{T}}
end
# A raw `Ptr{Nothing}` carries `N` per-lane pointers: the `pointer(::Array)` chain
# `getfield(:ref) → getfield(:ptr_or_offset) → bitcast` passes through a
# `Ptr{Nothing}` intermediate, whose V must survive so the re-typed `Ptr{T}` after
# the bitcast lands the per-lane partial pointers a foreigncall consumes.
@foldable @inline function dual_type(::Val{N}, ::Type{Ptr{Nothing}}) where {N}
    return NTuple{N,Ptr{Nothing}}
end
# `Ptr{NoTangent}` is the fdata-placeholder element type carried through the reverse-mode pointer
# chain (a `Ptr{Nothing}` bitcast to the fdata tag, later re-bitcast to a differentiable `Ptr{T}` a
# foreigncall/BLAS consumes) — e.g. under forward-over-reverse of a `dot`/BLAS quadratic form. Like
# `Ptr{Nothing}` it must carry the `N` per-lane partial pointers, so the `bitcast` frule's per-lane
# output stays canonical (`V === dual_type`). This is NOT a genuine non-differentiable-element pointer
# (`Ptr{UInt8}`/`Ptr{Int}`, which stay `NoDual` via the generic rule below); `NoTangent` is the tangent
# sentinel and never appears as a user pointer's element type.
@foldable @inline function dual_type(::Val{N}, ::Type{Ptr{NoTangent}}) where {N}
    return NTuple{N,Ptr{NoTangent}}
end
# Non-differentiable-element pointers (e.g. `Ptr{UInt8}`) carry no forward
# derivative — V is `NoDual`, mirroring `tangent_type(T) === NoTangent`. Without
# this the zero-field generic fallback returns `NTuple{N, Ptr{NoTangent}}`, which
# doesn't match the structurally-NoDual `frule!!` outputs for such pointers (e.g.
# the `jl_string_ptr` foreigncall). The `NDualEltype` overload above is more
# specific and wins for differentiable element types.
@foldable @generated function dual_type(::Val{N}, ::Type{Ptr{T}}) where {N,T}
    # `tangent_type($T)` resolves in the RETURNED expression (at the call world, where an
    # extension's element-type overload is visible), never the generator body.
    return :(tangent_type($T) === NoTangent ? NoDual : NTuple{$N,Ptr{tangent_type($T)}})
end
# MemoryRef canonical V (Julia 1.11+); paired with NDualMemoryRef above.
# Memory itself is `<: AbstractArray{T, 1}` on 1.11+ — its canonical V is
# an NDualArray over `Memory{T}`: `Memory{T} → NDualArray{T, N, 1, Memory{T}}`.
@static if VERSION >= v"1.11-rc4"
    @foldable @inline function dual_type(
        ::Val{N}, ::Type{MemoryRef{T}}
    ) where {N,T<:IEEEFloat}
        return NDualMemoryRef{T,N,Memory{T}}
    end
    @foldable @inline function dual_type(::Val{N}, ::Type{Memory{T}}) where {N,T<:IEEEFloat}
        return NDualArray{T,N,1,Memory{T},NDual{T,N},NDualBlock{T,2}}
    end
    @foldable @inline function dual_type(
        ::Val{N}, ::Type{Memory{Complex{R}}}
    ) where {N,R<:IEEEFloat}
        return NDualArray{
            Complex{R},N,1,Memory{Complex{R}},Complex{NDual{R,N}},NDualBlock{Complex{R},2}
        }
    end
    # Complex `MemoryRef`: the memory.jl frules build/consume `NDualMemoryRef` for any
    # `P<:NDualEltype` (complex included), so the canonical V must be `NDualMemoryRef`, not the
    # element-wise `MemoryRef{Complex{NDual}}` the general fallback would give (mismatch ->
    # MethodError on complex `push!`/grow). Parallels the float `MemoryRef` overload above.
    @foldable @inline function dual_type(
        ::Val{N}, ::Type{MemoryRef{Complex{R}}}
    ) where {N,R<:IEEEFloat}
        return NDualMemoryRef{Complex{R},N,Memory{Complex{R}}}
    end
    # Non-float Memory / MemoryRef recurse element-wise, including reverse pullback storage
    # under forward-over-reverse. Float overloads above provide block-backed Vs.
    @foldable @generated function dual_type(::Val{N}, ::Type{Memory{T}}) where {N,T}
        return :(Memory{dual_type(Val($N), $T)})
    end
    @foldable @generated function dual_type(::Val{N}, ::Type{MemoryRef{T}}) where {N,T}
        return :(MemoryRef{dual_type(Val($N), $T)})
    end
end

# Structural fallback: abstract P widens to Any; non-differentiable P gives NoDual.
# Fields recurse uniformly, with seed factories coercing into the declared backing NamedTuple.
@foldable @generated function dual_type(::Val{N}, ::Type{P}) where {N,P}
    # Deliberately does NOT distribute over `Union` the way reverse-mode `tangent_type`
    # union-splits: a non-concrete `P` (including any `Union`) widens to `Any`. `Lifted` is
    # invariant in its primal parameter, so a slot annotated `Lifted{Union{A,B},N,V}` cannot
    # hold the runtime `Lifted{A,...}`/`Lifted{B,...}` values an IR join actually produces —
    # a union-distributed `V === Union{dual_type(A),dual_type(B)}` would let `PiNode`s /
    # OpaqueClosures lower a valid branch to `unreachable`. The `Any` widening keeps the slot
    # type sound; concrete leaves recover the exact `V` when the seed factories feed `typeof(x)`.
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
    # The structural lift mirrors a STRUCTURAL reverse tangent and nothing else. A `P` with a
    # custom `tangent_type` (`=== P`, say) and no `dual_type` of its own would otherwise be
    # lifted field-wise, seed and run, and fail only at the reverse bridge with
    # `FieldError: type P has no field fields`. Refuse it here; the check stays in the returned
    # expression so a later extension overload takes effect.
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

"""
    lifted_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `Lifted{P, N, V}` slot type for a primal of type `P`
at width `N`. For concrete `P`, equals `Lifted{P, N, dual_type(Val(N), P)}`.

Shapes defined so far:

- `P <: IEEEFloat`: `Lifted{P, N, NDual{P, N}}`.
- `Complex{R}` with `R <: IEEEFloat`: `Lifted{Complex{R}, N, Complex{NDual{R, N}}}`.
- `Array{T, D}` with `T <: IEEEFloat`: `Lifted{Array{T, D}, N, NDualArray{T, N, D, Array{T, D}, NDual{T, N}, NDualBlock{T, D+1}}}`.
- `Array{Complex{R}, D}` with `R <: IEEEFloat`: `Lifted{Array{Complex{R}, D}, N, NDualArray{Complex{R}, N, D, Array{Complex{R}, D}, Complex{NDual{R, N}}, NDualBlock{Complex{R}, D+1}}}`.
- `MemoryRef{T}` with `T <: IEEEFloat` (Julia 1.11+):
  `Lifted{MemoryRef{T}, N, NDualMemoryRef{T, N, Memory{T}}}`.
- `P <: Tuple` (concrete): `Lifted{P, N, dual_type(Val(N), P)}`.
- `P <: NamedTuple{names, <:Tuple}`: `Lifted{P, N, dual_type(Val(N), P)}`.
- Concrete struct `P`: `Lifted{P, N, dual_type(Val(N), P)}` where the inner
  V is `ImmutableDual` (immutable) or `MutableDual` (mutable).
"""
# The four metatype kinds, whose VALUES are themselves types. Each is `isconcretetype`, so the
# generic `lifted_type` would give it a bounded slot, which the runtime `Lifted{Type{X}}` cannot
# satisfy — `Lifted` is invariant in `P`. `_fwd_zd_arg_bound` needs the same set for
# `@zero_derivative` argument bounds, so both read it from here.
@inline _is_metatype_kind(@nospecialize(T)) =
    T === DataType || T === UnionAll || T === Union || T === Core.TypeofBottom

# The slot for a concrete `P`. `dual_type` may return a WIDENED upper bound rather than the
# exact `V` (the `Tuple`/`NamedTuple` methods do so whenever an element's own dual type is
# non-concrete), and `Lifted` is invariant in `V`, so `Lifted{P,N,Any}` would be an exact claim
# that no runtime slot satisfies. Emit the sound `Lifted{P,N,V} where V` in that case.
@inline function _concrete_lifted_type(::Val{N}, ::Type{P}) where {N,P}
    V = dual_type(Val(N), P)
    return isconcretetype(V) ? Lifted{P,N,V} : (Lifted{P,N,W} where {W})
end

@foldable @inline lifted_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
# Abstract tuple/named-tuple `P` (e.g. a grouped-vararg `Tuple{Function,
# Vararg{Any}}` in the forward IR) must widen to a UnionAll: `Lifted` is invariant
# in `P`, so a concrete runtime `Lifted{Tuple{f,x},…}` is *not* a subtype of
# `Lifted{Tuple{Function,Vararg},N,Any}` and the OpaqueClosure arg typeassert
# would reject it (mirrors the generic struct overload below).
# `@foldable` (rather than a plain `@inline`): asserts `:foldable` effects so the foldability holds
# under code-coverage instrumentation (which otherwise kills `effect_free`), and — being registered
# as a DispatchDoctor `IncompatibleMacro` — also exempts this method from the module's `@stable`
# wrapper, which would otherwise read the static parameters unconditionally and defeat the
# `@isdefined(P)` phantom-TypeVar guard below (UndefVarError under the dispatch_doctor suite).
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    @isdefined(P) || return Lifted  # phantom free-TypeVar Tuple — broad `Lifted` slot, as the generic `lifted_type` guard below
    return if isconcretetype(P)
        _concrete_lifted_type(Val(N), P)
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# True when every member of `U` is non-differentiable (`tangent_type === NoTangent`), so its
# `lifted_type` is a concrete `NoDual`-V `Lifted` that union-splits safely (see `lifted_type` below).
function _all_nodual_union_members(@nospecialize(U))
    if U isa Union
        _all_nodual_union_members(U.a) && _all_nodual_union_members(U.b)
    else
        (U === Union{} || tangent_type(U) === NoTangent)
    end
end

# Abstract P and metatypes need a UnionAll slot: Lifted is invariant in P.
# @foldable preserves folding under coverage and exempts the phantom-P guard from
# DispatchDoctor's unconditional static-parameter reads, as in the Tuple overload.
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    # `@isdefined(P)` is false when the static parameter couldn't be bound — e.g. a `UnionAll`
    # with a free `TypeVar` in its body. Touching `P` would then throw `UndefVarError`; return the
    # broad `Lifted` instead — the same `@isdefined` fallback the `CoDual` ctors and
    # `codual_type(::Type{Type{P}})` use for this phantom-`TypeVar` case (chalk-lab/Mooncake.jl#1191).
    @isdefined(P) || return Lifted
    # Distribute over a `Union` of purely non-differentiable results (e.g. `findfirst` returning
    # `Union{Nothing,Int}`): the slot type is the small `Union{Lifted{A,…}, Lifted{B,…}}` of concrete
    # `NoDual`-V `Lifted`s, which union-splits box-free at the OpaqueClosure return, rather than the
    # widened `Lifted{T,…} where T<:Union{A,B}` that boxes (`Lifted` is invariant). Restricted to
    # all-non-differentiable members: a member whose V is not `NoDual` (e.g. `Vector{Any}`) in an
    # invariant-`Lifted` Union return tripped an OC return-assertion crash, so those keep the UnionAll.
    P isa Union &&
        _all_nodual_union_members(P) &&
        return Union{lifted_type(Val(N), P.a),lifted_type(Val(N), P.b)}
    return if isconcretetype(P) && !_is_metatype_kind(P)
        _concrete_lifted_type(Val(N), P)
    elseif P <: Type
        # Metatype kind (e.g. `DataType`): a type-unstable type-valued result is inferred as `P`,
        # but the runtime value is sharpened to `Lifted{Type{X}}`. A *bounded* `Lifted{T<:P}` slot
        # trips an OpaqueClosure return-type-assertion quirk (the bound + `Lifted` invariance) and
        # rejects the value even though `Type{X} <: P` holds statically. Drop the bound entirely.
        Lifted{T,N,V} where {T,V}
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# A precise `Type{X}` slot (e.g. a Type-valued callable like the `Pt2{Float64}` constructor) is
# monomorphic — its sole inhabitant is `X` — so the concrete `Lifted{Type{X}, N, …}` is exact and
# carries no impossible type fact. `isconcretetype(Type{X})` is `false`, so the generic method above
# would route it to the UnionAll-widened branch, forcing the runtime slot to box at the OC argument
# boundary. The kind-widening is needed only for the genuinely abstract metatypes
# (`DataType`, `Type`, `Type{<:T}` — which stay on the generic method); it explicitly excludes these
# well-behaved `Type{X}` singletons, so narrow them here to keep the slot box-free.
@foldable @inline function lifted_type(::Val{N}, ::Type{Type{X}}) where {N,X}
    return Lifted{Type{X},N,dual_type(Val(N), Type{X})}
end

# Width-N seed factories return bare Vs; *_lifted factories wrap them in Lifted slots.

@inline function zero_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

# lift(primal, tangent_type(P)) is the width-1 user-JVP boundary. Width-N basis seeds
# are built with basis_lifted!! and Lifted{P,N}.
@inline lift(x::T, ẋ::T) where {T<:IEEEFloat} = Lifted{T,1}(x, NDual{T,1}(x, (ẋ,)))
@inline function lift(x::A, ẋ::A) where {E<:NDualEltype,D,A<:Array{E,D}}
    return Lifted{A,1}(x, NDualArray{E,1,D,A}(x, (ẋ,)))
end
@inline function lift(x::Complex{R}, ẋ::Complex{R}) where {R<:IEEEFloat}
    re = NDual{R,1}(real(x), (real(ẋ),))
    im_ = NDual{R,1}(imag(x), (imag(ẋ),))
    return Lifted{Complex{R},1}(x, Complex{NDual{R,1}}(re, im_))
end
@inline lift(x, ::NoTangent) = uninit_lifted(Val(1), x)
@inline lift(x::Ptr{T}, ẋ::Ptr{T}) where {T} = Lifted{Ptr{T},1}(x, (ẋ,))
# A `Ptr` to a non-differentiable element (`Ptr{Int}`, `Ptr{UInt8}`, …) has reverse tangent
# `Ptr{NoTangent}` and canonical forward V `NoDual` (`dual_type(Ptr{T}) === NoDual`). The
# generic method above only matches a matching `Ptr{T}` tangent, so this covers the non-diff
# case (e.g. lifting a hand-written `CoDual{Ptr{Int},Ptr{NoTangent}}` test input).
@inline lift(x::Ptr{T}, ::Ptr{NoTangent}) where {T} = Lifted{Ptr{T},1}(x, NoDual())
# `Ptr{Nothing}` is the exception: its forward V is the `NTuple{1,Ptr{Nothing}}` per-lane
# shape (cf. `pointer_from_objref`), the primal address (a raw address has no derivative).
# More specific than the `Ptr{T}` method above (same `Ptr{NoTangent}` tangent) so it wins.
@inline lift(x::Ptr{Nothing}, ::Ptr{NoTangent}) = Lifted{Ptr{Nothing},1}(x, (x,))
# `Ptr{NoTangent}` carries a real forward lane (`dual_type` gives `NTuple{1,Ptr{NoTangent}}`, not
# `NoDual`), so it must not fall to the non-differentiable method above.
@inline lift(x::Ptr{NoTangent}, ẋ::Ptr{NoTangent}) = Lifted{Ptr{NoTangent},1}(x, (ẋ,))
# A `Ptr{Nothing}`'s reverse tangent is an `VoidPtrTangent`; the forward lane keeps the `uninit_*`
# placeholder convention, as the `Ptr{NoTangent}` method above does.
@inline lift(x::Ptr{Nothing}, ::VoidPtrTangent) = Lifted{Ptr{Nothing},1}(x, (x,))
@static if VERSION >= v"1.11-rc4"
    # `MemoryRef{T}` (T<:NDualEltype) reverse fdata is itself a `MemoryRef{T}` (the
    # derivative storage); its forward V is the block-backed `NDualMemoryRef`. The eltype bound
    # must match the `Memory` lift below: where they disagree, a ref falls through to the generic
    # element-wise lift, which builds a V that is not what `dual_type` declares. Reached in
    # forward-over-reverse, where a reverse rule's `dx::MemoryRef` field is lifted. The seed
    # values are PACKED into a fresh block (copy semantics, like the `Array` lift): `unlift`
    # reads the result back out of the block via the lane accessor, so the round-trip is
    # consistent even though `ẋ` itself is not aliased.
    @inline function lift(x::MemoryRef{T}, ẋ::MemoryRef{T}) where {T<:NDualEltype}
        len = length(ẋ.mem)
        block = NDualBlock{T,2}(undef, 1, len)
        copyto!(getfield(block, :parent), 1, ẋ.mem, 1, len)
        return Lifted{MemoryRef{T},1}(
            x, NDualMemoryRef{T,1,Memory{T}}(x, block, Core.memoryrefoffset(x))
        )
    end
    # `_lift_backing` always calls the 3-arg form; without these `NDualArray`-specific 3-arg
    # passthroughs a lifted reverse rule's float `dx::MemoryRef`/`Memory` field falls
    # to the generic element-wise lift below, producing a `MemoryRef{NDual}` that cannot convert
    # to the declared `NDualArray` V. Mirrors the float `Array` passthroughs.
    # Honour the aliasing cache, as the float `Array` overload does and for the same reason: the V
    # packs a fresh block per lift, so without it two aliased primals get distinct V objects and a
    # mutation through one is invisible through the other.
    @inline function lift(
        x::MemoryRef{T}, ẋ::MemoryRef{T}, c::Union{Nothing,IdDict}
    ) where {T<:NDualEltype}
        c isa IdDict || return lift(x, ẋ)
        haskey(c, x) && return c[x]::Lifted{MemoryRef{T},1}
        # Window the backing `Memory`'s V rather than copying `ẋ.mem` into a private block: a
        # `Memory` and a ref into it are one storage, so a private copy drops every contribution
        # reaching the value through the other position. Guarded on `ẋ` mirroring the primal's
        # geometry, as the float `Array` lift is, since that is what makes `ẋ.mem` the tangent for
        # `x.mem` rather than a buffer of its own; the 2-argument copying form is the fallback.
        lifted =
            if length(ẋ.mem) == length(x.mem) &&
                Core.memoryrefoffset(ẋ) == Core.memoryrefoffset(x)
                memv = tangent(lift(x.mem, ẋ.mem, c))
                Lifted{MemoryRef{T},1}(
                    x,
                    NDualMemoryRef{T,1,Memory{T}}(
                        x, getfield(memv, :partials_block), Core.memoryrefoffset(x)
                    ),
                )
            else
                lift(x, ẋ)
            end
        c[x] = lifted
        return lifted
    end
    # `Memory{T}` (T<:IEEEFloat / Complex{<:IEEEFloat}) lifts to the NDualArray,
    # mirroring the `Array` overloads above; reached when a reverse rule's `Memory`
    # field is lifted under forward-over-reverse, or a Memory primal is seeded.
    @inline function lift(x::A, ẋ::A) where {E<:NDualEltype,A<:Memory{E}}
        return Lifted{A,1}(x, NDualArray{E,1,1,A}(x, (ẋ,)))
    end
    @inline function lift(
        x::A, ẋ::A, c::Union{Nothing,IdDict}
    ) where {E<:NDualEltype,A<:Memory{E}}
        c isa IdDict || return lift(x, ẋ)
        haskey(c, x) && return c[x]::Lifted{A,1}
        lifted = lift(x, ẋ)
        c[x] = lifted
        return lifted
    end
    # Non-differentiable-element `Memory` (reverse tangent `Memory{NoTangent}`)
    # lifts element-wise to `Memory{NoDual}`, mirroring the `Array{<:NoTangent}`
    # overload and `dual_type(Memory{T}) = Memory{NoDual}`.
    @inline lift(x::Memory, ẋ::Memory{<:NoTangent}) = Lifted{typeof(x),1}(
        x, map(_ -> NoDual(), ẋ)
    )
    # General element-wise `Memory` (differentiable non-float / nested / `Any` element):
    # element-wise V `Memory{dual_type(elt)}`, mirroring the generic `Array` lift.
    @inline lift(x::Memory, ẋ::Memory) = lift(x, ẋ, nothing)
    @inline function lift(x::Memory, ẋ::Memory, c::Union{Nothing,IdDict})
        # A top-level call arrives with `c === nothing`; upgrade it to a shared `IdDict` as every
        # other aggregate lift does, or two elements holding one array get independent partials
        # and the JVP is silently wrong. Register before filling — see the `Array` lift — so a
        # cycle reaching `x` again returns the shell instead of recursing forever.
        d = c === nothing ? IdDict() : c
        haskey(d, x) && return d[x]::Lifted{typeof(x),1}
        v = similar(x, dual_type(Val(1), eltype(x)))
        lifted = Lifted{typeof(x),1,typeof(v)}(x, v)
        d[x] = lifted
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = tangent(lift(x[i], ẋ[i], d)))
        end
        return lifted
    end
    # General element-wise `MemoryRef` lift (non-float / nested / `Any` / `NoTangent`
    # element): lift the `.mem` via the Memory lift, then `memoryref` at the
    # offset. The `MemoryRef{IEEEFloat}` `NDualMemoryRef` overload above is more specific.
    @inline lift(x::MemoryRef, ẋ::MemoryRef) = lift(x, ẋ, nothing)
    @inline function lift(x::MemoryRef, ẋ::MemoryRef, c::Union{Nothing,IdDict})
        mem_v = tangent(lift(x.mem, ẋ.mem, c))
        ref_v = _memoryref_at(mem_v, Core.memoryrefoffset(x))
        return Lifted{typeof(x),1,typeof(ref_v)}(x, ref_v)
    end
end
# Thread one aliasing cache through aggregate lifts. MistyClosure uses it for shared
# forward/pullback captures, or HVP partials pushed forward are invisible on the pullback.
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
# `Ref{P<:NDualEltype}` (V `NDualRef`): build the parallel partials buffer from the reverse
# tangent's scalar (its `:x` field is non-always-init, hence `PossiblyUninitTangent`-wrapped). More
# specific than the generic `MutableTangent` lift below, which would route through `MutableDual`.
@inline lift(x::Base.RefValue{P}, ẋ::MutableTangent) where {P<:NDualEltype} = lift(
    x, ẋ, nothing
)
@inline function lift(
    x::Base.RefValue{P}, ẋ::MutableTangent, c::Union{Nothing,IdDict}
) where {P<:NDualEltype}
    # Register storage-owning leaves so repeated Refs share partials. Build here because
    # the two-argument entry delegates to this method with nothing.
    c isa IdDict && haskey(c, x) && return c[x]::Lifted{Base.RefValue{P},1}
    lifted = Lifted{Base.RefValue{P},1}(
        x, NDualRef{P,1}(Base.RefValue{NTuple{1,P}}((val(ẋ.fields.x),)))
    )
    c isa IdDict && (c[x] = lifted)
    return lifted
end
@inline lift(x::P, ẋ::Tangent) where {P} = lift(x, ẋ, nothing)
@inline function lift(x::P, ẋ::Tangent, c::Union{Nothing,IdDict}) where {P}
    backing = fieldtype(dual_type(Val(1), P), 1)
    # A top-level call arrives with `c === nothing`; upgrade it to a shared `IdDict` so that
    # mutable children (e.g. an `Array` field) aliased across fields dedup to one V, matching
    # the reverse `zero_tangent_internal` aliasing invariant. (Immutable structs can't cycle
    # back to themselves, so — unlike the mutable-struct lift — no self-registration is needed.)
    d = c === nothing ? IdDict() : c
    return Lifted{P,1}(x, ImmutableDual(_lift_backing(x, ẋ.fields, backing, d)))
end
@inline lift(x::P, ẋ::MutableTangent) where {P} = lift(x, ẋ, nothing)
@inline function lift(x::P, ẋ::MutableTangent, c::Union{Nothing,IdDict}) where {P}
    backing = fieldtype(dual_type(Val(1), P), 1)
    LT = Lifted{P,1,MutableDual{backing}}
    # A mutable struct may reference itself (directly or through a cycle), so
    # register an uninitialised `MutableDual` shell in the aliasing cache before
    # building its fields: the recursive `_lift_backing` then finds and returns
    # this shell when the cycle reaches `x` again. Mirrors `zero_tangent_internal`.
    d = c === nothing ? IdDict() : c
    haskey(d, x) && return d[x]::LT
    lifted = LT(x, MutableDual{backing}())
    d[x] = lifted
    lifted.rep.fields = _lift_backing(x, ẋ.fields, backing, d)
    return lifted
end
# A possibly-uninit reverse-tangent field lifts to the inner V (the forward
# backing slot is the plain V, not PUT-wrapped). Reached only for defined
# fields — `lift(::Tangent)`/`lift(::MutableTangent)` recurse via
# `getfield(x, name)`, which requires the primal field to be defined.
@inline lift(x, ẋ::PossiblyUninitTangent) = lift(x, val(ẋ))
@inline lift(x, ẋ::PossiblyUninitTangent, c::Union{Nothing,IdDict}) = lift(x, val(ẋ), c)
# Non-differentiable element array: the reverse tangent is an all-`NoTangent` array; the forward V
# mirrors it element-wise as an `Array{NoDual}` (coherent with `dual_type(Array{T,D}) =
# Array{NoDual,D}`). The 3-arg passthrough keeps this more-specific behaviour ahead of the
# element-wise overload below when a cache is threaded.
@inline lift(x::Array, ẋ::Array{<:NoTangent}) = Lifted{typeof(x),1}(
    x, map(_ -> NoDual(), ẋ)
)
@inline lift(x::Array, ẋ::Array{<:NoTangent}, ::Union{Nothing,IdDict}) = lift(x, ẋ)
# Float / Complex-float element arrays are terminal (their V aliases `ẋ`); they match the
# element-wise overload below by element type. Honor the aliasing cache so two aliased primals
# share ONE V, matching the reverse invariant: the V's block is a fresh copy per lift, so without
# the cache aliased arrays get distinct V objects.
@inline function lift(
    x::A, ẋ::A, c::Union{Nothing,IdDict}
) where {E<:NDualEltype,D,A<:Array{E,D}}
    c isa IdDict || return lift(x, ẋ)
    haskey(c, x) && return c[x]::Lifted{A,1}
    # Window the backing `Memory`'s V, as the seed path does (`_derived_array_dual`), so an
    # aggregate holding both a `Vector` and its `Memory` gets one partials store. Only when `ẋ`
    # mirrors the primal's `Memory` geometry: that is what an aliasing-preserving reverse tangent
    # gives (`zero_tangent` over the aliased pair does), and it is also what makes `ẋ.ref.mem` the
    # tangent for `x.ref.mem` rather than a shorter buffer of its own.
    lifted = @static if VERSION >= v"1.11-rc4"
        if length(getfield(ẋ, :ref).mem) == length(getfield(x, :ref).mem) &&
            Core.memoryrefoffset(getfield(ẋ, :ref)) ==
           Core.memoryrefoffset(getfield(x, :ref))
            memv = tangent(lift(getfield(x, :ref).mem, getfield(ẋ, :ref).mem, c))
            Lifted{A,1}(x, _derived_array_dual(Val(1), x, memv))
        else
            lift(x, ẋ)
        end
    else
        # 1.10 has no backing `Memory` to window, so share through a STORAGE key and a block over
        # the cached block's parent — the same device the 1.10 seed path uses. Keyed separately
        # from the object key below, which cannot see that `a` and `reshape(a)` are one buffer.
        sk = (Base.dataids(x), length(x))
        if haskey(c, sk)
            cached = tangent(c[sk]::Lifted)
            Lifted{A,1}(
                x,
                NDualArray{E,1,D,A}(
                    x,
                    Nfwd.NDualBlock{E,D + 1}(
                        getfield(getfield(cached, :partials_block), :parent),
                        (1, size(x)...),
                    ),
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
# Differentiable non-float-element array: element-wise V `Array{dual_type(Val(1), T), D}`,
# built element-wise from the per-element lift (coherent with `dual_type` above).
# The IEEEFloat / Complex / all-`NoTangent` overloads are more specific and win.
@inline lift(x::Array{T,D}, ẋ::Array) where {T,D} = lift(x, ẋ, nothing)
@inline function lift(x::Array{T,D}, ẋ::Array, c::Union{Nothing,IdDict}) where {T,D}
    # Register in the cache *before* filling, so aliased occurrences of `x` share one V (matching
    # the reverse `zero_tangent_internal(::Array)` aliasing invariant) and self-referential arrays
    # terminate. A top-level call may arrive with `c === nothing` (the 2-arg entry); upgrade it to
    # a fresh `IdDict` so the cycle break holds, mirroring the mutable-struct `lift` above. Float /
    # Complex-float element arrays don't reach here — their V aliases `ẋ`.
    d = c === nothing ? IdDict() : c
    haskey(d, x) && return d[x]::Lifted{typeof(x),1}
    Vel = dual_type(Val(1), T)
    v = similar(x, Vel)
    lifted = Lifted{typeof(x),1,typeof(v)}(x, v)
    d[x] = lifted
    @inbounds for i in eachindex(x)
        if isassigned(x, i)
            v[i] = tangent(lift(x[i], ẋ[i], d))
        end
    end
    return lifted
end
# V-shape passthrough — the test framework's tangent-shape arithmetic
# sometimes feeds raw Lifted V values (NoDual, ImmutableDual, MutableDual)
# back into `lift`. Wrap directly rather than re-deriving V from the (now-V)
# tangent input.
@inline lift(x::P, ẋ::Union{NoDual,ImmutableDual,MutableDual}) where {P} = Lifted{P,1}(x, ẋ)
# Tuple / NamedTuple primal + per-field reverse tangents → per-field V. Reached
# in real AD (lifting a tuple/named-tuple primal) and when the test framework's
# tangent-shape arithmetic recurses through a `Tangent`'s fields and feeds the V
# values back into `lift`.
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

# Primitive seeds bypass the structural fallback, matching their NoDual dual_type.
for f in (:zero_dual, :uninit_dual)
    @eval @inline $f(::Val{N}, ::Union{Integer,Char,Symbol,Nothing}) where {N} = NoDual()
    @eval @inline $f(::Val{N}, ::Union{Type,TypeVar,Module,Expr}) where {N} = NoDual()
    @eval @inline $f(::Val{N}, ::Union{Cstring,Cwstring}) where {N} = NoDual()
end
@inline randn_dual(::Val{N}, ::AbstractRNG, ::Union{Integer,Char,Symbol,Nothing}) where {N} = NoDual()
@inline randn_dual(::Val{N}, ::AbstractRNG, ::Union{Type,TypeVar,Module,Expr}) where {N} = NoDual()
@inline randn_dual(::Val{N}, ::AbstractRNG, ::Union{Cstring,Cwstring}) where {N} = NoDual()

# ── Array seed factories (T <: IEEEFloat) ───────────────────────────────────
#
# Build an `NDualArray` whose `primal` aliases the user's array and whose lane-partials
# block is slot-local — no aliasing with the user's array.

@inline function zero_dual(::Val{N}, x::A) where {N,E<:NDualEltype,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x)
end
@inline function uninit_dual(::Val{N}, x::A) where {N,E<:NDualEltype,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x, NDualBlock{E,D + 1}(undef, N, size(x)...))
end

# `Ref{P<:NDualEltype}` → `NDualRef` (scalar analogue of the `Array` factories above): fresh
# slot-local parallel partials. Zero and uninit coincide (the partials are bits scalars).
@inline function zero_dual(::Val{N}, ::Base.RefValue{P}) where {N,P<:NDualEltype}
    return NDualRef{P,N}()
end
@inline uninit_dual(w::Val{N}, r::Base.RefValue{P}) where {N,P<:NDualEltype} = zero_dual(
    w, r
)

# A `Ptr` has no numeric partials to zero or randomise (its V is `NTuple{N,Ptr}` of pointers, never
# a float tangent that could be read as garbage), so both seeds are the uninitialised one — and
# critically the generic struct fallback would route a `Ptr` field through the 1-arg
# `zero_tangent(::Ptr)`, which throws. (Constants seed via `zero_lifted`, so this path is live.)
@inline zero_dual(w::Val{N}, x::Ptr) where {N} = uninit_dual(w, x)
@inline randn_dual(w::Val{N}, ::AbstractRNG, x::Ptr) where {N} = uninit_dual(w, x)
# Take the lane type from `dual_type`, not from the reverse `uninit_tangent`: they agree for
# `Ptr{Float64}` but not for `Ptr{Nothing}`, whose reverse tangent is a `VoidPtrTangent`. Each lane
# keeps the `uninit_*` convention — the primal address as the declared pointer type, never read.
@inline function uninit_dual(w::Val{N}, x::Ptr) where {N}
    V = dual_type(w, typeof(x))
    V === NoDual && return NoDual()
    return ntuple(_ -> bitcast(fieldtype(V, 1), x), w)
end

@inline function randn_dual(
    ::Val{N}, rng::AbstractRNG, x::A
) where {N,E<:NDualEltype,D,A<:Array{E,D}}
    return NDualArray{E,N,D,A}(x, ntuple(_ -> randn(rng, E, size(x)), Val(N)))
end
@inline function randn_dual(
    ::Val{N}, rng::AbstractRNG, ::Base.RefValue{P}
) where {N,P<:NDualEltype}
    return NDualRef{P,N}(Base.RefValue{NTuple{N,P}}(ntuple(_ -> randn(rng, P), Val(N))))
end

# Element-wise array seeds skip undefined slots; numeric elements use NDualArray above.
# Keep dual_type in the caller's world so extension overloads remain visible.
for factory in (:zero_dual, :uninit_dual, :randn_dual)
    rng_args = factory === :randn_dual ? (:(rng::AbstractRNG),) : ()
    rng_vals = factory === :randn_dual ? (:rng,) : ()
    @eval @inline function $factory(w::Val{N}, $(rng_args...), x::Array{T,D}) where {N,T,D}
        v = similar(x, dual_type(w, T))
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = $factory(w, $(rng_vals...), x[i]))
        end
        return v
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

# Per-field seed expression: an always-initialised field is the bare seed call;
# a possibly-uninit field is a `PossiblyUninitTangent{Vfield}` guarded by
# `isdefined`, mirroring reverse-mode `zero_tangent_internal`. `callexpr` is the
# field's seed call (references the factory's runtime `x`, so it is spliced into
# the factory body).
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

# Seed factories coerce the per-field seed tuple into the *declared* backing
# NamedTuple `fieldtype(dual_type(Val(N), P), 1)` — abstract fields stored as
# `Any`, possibly-uninit fields wrapped in `PossiblyUninitTangent` — mirroring
# reverse-mode's `backing_type(P)(...)`. Keeps `typeof(seed) === dual_type(Val(N), P)`.
for (f, helper) in
    ((:zero_dual, :_zero_dual_zero_field), (:uninit_dual, :_uninit_dual_zero_field))
    @eval @generated function $f(::Val{N}, x::P) where {N,P}
        isconcretetype(P) || return :(error($("$($f): P=$P is not concrete")))
        # NoDual has no backing to seed. Resolve dual_type at the call world.
        if fieldcount(P) == 0
            return :(
                if dual_type(Val($N), typeof(x)) === NoDual
                    NoDual()
                else
                    $($(QuoteNode(helper)))(Val($N), x)
                end
            )
        end
        seeds = map(1:fieldcount(P)) do i
            nm = QuoteNode(fieldnames(P)[i])
            return _seed_field_expr(N, P, i, :($($f)(Val($N), getfield(x, $nm))))
        end
        wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
        msg =
            "$($f): $P declares a `dual_type` that is not the structural lift, so this fallback " *
            "cannot seed it. A type with its own V needs BOTH seed entry points: define " *
            "`zero_dual`/`uninit_dual`/`randn_dual` for it alongside its `_*_dual_internal` " *
            "overloads (see `TwicePrecision`)."
        return quote
            V = dual_type(Val($N), typeof(x))
            V === NoDual && return NoDual()
            V <: $wrapper || error($msg)
            $wrapper(fieldtype(V, 1)(($(seeds...),)))
        end
    end
end

@generated function randn_dual(::Val{N}, rng::AbstractRNG, x::P) where {N,P}
    isconcretetype(P) || return :(error("randn_dual: P=$P is not concrete"))
    # As `zero_dual` / `uninit_dual` above: `NoDual` V has no backing; the
    # `dual_type(...) === NoDual` test goes in the returned expression, not the body.
    if fieldcount(P) == 0
        return :(
            if dual_type(Val($N), typeof(x)) === NoDual
                NoDual()
            else
                _randn_dual_zero_field(Val($N), rng, x)
            end
        )
    end
    seeds = map(1:fieldcount(P)) do i
        nm = QuoteNode(fieldnames(P)[i])
        return _seed_field_expr(N, P, i, :(randn_dual(Val($N), rng, getfield(x, $nm))))
    end
    wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
    msg =
        "randn_dual: $P declares a `dual_type` that is not the structural lift, so this " *
        "fallback cannot seed it. A type with its own V needs BOTH seed entry points: define " *
        "`zero_dual`/`uninit_dual`/`randn_dual` for it alongside its `_*_dual_internal` " *
        "overloads (see `TwicePrecision`)."
    return quote
        V = dual_type(Val($N), typeof(x))
        V === NoDual && return NoDual()
        V <: $wrapper || error($msg)
        $wrapper(fieldtype(V, 1)(($(seeds...),)))
    end
end

# Cache-aware seeds register mutable shells before recursion to preserve cycles and aliases.
# Custom Vs need BOTH cache-free and cache-threading factories; direct callers use the former.
@static if VERSION >= v"1.11-rc4"
    # Window the backing Memory's cached block so Array/Memory aliases share partials.
    @inline function _derived_array_dual(
        ::Val{N}, x::Array{E,D}, memv
    ) where {N,E<:NDualEltype,D}
        dims = (N, size(x)...)
        block = Nfwd._window_block(
            getfield(memv, :partials_block),
            Val(N),
            Core.memoryrefoffset(getfield(x, :ref)),
            dims,
        )
        return NDualArray{E,N,D,typeof(x)}(x, block)
    end
end

# Julia 1.10 has no Memory: key on (Base.dataids, length) and share the cached block's
# parent across reshapes. Every factory uses this; owned runs only on a cache miss.
@static if VERSION < v"1.11-rc4"
    @inline function _cached_array_dual(
        w::Val{N}, x::Array, d::MaybeCache, owned::F
    ) where {N,F}
        d isa NoCache && return owned()
        sk = (Base.dataids(x), length(x))
        if haskey(d, sk)
            # The cached entry is whatever array claimed the buffer first, so its own type is not
            # `x`'s -- a vector's seed serves a reshape. Only the block's flat `parent` is shared,
            # and it is a `Vector{eltype(x)}` whatever the shape, so assert THAT: without it the
            # constructor takes its argument as `Any` and dispatches at runtime.
            shared = getfield(getfield(d[sk], :partials_block), :parent)::Vector{eltype(x)}
            return NDualArray{eltype(x),N,ndims(x),typeof(x)}(
                x, Nfwd.NDualBlock{eltype(x),ndims(x) + 1}(shared, (N, size(x)...))
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
    if factory !== :randn_dual
        @eval begin
            @generated function $internal(w::Val{N}, x::P, d::MaybeCache) where {N,P}
                # `fieldcount(P) == 0` is world-independent (gen-time); the
                # `dual_type(...) === NoDual` test goes in the returned expression, not the body.
                fieldcount(P) == 0 && return :($$(QuoteNode(factory))(w, x))
                seeds = map(1:fieldcount(P)) do i
                    nm = QuoteNode(fieldnames(P)[i])
                    _seed_field_expr(
                        N, P, i, :($$(QuoteNode(internal))(w, getfield(x, $nm), d))
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
        end
    end
    @eval begin
        # Assert cache hits to the concrete V or IdDict{Any,Any} poisons inference.
        # Numeric elements cannot cycle; their blocks still share the backing Memory's V.
        function $internal(
            w::Val{N}, $(rng_args...), x::Array{<:NDualEltype}, d::MaybeCache
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
                        w, x, $internal(w, $(rng_vals...), getfield(x, :ref).mem, d)
                    )
                end
            else
                v = _cached_array_dual(w, x, d, () -> $factory(w, $(rng_vals...), x))
            end
            d[x] = v
            return v
        end
        # Register before filling: nested arrays may alias or cycle. Thread the same cache
        # through children, unlike the deliberately cache-free factories.
        function $internal(w::Val{N}, $(rng_args...), x::Array, d::MaybeCache) where {N}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            shell = similar(x, eltype(dual_type(Val(N), typeof(x))))
            d[x] = shell
            @inbounds for i in eachindex(x)
                isassigned(x, i) && (shell[i] = $internal(w, $(rng_vals...), x[i], d))
            end
            return shell
        end
        # `Ref{P<:NDualEltype}` → `NDualRef` (scalar analogue of the `Array` branch): build the
        # wrapper directly and register by identity, so the generic struct recursion never re-lifts it.
        function $internal(
            w::Val{N}, $(rng_args...), x::Base.RefValue{P}, d::MaybeCache
        ) where {N,P<:NDualEltype}
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
        # Complex / Memory / MemoryRef have fields but their own canonical V (not a
        # structural lift), so delegate to the cache-free factory.
        $internal(w::Val{N}, $(rng_args...), z::Complex, ::MaybeCache) where {N} = $factory(
            w, $(rng_vals...), z
        )
    end
    @static if VERSION >= v"1.11-rc4"
        # With a cache, MemoryRef must share the backing Memory's partials. Leaf elements
        # window its block; aggregates reference its element-wise shell. NoCache owns a block.
        @eval function $internal(
            w::Val{N}, $(rng_args...), x::MemoryRef{E}, d::MaybeCache
        ) where {N,E<:NDualEltype}
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
            w::Val{N}, $(rng_args...), x::MemoryRef, d::MaybeCache
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
        # A `Memory` of LEAVES delegates to the block factory — nothing below it can alias. A
        # `Memory` of aggregates (a `Dict`'s `vals`, say) must thread `d` through its elements the
        # way the element-wise `Array` branch does, or two elements holding one array get
        # independent partials: registering `x` alone shares the container, not what is inside it.
        @eval function $internal(
            w::Val{N}, $(rng_args...), x::Memory, d::MaybeCache
        ) where {N}
            haskey(d, x) && return d[x]::dual_type(Val(N), typeof(x))
            if eltype(x) <: NDualEltype
                v = $factory(w, $(rng_vals...), x)
                d[x] = v
                return v
            end
            shell = Memory{eltype(dual_type(Val(N), typeof(x)))}(undef, length(x))
            d[x] = shell
            @inbounds for i in eachindex(x)
                isassigned(x, i) && (shell[i] = $internal(w, $(rng_vals...), x[i], d))
            end
            return shell
        end
    end
end

@generated function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, x::P, d::MaybeCache
) where {N,P}
    # `fieldcount(P) == 0` is world-independent (gen-time); the `dual_type(...) === NoDual`
    # test goes in the returned expression, not the body.
    fieldcount(P) == 0 && return :(randn_dual(w, rng, x))
    seeds = map(1:fieldcount(P)) do i
        nm = QuoteNode(fieldnames(P)[i])
        _seed_field_expr(N, P, i, :(_randn_dual_internal(w, rng, getfield(x, $nm), d)))
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
end

# Reseed lanes at slots[k] in tangent_dim order (complex: real then imaginary).
# Mutate array partials in place and rebuild immutable Vs; callers must use the result.
# Visit aliased arrays and cyclic mutable structs once, matching zero_lifted/tangent_dim.
@inline function basis_lifted!!(seed::Lifted{P,N,V}, slots::NTuple{N,Int}) where {P,N,V}
    # An isbits V has no arrays or mutable wrappers, so there is nothing to mutate in place and
    # nothing to dedup: thread a plain `Int` cursor and rebuild on the stack, skipping the
    # `Ref`/`IdDict` the general path allocates. This keeps forward seeding of scalar/tuple/
    # NamedTuple/struct-of-scalar inputs allocation-free (`isbitstype(V)` folds at compile time).
    if isbitstype(V)
        v, _ = _basis_seed_isbits(seed.rep, slots, 0)
        return Lifted{P,N}(primal(seed), v)
    end
    v = _basis_seed!!(seed.rep, slots, Ref(0), IdDict{Any,Any}())
    return Lifted{P,N}(primal(seed), v)
end

# Pure-functional isbits reseed: returns `(rebuilt V, advanced cursor)`, threading the global
# scalar-dimension cursor by value. Mirrors `_basis_seed!!` for the isbits (array-free) shapes only.
#
# Both must advance the cursor in exactly the order `tangent_dim` counts — one step per real element,
# real-then-imag for a complex one. A mismatch misplaces gradient entries and never errors.
@inline _basis_seed_isbits(::NoDual, _slots::NTuple{N,Int}, c::Int) where {N} = (
    NoDual(), c
)
# Ptr lanes are non-addressable placeholders with zero tangent_dim. Leave them unchanged
# and do not advance the cursor, including when reached through an NTuple V.
@inline _basis_seed_isbits(v::Ptr, _slots::NTuple{N,Int}, c::Int) where {N} = (v, c)
@inline function _basis_seed_isbits(v::NDual{T,N}, slots::NTuple{N,Int}, c::Int) where {T,N}
    c += 1
    return (NDual{T,N}(v.value, ntuple(k -> c == slots[k] ? one(T) : zero(T), Val(N))), c)
end
@inline function _basis_seed_isbits(
    v::Complex{NDual{T,N}}, slots::NTuple{N,Int}, c::Int
) where {T,N}
    re, c = _basis_seed_isbits(real(v), slots, c)
    im, c = _basis_seed_isbits(imag(v), slots, c)
    return (Complex(re, im), c)
end
@inline _basis_seed_isbits(::Tuple{}, _slots::NTuple{N,Int}, c::Int) where {N} = ((), c)
@inline function _basis_seed_isbits(v::Tuple, slots::NTuple{N,Int}, c::Int) where {N}
    h, c = _basis_seed_isbits(first(v), slots, c)
    t, c = _basis_seed_isbits(Base.tail(v), slots, c)
    return ((h, t...), c)
end
@inline function _basis_seed_isbits(
    v::NamedTuple{names}, slots::NTuple{N,Int}, c::Int
) where {names,N}
    t, c = _basis_seed_isbits(values(v), slots, c)
    # `typeof(v)`, not `NamedTuple{names}`: the latter re-derives each field type from the REBUILT
    # value, narrowing an `Any`-declared backing field to the concrete seed's type. `zero_lifted`
    # coerces into the declared backing NamedTuple, so re-deriving here breaks
    # `V === dual_type(Val(N), P)` and the OpaqueClosure argument typeassert rejects the slot.
    return (typeof(v)(t), c)
end
@inline function _basis_seed_isbits(
    v::ImmutableDual, slots::NTuple{N,Int}, c::Int
) where {N}
    inner, c = _basis_seed_isbits(v.fields, slots, c)
    return (ImmutableDual(inner), c)
end
@inline function _basis_seed_isbits(
    v::PossiblyUninitTangent, slots::NTuple{N,Int}, c::Int
) where {N}
    is_init(v) || return (v, c)
    inner, c = _basis_seed_isbits(val(v), slots, c)
    return (typeof(v)(inner), c)
end

_basis_seed!!(::NoDual, _slots, _cursor, _dict) = NoDual()
# `Ptr` lane: 0-dimension placeholder (see the isbits terminal above) — return it unchanged and
# do not advance the cursor.
_basis_seed!!(v::Ptr, _slots, _cursor, _dict) = v
function _basis_seed!!(v::NDual{T,N}, slots::NTuple{N,Int}, cursor, _dict) where {T,N}
    cursor[] += 1
    c = cursor[]
    return NDual{T,N}(v.value, ntuple(k -> c == slots[k] ? one(T) : zero(T), Val(N)))
end
function _basis_seed!!(
    v::Complex{NDual{T,N}}, slots::NTuple{N,Int}, cursor, dict
) where {T,N}
    re = _basis_seed!!(real(v), slots, cursor, dict)
    im = _basis_seed!!(imag(v), slots, cursor, dict)
    return Complex(re, im)
end
# The block storage's whole allocation: a windowed block's flat storage is a fresh `Array` header
# over a shared `Memory`, so the `Memory` is what two windows have in common. Before 1.11, and for
# storage that is not an `Array` (the CUDA extension's `CuArray` block), the storage is itself the
# allocation.
@inline _partials_allocation(store) = store
@static if VERSION >= v"1.11-rc4"
    @inline _partials_allocation(store::Array) = getfield(store, :ref).mem
end

# Claim and clear the whole allocation once, then let each container write its hot lanes.
# V identity misses shared windows; (address, length) misses nested windows with capacity
# slack, allowing a later clear to erase an earlier container's hot lane.
@inline function _clear_partials_store!(dict, block, z)
    store = _partials_allocation(Nfwd._block_storage(block))
    haskey(dict, store) && return nothing
    dict[store] = nothing
    fill!(store, z)
    return nothing
end

# tangent_dim dedups Memory and MemoryRef through the primal Memory; NDualMemoryRef's
# flattened V does not expose that sharing. Claim the primal after clearing each block.
# Allocation identity is insufficient: Array and Memory can share allocation but differ
# in reverse container tangents. Return true when the primal was already seen.
@inline _memory_seen!(_dict, _primal) = false
@static if VERSION >= v"1.11-rc4"
    @inline function _memory_seen!(dict, mem::Memory)
        haskey(dict, mem) && return true
        dict[mem] = nothing
        return false
    end
    @inline _memory_seen!(dict, p::MemoryRef) = _memory_seen!(dict, p.mem)
end

function _basis_seed!!(
    v::NDualArray{T,N}, slots::NTuple{N,Int}, cursor, dict
) where {T<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    _clear_partials_store!(dict, getfield(v, :partials_block), zero(T))
    _memory_seen!(dict, v.primal) && return v
    parts = Nfwd._lane_views(v)
    @inbounds for idx in eachindex(v.primal)
        cursor[] += 1
        c = cursor[]
        for k in 1:N
            c == slots[k] && (parts[k][idx] = one(T))
        end
    end
    return v
end
function _basis_seed!!(
    v::NDualArray{Complex{R},N}, slots::NTuple{N,Int}, cursor, dict
) where {R<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    _clear_partials_store!(dict, getfield(v, :partials_block), zero(Complex{R}))
    _memory_seen!(dict, v.primal) && return v
    parts = Nfwd._lane_views(v)
    @inbounds for idx in eachindex(v.primal)
        cursor[] += 1
        cr = cursor[]
        cursor[] += 1
        ci = cursor[]
        for k in 1:N
            cr == slots[k] && (parts[k][idx] = Complex(one(R), zero(R)))
            ci == slots[k] && (parts[k][idx] = Complex(zero(R), one(R)))
        end
    end
    return v
end
# Element-wise V (plain `Array` of inner duals, e.g. an abstract-element array):
# rebuild each element in place.
function _basis_seed!!(v::Array, slots::NTuple{N,Int}, cursor, dict) where {N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    @inbounds for i in eachindex(v)
        isassigned(v, i) && (v[i] = _basis_seed!!(v[i], slots, cursor, dict))
    end
    return v
end
# IdDict values must follow backing ht slot order, as tangent_dim does. Do not generalise
# to unvetted struct Vs: missing methods must fail loudly.
function _basis_seed!!(v::IdDict, slots::NTuple{N,Int}, cursor, dict) where {N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    # Values only: the keys share the backing `ht` with them but carry no derivative, and `tangent_dim`
    # scores them 0. `IdDict` iteration walks `ht` in slot order, which is the order `tangent_dim` counts,
    # so the cursor advances in step. Keys collected first — the loop assigns into `v`.
    for k in collect(keys(v))
        v[k] = _basis_seed!!(v[k], slots, cursor, dict)
    end
    return v
end
# A bare `Memory` reaches here as the backing of a lifted `Dict`/`Set`, whose `slots` and `keys`
# fields carry no derivative while `vals` does. Same element-wise walk as the `Array` above.
@static if VERSION >= v"1.11-rc4"
    function _basis_seed!!(v::Memory, slots::NTuple{N,Int}, cursor, dict) where {N}
        haskey(dict, v) && return dict[v]
        dict[v] = v
        @inbounds for i in eachindex(v)
            isassigned(v, i) && (v[i] = _basis_seed!!(v[i], slots, cursor, dict))
        end
        return v
    end
end
function _basis_seed!!(v::Tuple, slots::NTuple{N,Int}, cursor, dict) where {N}
    return map(e -> _basis_seed!!(e, slots, cursor, dict), v)
end
function _basis_seed!!(
    v::NamedTuple{names}, slots::NTuple{N,Int}, cursor, dict
) where {names,N}
    # `typeof(v)`, as in `_basis_seed_isbits` above and for the same reason.
    return typeof(v)(map(e -> _basis_seed!!(e, slots, cursor, dict), values(v)))
end
function _basis_seed!!(
    v::PossiblyUninitTangent, slots::NTuple{N,Int}, cursor, dict
) where {N}
    is_init(v) || return v
    return typeof(v)(_basis_seed!!(val(v), slots, cursor, dict))
end
# `Ref{P}` forward V: one scalar dimension held in a mutable `Base.RefValue` partials shadow. Mirror the
# `NDual` scalar method (real: one cursor step; complex: two, real then imag), and register in `dict`
# so an aliased `Ref` seeds once — like `NDualArray`/`MutableDual`.
function _basis_seed!!(
    v::NDualRef{P,N}, slots::NTuple{N,Int}, cursor, dict
) where {P<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    cursor[] += 1
    c = cursor[]
    v.partials[] = ntuple(k -> c == slots[k] ? one(P) : zero(P), Val(N))
    return v
end
function _basis_seed!!(
    v::NDualRef{Complex{R},N}, slots::NTuple{N,Int}, cursor, dict
) where {R<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    cursor[] += 1
    cr = cursor[]
    cursor[] += 1
    ci = cursor[]
    v.partials[] = ntuple(
        k -> Complex(cr == slots[k] ? one(R) : zero(R), ci == slots[k] ? one(R) : zero(R)),
        Val(N),
    )
    return v
end
function _basis_seed!!(v::ImmutableDual, slots::NTuple{N,Int}, cursor, dict) where {N}
    return ImmutableDual(_basis_seed!!(v.fields, slots, cursor, dict))
end
function _basis_seed!!(v::MutableDual, slots::NTuple{N,Int}, cursor, dict) where {N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    v.fields = _basis_seed!!(v.fields, slots, cursor, dict)
    return v
end
# Factory-built NDualMemoryRefs cover the whole Memory (column j ↔ slot j), which
# tangent_dim walks. Seed each column, real then imaginary for complex elements, and dedup.
@static if VERSION >= v"1.11-rc4"
    function _basis_seed!!(
        v::NDualMemoryRef{T,N}, slots::NTuple{N,Int}, cursor, dict
    ) where {T<:IEEEFloat,N}
        haskey(dict, v) && return dict[v]
        dict[v] = v
        block = Nfwd._reconstruct_block(v)
        _clear_partials_store!(dict, block, zero(T))
        _memory_seen!(dict, v.primal) && return v
        @inbounds for idx in 1:size(block, 2)
            cursor[] += 1
            c = cursor[]
            for k in 1:N
                c == slots[k] && (block[k, idx] = one(T))
            end
        end
        return v
    end
    function _basis_seed!!(
        v::NDualMemoryRef{Complex{R},N}, slots::NTuple{N,Int}, cursor, dict
    ) where {R<:IEEEFloat,N}
        haskey(dict, v) && return dict[v]
        dict[v] = v
        block = Nfwd._reconstruct_block(v)
        _clear_partials_store!(dict, block, zero(Complex{R}))
        _memory_seen!(dict, v.primal) && return v
        @inbounds for idx in 1:size(block, 2)
            cursor[] += 1
            cr = cursor[]
            cursor[] += 1
            ci = cursor[]
            for k in 1:N
                cr == slots[k] && (block[k, idx] = Complex(one(R), zero(R)))
                ci == slots[k] && (block[k, idx] = Complex(zero(R), one(R)))
            end
        end
        return v
    end
end

# Width-1 compatibility entry points return Lifted slots.
@inline zero_dual(x) = zero_lifted(Val(1), x)
@inline uninit_dual(x) = uninit_lifted(Val(1), x)
@inline randn_dual(rng::AbstractRNG, x) = randn_lifted(Val(1), rng, x)

# ── MemoryRef seed factories (Julia 1.11+) ──────────────────────────────────
#
# `zero_dual` is the canonical MemoryRef seed factory (bits-element dense zero-init).

@static if VERSION >= v"1.11-rc4"
    # Numeric Memory / MemoryRef seeds must stay more specific than element-wise factories.
    # MemoryRef blocks cover the whole backing Memory, with the ref's column at its offset.
    @inline function zero_dual(::Val{N}, p::MemoryRef{E}) where {N,E<:NDualEltype}
        return NDualMemoryRef{E,N,Memory{E}}(p)
    end
    @inline function uninit_dual(::Val{N}, p::MemoryRef{E}) where {N,E<:NDualEltype}
        return NDualMemoryRef{E,N,Memory{E}}(
            p, NDualBlock{E,2}(undef, N, length(p.mem)), Core.memoryrefoffset(p)
        )
    end
    @inline function randn_dual(
        ::Val{N}, rng::AbstractRNG, p::MemoryRef{E}
    ) where {N,E<:NDualEltype}
        len = length(p.mem)
        block = NDualBlock{E,2}(randn(rng, E, N * len), (N, len))
        return NDualMemoryRef{E,N,Memory{E}}(p, block, Core.memoryrefoffset(p))
    end
    @inline function zero_dual(::Val{N}, m::Memory{E}) where {N,E<:NDualEltype}
        return NDualArray{E,N,1,Memory{E}}(m)
    end
    @inline function uninit_dual(::Val{N}, m::Memory{E}) where {N,E<:NDualEltype}
        return NDualArray{E,N,1,Memory{E}}(m, NDualBlock{E,2}(undef, N, length(m)))
    end
    @inline function randn_dual(
        ::Val{N}, rng::AbstractRNG, m::Memory{E}
    ) where {N,E<:NDualEltype}
        return NDualArray{E,N,1,Memory{E}}(
            m, ntuple(_ -> Memory{E}(randn(rng, E, length(m))), Val(N))
        )
    end
    # Non-float Memory seeds recurse element-wise. Plain functions keep extension dual_type
    # overloads visible at the caller's world; foldability preserves concrete element types.
    for factory in (:zero_dual, :uninit_dual, :randn_dual)
        rng_args = factory === :randn_dual ? (:(rng::AbstractRNG),) : ()
        rng_vals = factory === :randn_dual ? (:rng,) : ()
        @eval @inline function $factory(::Val{N}, $(rng_args...), m::Memory{T}) where {N,T}
            v = Memory{dual_type(Val(N), T)}(undef, length(m))
            @inbounds for i in eachindex(m)
                isassigned(m, i) && (v[i] = $factory(Val(N), $(rng_vals...), m[i]))
            end
            return v
        end
    end
    @inline function zero_dual(::Val{N}, p::MemoryRef{T}) where {N,T}
        return _memoryref_at(zero_dual(Val(N), p.mem), Core.memoryrefoffset(p))
    end
    # `MemoryRef`'s V is built via `memoryref` over the `.mem`'s V (a plain
    # `MemoryRef` can't be constructed field-wise from a raw `Ptr`), so mirror
    # `zero_dual` rather than fall through to the generic struct seed.
    @inline function uninit_dual(::Val{N}, p::MemoryRef{T}) where {N,T}
        return _memoryref_at(uninit_dual(Val(N), p.mem), Core.memoryrefoffset(p))
    end
    @inline function randn_dual(::Val{N}, rng::AbstractRNG, p::MemoryRef{T}) where {N,T}
        return _memoryref_at(randn_dual(Val(N), rng, p.mem), Core.memoryrefoffset(p))
    end
end
