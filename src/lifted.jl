# Forward-mode slot wrapper `Lifted{P, N, V}` and its associated width-N
# `dual_type` / `lifted_type` queries and seed factories. Loaded after
# `nfwd/Nfwd.jl` so the `NDual{T, N}` IEEEFloat carrier is in scope.
#
# Design note (forward vs reverse). The forward V is *type-precise*: `V ===
# dual_type(P)` mirrors the primal, so differentiability (`NDual` vs `NoDual`),
# wrapper nesting, and array layout (SoA `NDualArray` vs AoS) are each distinct
# *types*. Reverse (`CoDual`) is *runtime-uniform* — non-differentiability is the
# value `NoFData`, and wrappers are flattened by a runtime `arrayify` — so one
# `rrule!!` subsumes shapes that a `frule!!` must enumerate as separate methods
# (e.g. a `NoDual`-V overload beside the `NDual`-V one; one signature per wrapper
# backing). Reverse is therefore both the TEMPLATE (broaden a forward frule to its
# rrule sibling's breadth) and the ORACLE (a forward-fail / reverse-pass on the
# same input flags a real bug — including convention slips the type system cannot
# catch, e.g. a dropped `Symmetric` factor-of-2). When porting a frule from an
# rrule, replicate the *convention*, not just the formula.

"""
    ImmutableDual{T<:NamedTuple}

Single-field immutable wrapper used as the canonical V for *immutable struct*
primals under the forward-mode structural lift. Its `value::T` field holds
the recursive `NamedTuple{fieldnames(P), Tuple{V_i...}}` of canonical field
Vs, where each `V_i = dual_type(Val(N), fieldtype(P, i))`.
"""
struct ImmutableDual{T<:NamedTuple}
    value::T
end

Base.:(==)(x::ImmutableDual, y::ImmutableDual) = x.value == y.value

"""
    MutableDual{T<:NamedTuple}

Mutable counterpart to `ImmutableDual`. Mutability is load-bearing for the
`MutableDualTangentView` proxy that writes back to `value` via `setfield!`.
"""
mutable struct MutableDual{T<:NamedTuple}
    value::T
    # Uninitialised form, used by the cyclic-struct `lift` to register a shell
    # in the aliasing cache before its fields (which may reference back to it)
    # are built. Mirrors reverse-mode `MutableTangent()`.
    MutableDual{T}() where {T<:NamedTuple} = new{T}()
    MutableDual{T}(value) where {T<:NamedTuple} = new{T}(value)
end
@inline MutableDual(value::T) where {T<:NamedTuple} = MutableDual{T}(value)

Base.:(==)(x::MutableDual, y::MutableDual) = x.value == y.value

"""
    Lifted{P, N, V}

Forward-mode slot wrapper for a primal value of type `P` and its canonical
`N`-width forward representation `V`. Two fields:

- `primal::P` — slot-level back-reference to the user's primal value. For
  mutable struct primals this aliases user storage; for immutable primals
  it carries the same value.
- `value::V` — the canonical `N`-width forward representation. For
  concrete runtime wrappers `V === dual_type(Val(N), P)`.

Rules dispatch on `Lifted{P, N}` (V left abstract) and use `primal`,
`tangent`, and per-lane extractors to access tangent data. Outputs are
constructed via `Lifted{P_out, N}(primal_out, value_out)`.

Width `N == 1` is ordinary forward mode; `N >= 2` is chunked forward
mode. `Lifted` never nests inside another `Lifted`'s `V`; only `Val(0)`
slots are unwrapped (primal passthrough).
"""
struct Lifted{P,N,V}
    primal::P
    value::V
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

# Two-argument constructor with V inferred from typeof(value). The
# canonical wrapping path: `value` is already a built inner V of the
# correct shape; this overload just wraps the (primal, value) pair.
@inline function Lifted{P,N}(primal::P, value::V) where {P,N,V}
    return Lifted{P,N,V}(primal, value)
end

# Sharpen `P` when constructing with a `Type{X}` primal, at the boundary
# between widened (e.g. `DataType`) and concrete (`Type{X}`) primals.
# Without this, `Lifted{DataType, N}(ComplexF64, NoDual())` would
# produce `Lifted{DataType, N, NoDual}` and miss `frule!!` rules
# dispatched on `Lifted{Type{Complex{P}}, N}`.
@inline function Lifted{P_user,N}(
    primal::Type{P_inner}, value::V
) where {P_user,P_inner,N,V}
    return Lifted{Type{P_inner},N,V}(primal, value)
end

# Accessors — mirror the `CoDual` API.
primal(d::Lifted) = d.primal
tangent(d::Lifted) = d.value
# `_primal` extracts a primal value from a forward-mode slot; the generic fallback returns
# the value unchanged. The interpreter's dual IR calls `_primal` on each operand, which is
# either a `Lifted` slot or a lifted constant.
_primal(x) = x
_primal(x::Lifted) = primal(x)

# Forward-mode equivalent of `verify_dual_type` — checks the slot's `V` is
# compatible with `dual_type(Val(N), P)`. Used by the test framework.
# Returns `true` for any well-formed Lifted slot; specific V-shape checks
# happen at construction time via the V's invariants.
verify_dual_type(::Lifted) = true

"""
    extract(d::Lifted) -> (primal, value)

Helper that returns the `(primal(d), tangent(d))` tuple. Mirrors
`extract(::Dual)` and `extract(::CoDual)` for symmetric ergonomics.
"""
extract(d::Lifted) = (primal(d), tangent(d))

function Base.copy(d::Lifted{P,N,V}) where {P,N,V}
    return Lifted{P,N,V}(copy(primal(d)), copy(tangent(d)))
end

# Lifted slots are safe to share by reference within a forward pass —
# slot-local tangent storage rules out cross-slot aliasing hazards.
_copy(d::Lifted) = d

function Base.:(==)(a::Lifted, b::Lifted)
    return primal(a) == primal(b) && tangent(a) == tangent(b)
end

# `NDualArray` / `NDualMemoryRef` (and the `NDualEltype` constant) now
# live in `src/nfwd/Nfwd.jl` and are re-exported into Mooncake via
# `using .Nfwd: NDualArray, NDualMemoryRef, NDualEltype` in `src/Mooncake.jl`.
# The Mooncake-namespace method extensions for these types are below.

# Whole-array accessors — O(1) by aliasing.
@inline primal(a::NDualArray) = a.primal
@inline tangent(a::NDualArray) = a.partials
@inline unpack_ndual(a::NDualArray) = (a.primal, a.partials)

# Per-lane native-tangent accessor. The `Lifted{MutS, N, <:MutableDual}` overload
# at the bottom of this file returns a `MutableDualTangentView` proxy so rule
# bodies can write through it; `unlift` has its own MutableDual path that
# materialises a fresh `MutableTangent` instead.
# Leaf V accessors key on the V shape and leave `P` free: the inner V uniquely
# determines extraction, and an abstract slot (e.g. `Lifted{Real, N, NDual{Float64, N}}`,
# where the static primal type is abstract but the runtime V is concrete) must
# still resolve here.
@inline function tangent(x::Lifted{P,N,NDual{T,N}}, lane::Integer) where {P,T<:IEEEFloat,N}
    return tangent(x).partials[lane]
end
@inline function tangent(x::Lifted{P,N,<:NDualArray}, lane::Integer) where {P,N}
    return tangent(x).partials[lane]
end
# `NDualMemoryRef` / `MemoryRef` / `Core.memoryref` are 1.11+; gate to avoid an
# `UndefVarError` at precompile on 1.10.
@static if VERSION >= v"1.11-rc4"
    @inline function tangent(x::Lifted{P,N,<:NDualMemoryRef}, lane::Integer) where {P,N}
        return tangent(x).partials[lane]
    end
    # `Core.memoryref(mem, offset)` is out-of-bounds for an empty `mem` (length 0); the 1-arg
    # `Core.memoryref(mem)` gives the canonical empty ref. Empty `Memory` is a valid primal
    # (empty arrays/vectors), so re-build a `MemoryRef` V through this guard.
    @inline _memoryref_at(mem, offset) =
        length(mem) == 0 ? Core.memoryref(mem) : Core.memoryref(mem, offset)
    # AoS `MemoryRef` V (a plain `MemoryRef` into an AoS V `Memory`, e.g. `MemoryRef{NoDual}`
    # for `MemoryRef{Int}`, or `MemoryRef{NDualArray}` for `MemoryRef{Vector}` comms). Project
    # the lane through the `.mem`'s AoS-array lane accessor, then re-`memoryref` at the offset.
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
@inline tangent(::Lifted{P,N,NoDual}, ::Integer) where {P,N} = NoTangent()
# Element-wise non-differentiable array V (`Array{NoDual}`, e.g. `Vector{Int}`): the lane tangent
# mirrors reverse `tangent_type(Array{T,D}) === Array{NoTangent,D}` — a same-shape `NoTangent`
# array, the element-wise analogue of the whole-`NoDual` case above.
@inline function tangent(x::Lifted{P,N,<:AbstractArray{NoDual}}, ::Integer) where {P,N}
    return map(_ -> NoTangent(), tangent(x))
end
# General AoS container V — a plain `Array`/`Memory` of per-element forward Vs (e.g.
# `Array{NDualArray}` from a nested array, or `Memory{…}` forward-over-reverse comms).
# Project lane `k` element-wise to the same-shape array of each element's lane-`k`
# tangent, mirroring reverse `tangent_type(Array{T}) === Array{tangent_type(T)}`. The SoA
# `NDualArray` (also `<:AbstractArray`) and the `AbstractArray{NoDual}` case have more
# specific overloads above; undefined slots stay undefined (reverse-PUT semantics).
@inline function tangent(x::Lifted{P,N,V}, lane::Integer) where {P,N,V<:AbstractArray}
    p = primal(x)
    v = tangent(x)
    t = similar(p, tangent_type(eltype(P)))
    _map_if_assigned!((pe, ve) -> tangent(Lifted{eltype(P),N}(pe, ve), lane), t, p, v)
    return t
end
@inline function tangent(x::Lifted{P,N,<:Tuple}, lane::Integer) where {P,N}
    return tangent(x)[lane]
end
# A `PossiblyUninitTangent` backing field reproduces the reverse `Tangent`'s
# PUT shape: an undefined primal field maps to an uninit reverse PUT.
@inline _field_lane_tangent(::Val{N}, ::Type{P}, p, name, vfield, lane) where {N,P} = tangent(
    Lifted{fieldtype(P, name),N}(getfield(p, name), vfield), lane
)
@inline function _field_lane_tangent(
    ::Val{N}, ::Type{P}, p, name, vfield::PossiblyUninitTangent, lane
) where {N,P}
    Rt = tangent_type(fieldtype(P, name))
    (is_init(vfield) && isdefined(p, name)) || return PossiblyUninitTangent{Rt}()
    return PossiblyUninitTangent{Rt}(
        tangent(Lifted{fieldtype(P, name),N}(getfield(p, name), val(vfield)), lane)
    )
end
@inline function tangent(x::Lifted{P,N,<:ImmutableDual}, lane::Integer) where {P,N}
    nt = tangent(x).value
    p = primal(x)
    names = keys(nt)
    field_tangents = map(names) do name
        return _field_lane_tangent(Val(N), P, p, name, getfield(nt, name), lane)
    end
    # Coerce into the declared reverse backing `tangent_type(P)`, so an abstract
    # field is stored at its widened type (e.g. `a::Any`, not the concrete
    # `a::Float64`) — matching what reverse mode produces.
    backing = fieldtype(tangent_type(P), :fields)
    return Tangent(backing(field_tangents))
end
# Tuple primal: V is a Tuple of per-element V; recurse element-wise.
@inline function tangent(x::Lifted{P,N,<:Tuple}, lane::Integer) where {P<:Tuple,N}
    p = primal(x)
    v = tangent(x)
    return ntuple(length(v)) do i
        return tangent(Lifted{fieldtype(P, i),N}(p[i], v[i]), lane)
    end
end
# NamedTuple primal: V is a NamedTuple of per-element V; recurse element-wise.
@inline function tangent(x::Lifted{P,N,<:NamedTuple}, lane::Integer) where {P<:NamedTuple,N}
    p = primal(x)
    v = tangent(x)
    names = keys(v)
    field_tangents = map(names) do name
        return tangent(Lifted{fieldtype(P, name),N}(getfield(p, name), getfield(v, name)), lane)
    end
    return NamedTuple{names}(field_tangents)
end

# Public 2-tuple unpack at the slot boundary. Width-1 only — chunked slots
# carry per-lane derivatives in their V and have no single native-tangent
# unpack; use per-lane access (`tangent(x, lane)`) for width N > 1.
@inline unlift(x::Lifted{P,1}) where {P} = (primal(x), tangent(x, 1))
# Aggregate slots (struct / tuple / named-tuple) need a fresh reverse tangent (not the
# `MutableDualTangentView` write proxy `tangent(x, 1)` returns for mutable structs), and
# may be self-referential (`node.next === node`) or hold aliased mutables. `_unlift_seed`
# threads an aliasing cache (keyed on the primal) and registers a `MutableTangent` shell
# before recursing, so a cycle reaching the same primal returns the shell — mirroring
# reverse `zero_tangent_internal`. Leaf slots (scalar / array / complex) have no cycle and
# convert via the lane accessor (aliased arrays already share one `partials` array).
@inline unlift(x::Lifted{P,1,<:Union{MutableDual,ImmutableDual,Tuple,NamedTuple}}) where {P} = (
    primal(x), _unlift_seed(x, IdDict{Any,Any}())
)

@inline _unlift_seed(x::Lifted{P,1}, ::IdDict) where {P} = tangent(x, 1)
function _unlift_seed(x::Lifted{P,1,<:MutableDual}, cache::IdDict) where {P}
    p = primal(x)
    haskey(cache, p) && return cache[p]
    Tt = tangent_type(P)
    shell = Tt()
    cache[p] = shell
    nt = tangent(x).value
    field_tangents = map(keys(nt)) do name
        return _field_unlift_seed(P, p, name, getfield(nt, name), cache)
    end
    # Coerce into the declared reverse backing `tangent_type(P)` (an abstract field is
    # stored widened, e.g. `a::Any`, matching reverse mode).
    shell.fields = fieldtype(Tt, :fields)(field_tangents)
    return shell
end
function _unlift_seed(x::Lifted{P,1,<:ImmutableDual}, cache::IdDict) where {P}
    nt = tangent(x).value
    p = primal(x)
    field_tangents = map(keys(nt)) do name
        return _field_unlift_seed(P, p, name, getfield(nt, name), cache)
    end
    return Tangent(fieldtype(tangent_type(P), :fields)(field_tangents))
end
function _unlift_seed(x::Lifted{P,1,<:Tuple}, cache::IdDict) where {P}
    p = primal(x)
    v = tangent(x)
    return ntuple(length(v)) do i
        return _unlift_seed(Lifted{fieldtype(P, i),1}(p[i], v[i]), cache)
    end
end
function _unlift_seed(x::Lifted{P,1,<:NamedTuple}, cache::IdDict) where {P}
    p = primal(x)
    v = tangent(x)
    names = keys(v)
    return NamedTuple{names}(
        map(names) do name
            return _unlift_seed(
                Lifted{fieldtype(P, name),1}(getfield(p, name), getfield(v, name)), cache
            )
        end,
    )
end
@inline _field_unlift_seed(::Type{P}, p, name, vfield, cache::IdDict) where {P} = _unlift_seed(
    Lifted{fieldtype(P, name),1}(getfield(p, name), vfield), cache
)
@inline function _field_unlift_seed(
    ::Type{P}, p, name, vfield::PossiblyUninitTangent, cache::IdDict
) where {P}
    Rt = tangent_type(fieldtype(P, name))
    (is_init(vfield) && isdefined(p, name)) || return PossiblyUninitTangent{Rt}()
    return PossiblyUninitTangent{Rt}(
        _unlift_seed(Lifted{fieldtype(P, name),1}(getfield(p, name), val(vfield)), cache)
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

# `_dot_internal` / `_scale_internal` overloads for forward-mode V
# shapes that the test framework's tangent-shape arithmetic may see
# when it operates on raw Lifted V values (e.g. `tangent(y_ẏ_a)` in
# `test_frule_reuse`).
_dot_internal(::MaybeCache, ::NoDual, ::NoDual) = 0.0
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:ImmutableDual}
    return _dot_internal(c, t.value, s.value)::Float64
end
# A `MutableDual` may be self-referential; cache the pair to break the cycle
# (mirrors reverse-mode `_dot_internal(::MaybeCache, ::MutableTangent, …)`).
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:MutableDual}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return _dot_internal(c, t.value, s.value)::Float64
end
# Scalar NDual (forward-mode width-1 V for IEEEFloat) — sum the partials' dot.
function _dot_internal(::MaybeCache, t::NDual{T,N}, s::NDual{T,N}) where {T<:IEEEFloat,N}
    return Float64(sum(map(*, t.partials, s.partials); init=zero(T)))
end

_scale_internal(::MaybeCache, ::Float64, ::NoDual) = NoDual()
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:ImmutableDual}
    return T(_scale_internal(c, a, t.value))
end
# Register an uninitialised result before recursing so a self-referential
# `MutableDual` terminates (mirrors reverse-mode `_scale_internal` for `MutableTangent`).
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MutableDual}
    haskey(c, t) && return c[t]::T
    y = T()
    c[t] = y
    y.value = _scale_internal(c, a, t.value)
    return y
end
# Scalar NDual scale — scale `.value` and each lane.
function _scale_internal(::MaybeCache, a::Float64, t::NDual{T,N}) where {T<:IEEEFloat,N}
    aT = T(a)
    return NDual{T,N}(aT * t.value, map(p -> aT * p, t.partials))
end

_add_to_primal_internal(::MaybeCache, x, ::NoDual, ::Bool) = x
# Scalar NDual: add `.value + sum(.partials)` (matches the test-framework's
# usage where it treats NDual.value as the "primal-side" content).
function _add_to_primal_internal(
    ::MaybeCache, x::T, t::NDual{T,N}, ::Bool
) where {T<:IEEEFloat,N}
    return x + t.value + sum(t.partials; init=zero(T))
end
function _add_to_primal_internal(c::MaybeCache, x, t::ImmutableDual, unsafe::Bool)
    # The V wraps a NamedTuple of per-field Vs; reconstruct `x` by adding
    # each field's V back to the corresponding primal field. This mirrors
    # what `_add_to_primal_internal(::MaybeCache, x, ::Tangent, ::Bool)`
    # does for reverse-mode tangents in src/tangents/tangents.jl.
    return _add_to_primal_internal_struct(c, x, t.value, unsafe)
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
    nt = t.value
    init_fields = map(fieldnames(P)) do name
        return if isconst(P, name)
            _add_to_primal_internal(c, getfield(x, name), getfield(nt, name), unsafe)
        else
            getfield(x, name)
        end
    end
    p′ = _new_(P, init_fields...)::P
    c[key] = p′
    for name in fieldnames(P)
        isconst(P, name) || setfield!(
            p′,
            name,
            _add_to_primal_internal(c, getfield(x, name), getfield(nt, name), unsafe),
        )
    end
    return p′
end
@unstable function _add_to_primal_internal_struct(c, x, nt::NamedTuple, unsafe)
    isempty(propertynames(nt)) && return x
    names = keys(nt)
    new_fields = map(names) do name
        return _add_to_primal_internal(c, getfield(x, name), getfield(nt, name), unsafe)
    end
    # Rebuild via _new_ to handle both mutable and immutable structs.
    return _new_(typeof(x), new_fields...)
end

# `NDualMemoryRef` (and its constructor) now lives in `src/nfwd/Nfwd.jl`.
# Mooncake-namespace method extensions follow.

@static if VERSION >= v"1.11-rc4"
    @inline primal(a::NDualMemoryRef) = a.primal
    @inline tangent(a::NDualMemoryRef) = a.partials
    @inline unpack_ndual(a::NDualMemoryRef) = (a.primal, a.partials)

    # Element access via Core.memoryref* — `MemoryRef` is not AbstractArray.
    @inline function _memoryrefget_ndual(
        a::NDualMemoryRef{Element,N}, order::Symbol, boundscheck::Bool
    ) where {Element<:IEEEFloat,N}
        v = Core.memoryrefget(a.primal, order, boundscheck)
        parts = ntuple(k -> Core.memoryrefget(a.partials[k], order, boundscheck), Val(N))
        return NDual{Element,N}(v, parts)
    end
end

# ──────────────────────────────────────────────────────────────────────────
# `MutableDualTangentView{SD, P}` — per-lane proxy view for mutable struct
# slots (dual-types.md §13.6). The view is an immutable struct with three
# fields:
#
#   parent::SD  — the underlying `MutableDual` (writeback target).
#   primal::P   — back-reference to the slot's primal struct.
#   lane::Int   — which lane this view refers to.
#
# `getproperty` reads from the parent's NamedTuple and extracts the lane;
# `setproperty!` writes the lane back to the parent via `setfield!`. This
# enables `view.field = x` to mutate the slot's V from within a forward-mode
# rule body.
#
# This initial commit supports V_i = `NDual{T, N}` (scalar IEEEFloat field)
# only. Other V_i shapes (NDualArray, Complex{NDual}, nested MutableDual,
# PossiblyUninitTangent) are added in follow-up commits.
# ──────────────────────────────────────────────────────────────────────────

struct MutableDualTangentView{SD<:MutableDual,P}
    parent::SD
    primal::P
    lane::Int
end

# Lane-extraction (read) and lane-replacement (write) for individual V_i shapes.
# Add new V_i methods as additional shapes (NDualArray, Complex{NDual}, …) come
# online for mutable-struct field tangents.
@inline _lane_tangent(v::NDual, lane::Int) = v.partials[lane]

@inline function _replace_lane_tangent(v::NDual{T,N}, lane::Int, x::T) where {T,N}
    new_partials = ntuple(k -> k == lane ? x : v.partials[k], Val(N))
    return NDual{T,N}(v.value, new_partials)
end

function Base.getproperty(v::MutableDualTangentView, name::Symbol)
    name in (:parent, :primal, :lane) && return getfield(v, name)
    nt = getfield(v, :parent).value
    return _lane_tangent(getfield(nt, name), getfield(v, :lane))
end

function Base.setproperty!(v::MutableDualTangentView, name::Symbol, x)
    parent = getfield(v, :parent)
    lane = getfield(v, :lane)
    nt = parent.value
    new_V_i = _replace_lane_tangent(getfield(nt, name), lane, x)
    setfield!(parent, :value, merge(nt, NamedTuple{(name,)}((new_V_i,))))
    return x
end

# Per-lane tangent accessor on a `Lifted{MutS, N, <:MutableDual}` slot.
@inline function tangent(d::Lifted{MutS,N,<:MutableDual}, lane::Integer) where {MutS,N}
    return MutableDualTangentView{typeof(d.value),MutS}(d.value, d.primal, Int(lane))
end

# ──────────────────────────────────────────────────────────────────────────
# Width-N `dual_type` and `lifted_type` queries.
#
# `dual_type(Val(N), P)` returns the canonical inner V for a primal of
# type `P` at width `N` — i.e. the type of `tangent(d::Lifted{P, N})`'s
# payload, equal to the type of the slot's `value::V` field.
#
# `lifted_type(Val(N), P)` returns the corresponding wrapped slot type.
# For concrete `P`, `lifted_type(Val(N), P) === Lifted{P, N, dual_type(Val(N), P)}`.
#
# This file defines the IEEEFloat scalar case; container shapes (Array,
# Complex, Tuple, NamedTuple, struct lifts) are handled by the further
# `dual_type(::Val{N}, ...)` methods below.
# ──────────────────────────────────────────────────────────────────────────

"""
    dual_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `N`-width forward-mode value type for a primal of
type `P`. This is the type of the inner `V` carried by a
`Lifted{P, N, V}` slot.

Shapes defined so far:

- `P <: IEEEFloat`: `NDual{P, N}` — the packed scalar forward value.
- `Complex{R}` with `R <: IEEEFloat`: `Complex{NDual{R, N}}` — element-wise
  recursion through the complex real/imag parts.
- `Array{T, D}` with `T <: IEEEFloat`: `NDualArray{T, N, D, Array{T, D}, NDual{T, N}}`
  — the SoA canonical V wrapper (see §14 in dual-types.md).
- `Array{Complex{R}, D}` with `R <: IEEEFloat`: `NDualArray{Complex{R}, N, D, Array{Complex{R}, D}, Complex{NDual{R, N}}}`
  — complex-eltype SoA variant.
- `Tuple{T1, T2, …}` (concrete tuple): `Tuple{dual_type(Val(N), T1), …}` —
  element-wise recursion via head/tail type-cons.
- `NamedTuple{names, T}` with `T <: Tuple`: `NamedTuple{names, dual_type(Val(N), T)}`
  — same names, fields recursively lifted via the tuple-type path.
- Concrete struct `P` (immutable): `ImmutableDual{NamedTuple{fieldnames(P), Tuple{V_i...}}}`
  where each `V_i = dual_type(Val(N), fieldtype(P, i))`.
- Concrete struct `P` (mutable): `MutableDual{NamedTuple{...}}` — mutable
  counterpart for in-place tangent updates.
- `MemoryRef{T}` with `T <: IEEEFloat` (Julia 1.11+):
  `NDualMemoryRef{T, N, Memory{T}}` — parallel SoA wrapper (§14.2).
"""
@foldable @inline dual_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat} = NDual{P,N}
# Non-differentiable primitives — mirrors `tangent_type(T) === NoTangent`
# in reverse mode, returning the forward-mode V sentinel `NoDual`.
@foldable @inline dual_type(::Val{N}, ::Type{<:Integer}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Char}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Symbol}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Nothing}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{<:Type}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{<:TypeVar}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Module}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Expr}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Cstring}) where {N} = NoDual
@foldable @inline dual_type(::Val{N}, ::Type{Cwstring}) where {N} = NoDual
@foldable @inline function dual_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}
end
@foldable @inline function dual_type(::Val{N}, ::Type{Array{T,D}}) where {N,T<:IEEEFloat,D}
    return NDualArray{T,N,D,Array{T,D},NDual{T,N}}
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{Array{Complex{R},D}}
) where {N,R<:IEEEFloat,D}
    return NDualArray{Complex{R},N,D,Array{Complex{R},D},Complex{NDual{R,N}}}
end
# General array V, mirroring reverse-mode `tangent_type(Array{T,D}) === Array{tangent_type(T), D}`:
# always the element-wise Array-of-Structures V `Array{dual_type(Val(N), T), D}`, including
# `Array{NoDual, D}` for a non-differentiable element (e.g. `Vector{Int}` → `Vector{NoDual}`,
# mirroring reverse's `Vector{NoTangent}`). `tangent_type(Array{T,D})` is never `NoTangent`, so an
# array is never collapsed to whole `NoDual` — coherence requires `dual_type(P) === NoDual` only
# when `tangent_type(P) === NoTangent`. The IEEEFloat / Complex overloads above are more specific
# and provide the SoA `NDualArray` optimisation for scalar-float elements; this AoS form covers
# everything else (non-diff elements, tuples, structs, closures).
@foldable @generated function dual_type(::Val{N}, ::Type{Array{T,D}}) where {N,T,D}
    return :(Array{dual_type(Val($N), $T),$D})
end
# Tuple recursion: head/tail cons (via `_dual_tuple_v`). Specialized per concrete tuple type by
# Julia's normal dispatch, so concrete tuples resolve at compile time without an @generated function.
# A *standalone* empty tuple is non-differentiable (`tangent_type(Tuple{}) === NoTangent`), so its V
# is `NoDual` — coherent with reverse. (The cons *base case* `Tuple{}` lives in `_dual_tuple_v`
# below, which must keep it `Tuple{}` to terminate the recursion.)
@foldable @inline dual_type(::Val{N}, ::Type{Tuple{}}) where {N} = NoDual
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
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
    return _dual_tuple_v(Val(N), P)
end
# Element-wise tuple V via head/tail cons, WITHOUT the whole-tuple collapse gate so tails stay
# `Tuple`. Concrete tails only (the top `dual_type(Tuple)` guards `isconcretetype`).
@foldable @inline _dual_tuple_v(::Val{N}, ::Type{Tuple{}}) where {N} = Tuple{}
@foldable @inline function _dual_tuple_v(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    H = Base.tuple_type_head(P)
    Tail = Base.tuple_type_tail(P)
    return Base.tuple_type_cons(dual_type(Val(N), H), _dual_tuple_v(Val(N), Tail))
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{NamedTuple{names,T}}
) where {N,names,T<:Tuple}
    # An all-non-differentiable NamedTuple collapses to whole `NoDual`, mirroring
    # the generic struct / Array / Memory / Ptr rules. (Tuple stays element-wise:
    # its head/tail `dual_type` recursion needs a Tuple tail to cons onto.) A
    # NamedTuple `V` is not consed into a parent, so whole-`NoDual` is safe here,
    # and it matches the value the forward machinery actually produces for
    # non-differentiable kwargs/config flowing through rule construction.
    tangent_type(NamedTuple{names,T}) === NoTangent && return NoDual
    # Mirror `tangent_type(NamedTuple)`: an abstract field (e.g. `parts::Any` in a reverse
    # `MutableTangent` NamedTuple flowing through the forward-over-reverse HVP path) makes
    # `dual_type(Val(N), T)` non-concrete (`Any`), and `NamedTuple{names, Any}` is invalid
    # (the 2nd param must be `<:Tuple`). Widen to `Any` in that case, matching `tangent_type`.
    DT = dual_type(Val(N), T)
    return isconcretetype(DT) ? NamedTuple{names,DT} : Any
end
# `Ptr{T}` canonical V — `NTuple{N, Ptr{T}}` per the design notes' Ptr
# entry: N parallel partial pointers, one per lane. Matches reverse-mode
# `tangent_type(Ptr{T}) === Ptr{tangent_type(T)}` at the per-lane level.
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
# an NDualArray over `Memory{T}` (per dual-types.md §14: "Memory{T}
# (1.11+) → NDualArray{T, N, 1, Memory{T}}").
@static if VERSION >= v"1.11-rc4"
    @inline function dual_type(::Val{N}, ::Type{MemoryRef{T}}) where {N,T<:IEEEFloat}
        return NDualMemoryRef{T,N,Memory{T}}
    end
    @inline function dual_type(::Val{N}, ::Type{Memory{T}}) where {N,T<:IEEEFloat}
        return NDualArray{T,N,1,Memory{T},NDual{T,N}}
    end
    @inline function dual_type(::Val{N}, ::Type{Memory{Complex{R}}}) where {N,R<:IEEEFloat}
        return NDualArray{Complex{R},N,1,Memory{Complex{R}},Complex{NDual{R,N}}}
    end
    # General (non-float) `Memory` / `MemoryRef` V, mirroring the AoS `Array` rule
    # above: non-diff element → `NoDual`; differentiable element → AoS
    # `Memory{dual_type(elt)}` / `MemoryRef{dual_type(elt)}` (a plain memory/ref
    # of per-element forward Vs). Covers the reverse rule's `Memory{Tuple{pullback}}`
    # comms storage under forward-over-reverse. The IEEEFloat overloads above are
    # more specific and provide the SoA optimisation for scalar-float elements.
    @generated function dual_type(::Val{N}, ::Type{Memory{T}}) where {N,T}
        return :(Memory{dual_type(Val($N), $T)})
    end
    @generated function dual_type(::Val{N}, ::Type{MemoryRef{T}}) where {N,T}
        return :(MemoryRef{dual_type(Val($N), $T)})
    end
end

# Recursive structural lift for concrete struct primals — the @generated
# fallback. The two terminal answers mirror reverse-mode `tangent_type`'s two
# distinct answers, and the distinction matters:
#  - non-concrete `P` widens to `Any` — "the derivative could be anything"
#    (`tangent_type` returns `Any` here too). This is an upper bound, not a
#    claim of no-derivative; abstract slot primals are sharpened to concrete V
#    at runtime via `lifted_type`'s UnionAll, and an abstract *field* of a
#    concrete struct keeps its derivative because a concrete runtime V is a
#    subtype of the `Any`-typed backing slot.
#  - `tangent_type(P) === NoTangent` (non-differentiable concrete types: `Int`,
#    `Symbol`, recursive Core internals like `CodeInstance`, …) maps to the
#    `NoDual` sentinel — the forward analogue of `NoTangent`, a specific
#    "no derivative here", which also terminates the field recursion.
# Fields are lifted uniformly via `dual_type(Val(N), fieldtype)`: concrete
# differentiable fields recurse, abstract fields hit the `Any` rule, and
# non-differentiable fields hit the `NoDual` rule. The seed factories below
# coerce field storage into the declared backing NamedTuple so a differentiable
# value flowing into an `Any`-typed field still yields `V === dual_type(Val(N), P)`.
@foldable @generated function dual_type(::Val{N}, ::Type{P}) where {N,P}
    isconcretetype(P) || return Any
    # The `NoDual` (non-differentiable) decision keys off `tangent_type(P)`, which an
    # extension may override (e.g. CUDA's `CuPtr`/`CuArray`). Emit that call in the RETURNED
    # expression — never the generator body — so it resolves at the `dual_type` call's world,
    # where extension overloads are visible, instead of the generator's (core) definition
    # world: a generator-body call bakes its resolution at definition time and a later
    # overload cannot dislodge it. The structural skeleton below (field names, mutability,
    # field count) is world-independent, so it stays in the generator body.
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
    return :(tangent_type($P) === NoTangent ? NoDual : $wrapper{$inner_nt_type})
end

"""
    lifted_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `Lifted{P, N, V}` slot type for a primal of type `P`
at width `N`. For concrete `P`, equals `Lifted{P, N, dual_type(Val(N), P)}`.

Shapes defined so far:

- `P <: IEEEFloat`: `Lifted{P, N, NDual{P, N}}`.
- `Complex{R}` with `R <: IEEEFloat`: `Lifted{Complex{R}, N, Complex{NDual{R, N}}}`.
- `Array{T, D}` with `T <: IEEEFloat`: `Lifted{Array{T, D}, N, NDualArray{T, N, D, Array{T, D}, NDual{T, N}}}`.
- `Array{Complex{R}, D}` with `R <: IEEEFloat`: `Lifted{Array{Complex{R}, D}, N, NDualArray{Complex{R}, N, D, Array{Complex{R}, D}, Complex{NDual{R, N}}}}`.
- `MemoryRef{T}` with `T <: IEEEFloat` (Julia 1.11+):
  `Lifted{MemoryRef{T}, N, NDualMemoryRef{T, N, Memory{T}}}`.
- `P <: Tuple` (concrete): `Lifted{P, N, dual_type(Val(N), P)}`.
- `P <: NamedTuple{names, <:Tuple}`: `Lifted{P, N, dual_type(Val(N), P)}`.
- Concrete struct `P`: `Lifted{P, N, dual_type(Val(N), P)}` where the inner
  V is `ImmutableDual` (immutable) or `MutableDual` (mutable).
"""
# A non-concrete `P <: IEEEFloat` (e.g. a type-unstable closure inferred to
# `Union{Float32,Float64}`) must widen to a UnionAll: `Lifted` is invariant in `P`, so the
# invariant `Lifted{Union{…},N,NDual{Union{…},N}}` would reject the concrete runtime
# `Lifted{Float32,…}` at the OpaqueClosure arg typeassert (mirrors the Tuple/struct overloads).
@inline function lifted_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat}
    return isconcretetype(P) ? Lifted{P,N,NDual{P,N}} : (Lifted{T,N,V} where {T<:P,V})
end
@inline function lifted_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Lifted{Complex{R},N,Complex{NDual{R,N}}}
end
@inline function lifted_type(::Val{N}, ::Type{Array{T,D}}) where {N,T<:IEEEFloat,D}
    return Lifted{Array{T,D},N,NDualArray{T,N,D,Array{T,D},NDual{T,N}}}
end
@inline function lifted_type(::Val{N}, ::Type{Array{Complex{R},D}}) where {N,R<:IEEEFloat,D}
    return Lifted{
        Array{Complex{R},D},
        N,
        NDualArray{Complex{R},N,D,Array{Complex{R},D},Complex{NDual{R,N}}},
    }
end
@inline lifted_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
# Abstract tuple/named-tuple `P` (e.g. a grouped-vararg `Tuple{Function,
# Vararg{Any}}` in the forward IR) must widen to a UnionAll: `Lifted` is invariant
# in `P`, so a concrete runtime `Lifted{Tuple{f,x},…}` is *not* a subtype of
# `Lifted{Tuple{Function,Vararg},N,Any}` and the OpaqueClosure arg typeassert
# would reject it (mirrors the generic struct overload below).
@inline function lifted_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    return if isconcretetype(P)
        Lifted{P,N,dual_type(Val(N), P)}
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
@inline function lifted_type(
    ::Val{N}, ::Type{P}
) where {N,names,T<:Tuple,P<:NamedTuple{names,T}}
    return if isconcretetype(P)
        Lifted{P,N,dual_type(Val(N), P)}
    else
        (Lifted{S,N,V} where {S<:P,V})
    end
end
@inline function lifted_type(::Val{N}, ::Type{Ptr{T}}) where {N,T<:NDualEltype}
    return Lifted{Ptr{T},N,NTuple{N,Ptr{T}}}
end
# MemoryRef + Memory canonical lifts (Julia 1.11+).
@static if VERSION >= v"1.11-rc4"
    @inline function lifted_type(::Val{N}, ::Type{MemoryRef{T}}) where {N,T<:IEEEFloat}
        return Lifted{MemoryRef{T},N,NDualMemoryRef{T,N,Memory{T}}}
    end
    @inline function lifted_type(::Val{N}, ::Type{Memory{T}}) where {N,T<:IEEEFloat}
        return Lifted{Memory{T},N,NDualArray{T,N,1,Memory{T},NDual{T,N}}}
    end
end
# Concrete-struct fallback. More-specific overloads above (IEEEFloat,
# Complex, Array, Tuple, NamedTuple, MemoryRef) win when applicable;
# structs land here.
#
# For abstract `P` (or `DataType` and other metatypes whose instances are
# concrete subtypes), return a UnionAll-typed Lifted so a runtime arg
# with a more-specific concrete `T<:P` matches. The interpreter widens
# argtypes via `CC.widenconst` and may produce abstract `P`; without
# the UnionAll, `Lifted{Type{X}, N, V}` wouldn't be a subtype of
# `Lifted{DataType, N, NoDual}` (Lifted is invariant in `P`).
@inline function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    return if isconcretetype(P) && P !== DataType
        Lifted{P,N,dual_type(Val(N), P)}
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# A precise `Type{X}` slot (e.g. a Type-valued callable like the `Pt2{Float64}` constructor) is
# monomorphic — its sole inhabitant is `X` — so the concrete `Lifted{Type{X}, N, …}` is exact and
# carries no impossible type fact. `isconcretetype(Type{X})` is `false`, so the generic method above
# would route it to the UnionAll-widened branch, forcing the runtime slot to box at the OC argument
# boundary. The kind-widening (34782f41b) is needed only for the genuinely abstract metatypes
# (`DataType`, `Type`, `Type{<:T}` — which stay on the generic method); it explicitly excludes these
# well-behaved `Type{X}` singletons, so narrow them here to keep the slot box-free.
@inline function lifted_type(::Val{N}, ::Type{Type{X}}) where {N,X}
    return Lifted{Type{X},N,dual_type(Val(N), Type{X})}
end

# ──────────────────────────────────────────────────────────────────────────
# Seed factories.
#
# Layer 2 — bare inner V (the slot's `value::V` field content):
#   `zero_dual(Val(N), x)`     — `dual_type(Val(N), typeof(x))` with zero partials.
#   `uninit_dual(Val(N), x)`   — same shape; tangent payload semantically uninitialized.
#   `randn_dual(Val(N), rng, x)` — random partials sampled from `randn`.
#
# Layer 3 — wrapped Lifted slot:
#   `zero_lifted(Val(N), x)`     — `Lifted{typeof(x), N}` wrapping `zero_dual`.
#   `uninit_lifted(Val(N), x)`   — wrapping `uninit_dual`.
#   `randn_lifted(Val(N), rng, x)` — wrapping `randn_dual`.
# ──────────────────────────────────────────────────────────────────────────

@inline function zero_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

# ── Width-1 boundary helper for user-supplied tangents ──────────────────────
#
# `lift(primal, ẋ)` builds a width-1 `Lifted{P, 1, V}` slot from a primal and
# a tangent value of shape `tangent_type(P)`. Used by public-facing APIs
# (`value_and_derivative!!`, `test_rule`, etc.) that take a user-supplied JVP
# direction, and by the interpreter cutover boundary. Width-1 only — width-N
# chunk seeds are built internally via `basis_lifted!!` + `Lifted{P, N}(primal, value)`
# with the appropriate width-N V.
@inline lift(x::T, ẋ::T) where {T<:IEEEFloat} = Lifted{T,1}(x, NDual{T,1}(x, (ẋ,)))
@inline function lift(x::A, ẋ::A) where {T<:IEEEFloat,D,A<:Array{T,D}}
    return Lifted{A,1}(x, NDualArray{T,1,D,A}(x, (ẋ,)))
end
@inline function lift(x::A, ẋ::A) where {R<:IEEEFloat,D,A<:Array{Complex{R},D}}
    return Lifted{A,1}(x, NDualArray{Complex{R},1,D,A}(x, (ẋ,)))
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
@static if VERSION >= v"1.11-rc4"
    # `MemoryRef{T}` (T<:IEEEFloat) reverse fdata is itself a `MemoryRef{T}` (the
    # derivative storage); its forward V is the SoA `NDualMemoryRef`. Reached in
    # forward-over-reverse, where a reverse rule's `dx::MemoryRef` field is lifted.
    @inline function lift(x::MemoryRef{T}, ẋ::MemoryRef{T}) where {T<:IEEEFloat}
        return Lifted{MemoryRef{T},1}(x, NDualMemoryRef{T,1,Memory{T}}(x, (ẋ,)))
    end
    # `_lift_backing` always calls the 3-arg form; without these SoA-specific 3-arg
    # passthroughs a lifted reverse rule's float `dx::MemoryRef`/`Memory` field falls
    # to the generic AoS lift below, producing a `MemoryRef{NDual}` that cannot convert
    # to the declared SoA V. Mirrors the float `Array` passthroughs.
    @inline lift(x::MemoryRef{T}, ẋ::MemoryRef{T}, ::Union{Nothing,IdDict}) where {T<:IEEEFloat} = lift(
        x, ẋ
    )
    # `Memory{T}` (T<:IEEEFloat / Complex{<:IEEEFloat}) lifts to the SoA NDualArray,
    # mirroring the `Array` overloads above; reached when a reverse rule's `Memory`
    # field is lifted under forward-over-reverse, or a Memory primal is seeded.
    @inline function lift(x::A, ẋ::A) where {T<:IEEEFloat,A<:Memory{T}}
        return Lifted{A,1}(x, NDualArray{T,1,1,A}(x, (ẋ,)))
    end
    @inline function lift(x::A, ẋ::A) where {R<:IEEEFloat,A<:Memory{Complex{R}}}
        return Lifted{A,1}(x, NDualArray{Complex{R},1,1,A}(x, (ẋ,)))
    end
    @inline lift(x::A, ẋ::A, ::Union{Nothing,IdDict}) where {T<:IEEEFloat,A<:Memory{T}} = lift(
        x, ẋ
    )
    @inline lift(x::A, ẋ::A, ::Union{Nothing,IdDict}) where {R<:IEEEFloat,A<:Memory{Complex{R}}} = lift(
        x, ẋ
    )
    # Non-differentiable-element `Memory` (reverse tangent `Memory{NoTangent}`)
    # lifts element-wise to `Memory{NoDual}`, mirroring the `Array{<:NoTangent}`
    # overload and `dual_type(Memory{T}) = Memory{NoDual}`.
    @inline lift(x::Memory, ẋ::Memory{<:NoTangent}) = Lifted{typeof(x),1}(
        x, map(_ -> NoDual(), ẋ)
    )
    # General AoS `Memory` (differentiable non-float / nested / `Any` element):
    # element-wise V `Memory{dual_type(elt)}`, mirroring the generic `Array` lift.
    @inline lift(x::Memory, ẋ::Memory) = lift(x, ẋ, nothing)
    @inline function lift(x::Memory, ẋ::Memory, c::Union{Nothing,IdDict})
        v = similar(x, dual_type(Val(1), eltype(x)))
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = tangent(lift(x[i], ẋ[i], c)))
        end
        return Lifted{typeof(x),1,typeof(v)}(x, v)
    end
    # General AoS `MemoryRef` lift (non-float / nested / `Any` / `NoTangent`
    # element): lift the `.mem` via the Memory lift, then `memoryref` at the
    # offset. The `MemoryRef{IEEEFloat}` SoA overload above is more specific.
    @inline lift(x::MemoryRef, ẋ::MemoryRef) = lift(x, ẋ, nothing)
    @inline function lift(x::MemoryRef, ẋ::MemoryRef, c::Union{Nothing,IdDict})
        mem_v = tangent(lift(x.mem, ẋ.mem, c))
        ref_v = _memoryref_at(mem_v, Core.memoryrefoffset(x))
        return Lifted{typeof(x),1,typeof(ref_v)}(x, ref_v)
    end
end
# ── Aliasing cache for `lift` ───────────────────────────────────────────────
#
# `lift` threads an optional aliasing cache `c`, mirroring reverse-mode
# `zero_tangent`'s `MaybeCache`: a fresh `IdDict`, created once per top-level
# lift and passed down the structural recursion. The `MistyClosure` overload
# uses it to build a reverse rule's captures' forward V exactly once and share
# it (keyed by the primal captures identity) — so `fwds_oc` and `pb_oc`, which
# share those captures (block stacks), share the forward tangent too; otherwise
# partials pushed in the forward pass are invisible when popped in the reverse
# pass (the HVP silently zeroes). Leaf/passthrough overloads ignore `c` via this
# fallback; only the structural overloads below thread it.
@inline lift(x, ẋ, ::Union{Nothing,IdDict}) = lift(x, ẋ)

# Coerce the per-field V tuple into the declared backing NamedTuple
# (`fieldtype(dual_type(Val(1), P), 1)`), mirroring the seed factories so the
# resulting V matches `dual_type(Val(1), P)` even when a field declared abstract
# holds a differentiable value (its V stored as `Any`).
# Build the declared backing NamedTuple from a reverse Tangent's `fields`,
# field-by-field. A `PossiblyUninitTangent` backing field is `isdefined`-guarded
# (an undefined primal field — e.g. lazily-built `LazyDerivedRule.rule` — yields
# an uninit PUT); always-init fields lift directly. Mirrors the seed factories.
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
@inline lift(x::P, ẋ::Tangent) where {P} = lift(x, ẋ, nothing)
@inline function lift(x::P, ẋ::Tangent, c::Union{Nothing,IdDict}) where {P}
    backing = fieldtype(dual_type(Val(1), P), 1)
    return Lifted{P,1}(x, ImmutableDual(_lift_backing(x, ẋ.fields, backing, c)))
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
    lifted.value.value = _lift_backing(x, ẋ.fields, backing, d)
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
# Array{NoDual,D}`). The 3-arg passthrough keeps this more-specific behaviour ahead of the AoS
# overload below when a cache is threaded.
@inline lift(x::Array, ẋ::Array{<:NoTangent}) = Lifted{typeof(x),1}(
    x, map(_ -> NoDual(), ẋ)
)
@inline lift(x::Array, ẋ::Array{<:NoTangent}, ::Union{Nothing,IdDict}) = lift(x, ẋ)
# Float / Complex-float element arrays are terminal (their V aliases `ẋ`); they
# match the AoS overload below by element type, so give them cache-threading
# passthroughs that keep the more-specific terminal behaviour.
@inline lift(x::A, ẋ::A, ::Union{Nothing,IdDict}) where {T<:IEEEFloat,D,A<:Array{T,D}} = lift(
    x, ẋ
)
@inline lift(x::A, ẋ::A, ::Union{Nothing,IdDict}) where {R<:IEEEFloat,D,A<:Array{Complex{R},D}} = lift(
    x, ẋ
)
# Differentiable non-float-element array: AoS V `Array{dual_type(Val(1), T), D}`,
# built element-wise from the per-element lift (coherent with `dual_type` above).
# The IEEEFloat / Complex / all-`NoTangent` overloads are more specific and win.
@inline lift(x::Array{T,D}, ẋ::Array) where {T,D} = lift(x, ẋ, nothing)
@inline function lift(x::Array{T,D}, ẋ::Array, c::Union{Nothing,IdDict}) where {T,D}
    Vel = dual_type(Val(1), T)
    v = similar(x, Vel)
    @inbounds for i in eachindex(x)
        if isassigned(x, i)
            v[i] = tangent(lift(x[i], ẋ[i], c))
        end
    end
    return Lifted{typeof(x),1,typeof(v)}(x, v)
end
# V-shape passthrough — the test framework's tangent-shape arithmetic
# sometimes feeds raw Lifted V values (NoDual, ImmutableDual, MutableDual)
# back into `lift`. Wrap directly rather than re-deriving V from the (now-V)
# tangent input.
@inline lift(x::P, ẋ::NoDual) where {P} = Lifted{P,1}(x, ẋ)
@inline function lift(x::P, ẋ::Union{ImmutableDual,MutableDual}) where {P}
    return Lifted{P,1}(x, ẋ)
end
# Tuple / NamedTuple primal + per-field reverse tangents → per-field V. Reached
# in real AD (lifting a tuple/named-tuple primal) and when the test framework's
# tangent-shape arithmetic recurses through a `Tangent`'s fields and feeds the V
# values back into `lift`.
@inline lift(x::Tuple, ẋ::Tuple) = lift(x, ẋ, nothing)
@inline function lift(x::Tuple, ẋ::Tuple, c::Union{Nothing,IdDict})
    field_Vs = map((xi, vi) -> tangent(lift(xi, vi, c)), x, ẋ)
    return Lifted{typeof(x),1}(x, field_Vs)
end
@inline lift(x::NamedTuple, ẋ::NamedTuple) = lift(x, ẋ, nothing)
@inline function lift(
    x::NamedTuple{names}, ẋ::NamedTuple, c::Union{Nothing,IdDict}
) where {names}
    field_Vs = map((xi, vi) -> tangent(lift(xi, vi, c)), values(x), values(ẋ))
    return Lifted{typeof(x),1}(x, NamedTuple{names}(field_Vs))
end

@inline function uninit_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

@inline function randn_dual(::Val{N}, rng::AbstractRNG, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> randn(rng, T), Val(N)))
end

# Non-differentiable primitives — mirrors `dual_type(Val(N), T) === NoDual`
# above. Without these the @generated struct-lift fallback errors on
# primitive `T` (Int, Symbol, …), blocking `zero_lifted(Val(N), 42)` etc.
# at the interpreter boundary.
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
# Build an `NDualArray{T, N, D, Array{T, D}, NDual{T, N}}` whose `primal` aliases
# the user's array and whose `partials` is slot-local — no aliasing with the
# user's array. The zero-init path uses `zero(::Array)` for each lane.

@inline function zero_dual(::Val{N}, x::A) where {N,T<:IEEEFloat,D,A<:Array{T,D}}
    return NDualArray{T,N,D,A}(x)
end
@inline function zero_dual(::Val{N}, x::A) where {N,R<:IEEEFloat,D,A<:Array{Complex{R},D}}
    return NDualArray{Complex{R},N,D,A}(x)
end

@inline function uninit_dual(::Val{N}, x::A) where {N,T<:IEEEFloat,D,A<:Array{T,D}}
    return NDualArray{T,N,D,A}(x, ntuple(_ -> similar(x), Val(N)))
end
@inline function uninit_dual(::Val{N}, x::A) where {N,R<:IEEEFloat,D,A<:Array{Complex{R},D}}
    return NDualArray{Complex{R},N,D,A}(x, ntuple(_ -> similar(x), Val(N)))
end

@inline function randn_dual(
    ::Val{N}, rng::AbstractRNG, x::A
) where {N,T<:IEEEFloat,D,A<:Array{T,D}}
    return NDualArray{T,N,D,A}(x, ntuple(_ -> randn(rng, T, size(x)), Val(N)))
end
@inline function randn_dual(
    ::Val{N}, rng::AbstractRNG, x::A
) where {N,R<:IEEEFloat,D,A<:Array{Complex{R},D}}
    return NDualArray{Complex{R},N,D,A}(
        x, ntuple(_ -> randn(rng, Complex{R}, size(x)), Val(N))
    )
end

# ── Array seed factories (differentiable non-float elements: AoS) ────────────
#
# Mirrors `dual_type(Val(N), Array{T,D}) === Array{dual_type(Val(N), T), D}` and
# the AoS `lift(::Array, ::Array)` path: a per-element V array, built element-wise
# (skipping undefined slots). `@generated` so the `NoDual` short-circuit for a
# non-differentiable element type is resolved at compile time (type-stable).
# Float / Complex-float elements use the more-specific `NDualArray` methods above.
@generated function zero_dual(::Val{N}, x::Array{T,D}) where {N,T,D}
    dual_type(Val(N), Array{T,D}) === NoDual && return :(NoDual())
    return quote
        v = similar(x, dual_type(Val($N), T))
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = zero_dual(Val($N), x[i]))
        end
        return v
    end
end
@generated function uninit_dual(::Val{N}, x::Array{T,D}) where {N,T,D}
    dual_type(Val(N), Array{T,D}) === NoDual && return :(NoDual())
    return quote
        v = similar(x, dual_type(Val($N), T))
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = uninit_dual(Val($N), x[i]))
        end
        return v
    end
end
@generated function randn_dual(::Val{N}, rng::AbstractRNG, x::Array{T,D}) where {N,T,D}
    dual_type(Val(N), Array{T,D}) === NoDual && return :(NoDual())
    return quote
        v = similar(x, dual_type(Val($N), T))
        @inbounds for i in eachindex(x)
            isassigned(x, i) && (v[i] = randn_dual(Val($N), rng, x[i]))
        end
        return v
    end
end

# ── Tuple seed factories (concrete tuple) ───────────────────────────────────
#
# Element-wise build via Tuple-aware `map`. Each element's dispatch picks its own seed factory
# recursively. Gate on `dual_type === NoDual` (NOT `tangent_type === NoTangent`): a wholly-non-diff
# tuple collapses to `NoDual`, but the EMPTY tuple has `dual_type(Tuple{}) === Tuple{}` (not NoDual)
# even though `tangent_type(Tuple{}) === NoTangent` — so it must seed to `()`, not `NoDual()` (e.g.
# a `ReshapedArray`'s empty `mi::Tuple{}` field).

@inline function zero_dual(w::Val{N}, x::Tuple) where {N}
    dual_type(w, typeof(x)) === NoDual && return NoDual()
    return map(xi -> zero_dual(w, xi), x)
end
@inline function uninit_dual(w::Val{N}, x::Tuple) where {N}
    dual_type(w, typeof(x)) === NoDual && return NoDual()
    return map(xi -> uninit_dual(w, xi), x)
end
@inline function randn_dual(w::Val{N}, rng::AbstractRNG, x::Tuple) where {N}
    dual_type(w, typeof(x)) === NoDual && return NoDual()
    return map(xi -> randn_dual(w, rng, xi), x)
end

# ── NamedTuple seed factories ───────────────────────────────────────────────
#
# Julia's `map(f, ::NamedTuple)` preserves the names and returns a
# NamedTuple, so element-wise seed building works the same as for Tuple.

# All-non-differentiable NamedTuples seed to whole `NoDual`, matching their
# `dual_type`; otherwise build element-wise.
@inline function zero_dual(w::Val{N}, x::NamedTuple) where {N}
    tangent_type(typeof(x)) === NoTangent && return NoDual()
    return map(xi -> zero_dual(w, xi), x)
end
@inline function uninit_dual(w::Val{N}, x::NamedTuple) where {N}
    tangent_type(typeof(x)) === NoTangent && return NoDual()
    return map(xi -> uninit_dual(w, xi), x)
end
@inline function randn_dual(w::Val{N}, rng::AbstractRNG, x::NamedTuple) where {N}
    tangent_type(typeof(x)) === NoTangent && return NoDual()
    return map(xi -> randn_dual(w, rng, xi), x)
end

# ── Complex seed factories (R <: IEEEFloat) ─────────────────────────────────
#
# Defined before the generic struct-lift seed factory so the more-specific
# Complex overload wins for `Complex{<:IEEEFloat}` slots — the canonical V
# is `Complex{NDual{R, N}}`, not the structural-lift `ImmutableDual{...}`.

@inline function zero_dual(w::Val{N}, z::Complex{R}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}(zero_dual(w, real(z)), zero_dual(w, imag(z)))
end
@inline function uninit_dual(w::Val{N}, z::Complex{R}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}(uninit_dual(w, real(z)), uninit_dual(w, imag(z)))
end
@inline function randn_dual(
    w::Val{N}, rng::AbstractRNG, z::Complex{R}
) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}(randn_dual(w, rng, real(z)), randn_dual(w, rng, imag(z)))
end

# ── Concrete-struct seed factories (generic @generated fallback) ────────────
#
# Build a `NamedTuple{fieldnames(P), Tuple{V_i...}}` of recursive field Vs
# and wrap in `ImmutableDual` or `MutableDual` based on mutability. Sub-
# function calls (`zero_dual`, etc.) live in the returned expression per
# AGENTS.md; non-concrete and primitive `P` use the deferred-error pattern.
# More-specific overloads above (IEEEFloat, Complex, Array, Tuple,
# NamedTuple) take precedence, so this fallback only fires for user-defined
# struct primals.

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
        # Coherence with `dual_type`: a `NoDual` V has no backing to seed. This covers
        # `tangent_type(P) === NoTangent` and non-differentiable-element arrays/`Ptr`
        # (where `tangent_type(P) !== NoTangent` but the element is non-diff, e.g. a
        # reverse rule's `Vector{Int32}` block-stack storage). The `dual_type(...) ===
        # NoDual` test goes in the returned expression (call world), not the generator
        # body — see `dual_type(::Type{P})` above for why.
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
        return quote
            V = dual_type(Val($N), typeof(x))
            V === NoDual && return NoDual()
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
    return quote
        V = dual_type(Val($N), typeof(x))
        V === NoDual && return NoDual()
        $wrapper(fieldtype(V, 1)(($(seeds...),)))
    end
end

# ── Cache-aware seed construction (cycle/alias-aware) ───────────────────────
#
# The `zero_dual` / `uninit_dual` / `randn_dual` factories above are cache-free
# type-recursion — fine for flat/packable seeds, but they neither dedup aliased
# array fields (two struct fields pointing at one array would get two independent
# `NDualArray`s) nor terminate on cyclic mutable structs. The `*_internal`
# functions thread a `MaybeCache` (mirroring reverse `zero_tangent_internal` and
# the cyclic `lift`): arrays register by primal identity so aliased fields share
# one V, and a mutable struct registers a `MutableDual` shell BEFORE recursing
# its fields so a cycle reaching `x` again returns the shell. Leaf / non-struct
# types delegate to the cache-free factory. `zero_lifted` / `uninit_lifted` /
# `randn_lifted` enter through these, so the public forward gradient/derivative
# seeds (which may hold structs, aliasing, or cycles) are correct; the cache-free
# factories remain the fast path for direct callers and packable chunk seeds.
for (factory, internal) in
    ((:zero_dual, :_zero_dual_internal), (:uninit_dual, :_uninit_dual_internal))
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
                    shell.value = backing(($(seeds...),))
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
        # Array: register by identity so aliased struct fields share one V.
        function $internal(w::Val{N}, x::Array, d::MaybeCache) where {N}
            haskey(d, x) && return d[x]
            v = $factory(w, x)
            d[x] = v
            return v
        end
        function $internal(w::Val{N}, x::Tuple, d::MaybeCache) where {N}
            dual_type(w, typeof(x)) === NoDual && return NoDual()
            return map(xi -> $internal(w, xi, d), x)
        end
        function $internal(w::Val{N}, x::NamedTuple{names}, d::MaybeCache) where {N,names}
            tangent_type(typeof(x)) === NoTangent && return NoDual()
            return NamedTuple{names}(map(xi -> $internal(w, xi, d), values(x)))
        end
        # Complex / Memory / MemoryRef have fields but their own canonical V (not a
        # structural lift), so delegate to the cache-free factory.
        $internal(w::Val{N}, z::Complex, ::MaybeCache) where {N} = $factory(w, z)
    end
    @static if VERSION >= v"1.11-rc4"
        @eval $internal(w::Val{N}, x::Union{Memory,MemoryRef}, ::MaybeCache) where {N} = $factory(
            w, x
        )
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
            shell.value = backing(($(seeds...),))
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
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, x::Array, d::MaybeCache
) where {N}
    haskey(d, x) && return d[x]
    v = randn_dual(w, rng, x)
    d[x] = v
    return v
end
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, x::Tuple, d::MaybeCache
) where {N}
    dual_type(w, typeof(x)) === NoDual && return NoDual()
    return map(xi -> _randn_dual_internal(w, rng, xi, d), x)
end
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, x::NamedTuple{names}, d::MaybeCache
) where {N,names}
    tangent_type(typeof(x)) === NoTangent && return NoDual()
    return NamedTuple{names}(map(xi -> _randn_dual_internal(w, rng, xi, d), values(x)))
end
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, z::Complex, ::MaybeCache
) where {N}
    randn_dual(w, rng, z)
end
@static if VERSION >= v"1.11-rc4"
    _randn_dual_internal(w::Val{N}, rng::AbstractRNG, x::Union{Memory,MemoryRef}, ::MaybeCache) where {N} = randn_dual(
        w, rng, x
    )
end

@inline function zero_lifted(w::Val{N}, x::P) where {N,P}
    return Lifted{P,N}(
        x, _zero_dual_internal(w, x, isbitstype(P) ? NoCache() : IdDict{Any,Any}())
    )
end
@inline function uninit_lifted(w::Val{N}, x::P) where {N,P}
    return Lifted{P,N}(
        x, _uninit_dual_internal(w, x, isbitstype(P) ? NoCache() : IdDict{Any,Any}())
    )
end
@inline function randn_lifted(w::Val{N}, rng::AbstractRNG, x::P) where {N,P}
    return Lifted{P,N}(
        x, _randn_dual_internal(w, rng, x, isbitstype(P) ? NoCache() : IdDict{Any,Any}())
    )
end

# ── Standard-basis seed (`basis_lifted!!`) ──────────────────────────────────
#
# `basis_lifted!!(seed, slots)` resets `seed` (built by `zero_lifted`) to the
# standard-basis direction(s): lane `k` is hot at the `slots[k]`-th scalar dof of
# the input, where dofs are counted in the same order as `dof` / `zero_tangent`
# (a `Complex` element occupies two consecutive real dofs). It writes the partials
# in place for an `NDualArray` (so a preallocated array seed can be reseeded per
# chunk allocation-free) and rebuilds the immutable inner V otherwise (`NDual`,
# tuples, `ImmutableDual`, and the `MutableDual` NamedTuple); the caller always
# uses the return value (the `!!` convention). A `cursor` threads the global
# scalar-dof index and an `IdDict`
# visits aliased arrays / cyclic mutable structs once, matching the dedup in
# `dof` and the cycle/alias-aware `zero_lifted` the seed must come from.
@inline function basis_lifted!!(seed::Lifted{P,N}, slots::NTuple{N,Int}) where {P,N}
    v = _basis_seed!!(seed.value, slots, Ref(0), IdDict{Any,Any}())
    return Lifted{P,N}(primal(seed), v)
end

_basis_seed!!(::NoDual, _slots, _cursor, _dict) = NoDual()
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
function _basis_seed!!(
    v::NDualArray{T,N}, slots::NTuple{N,Int}, cursor, dict
) where {T<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    @inbounds for idx in eachindex(v.primal)
        cursor[] += 1
        c = cursor[]
        for k in 1:N
            v.partials[k][idx] = c == slots[k] ? one(T) : zero(T)
        end
    end
    return v
end
function _basis_seed!!(
    v::NDualArray{Complex{R},N}, slots::NTuple{N,Int}, cursor, dict
) where {R<:IEEEFloat,N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    @inbounds for idx in eachindex(v.primal)
        cursor[] += 1
        cr = cursor[]
        cursor[] += 1
        ci = cursor[]
        for k in 1:N
            v.partials[k][idx] = Complex(
                cr == slots[k] ? one(R) : zero(R), ci == slots[k] ? one(R) : zero(R)
            )
        end
    end
    return v
end
# AoS V (element-wise `Array` of inner duals, e.g. an abstract-element array):
# rebuild each element in place.
function _basis_seed!!(v::Array, slots::NTuple{N,Int}, cursor, dict) where {N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    @inbounds for i in eachindex(v)
        isassigned(v, i) && (v[i] = _basis_seed!!(v[i], slots, cursor, dict))
    end
    return v
end
function _basis_seed!!(v::Tuple, slots::NTuple{N,Int}, cursor, dict) where {N}
    return map(e -> _basis_seed!!(e, slots, cursor, dict), v)
end
function _basis_seed!!(
    v::NamedTuple{names}, slots::NTuple{N,Int}, cursor, dict
) where {names,N}
    return NamedTuple{names}(map(e -> _basis_seed!!(e, slots, cursor, dict), values(v)))
end
function _basis_seed!!(
    v::PossiblyUninitTangent, slots::NTuple{N,Int}, cursor, dict
) where {N}
    is_init(v) || return v
    return typeof(v)(_basis_seed!!(val(v), slots, cursor, dict))
end
function _basis_seed!!(v::ImmutableDual, slots::NTuple{N,Int}, cursor, dict) where {N}
    return ImmutableDual(_basis_seed!!(v.value, slots, cursor, dict))
end
function _basis_seed!!(v::MutableDual, slots::NTuple{N,Int}, cursor, dict) where {N}
    haskey(dict, v) && return dict[v]
    dict[v] = v
    v.value = _basis_seed!!(v.value, slots, cursor, dict)
    return v
end

# Width-1 helpers: zero_dual(x) / uninit_dual(x) / randn_dual(rng, x) produce a
# `Lifted{P,1}` slot directly. Kept under the same `zero_dual` / `uninit_dual` /
# `randn_dual` names so the many existing callsites (`zero_dual(f)` for function
# args, etc.) work without renames.
@inline zero_dual(x) = zero_lifted(Val(1), x)
@inline uninit_dual(x) = uninit_lifted(Val(1), x)
@inline randn_dual(rng::AbstractRNG, x) = randn_lifted(Val(1), rng, x)

# ── MemoryRef seed factories (Julia 1.11+) ──────────────────────────────────
#
# `uninit_dual` / `randn_dual` for MemoryRef are deferred — `zero_dual` is
# the canonical entry point per §14.2 (bits-element dense zero-init).

@static if VERSION >= v"1.11-rc4"
    @inline function zero_dual(::Val{N}, p::MemoryRef{T}) where {N,T<:IEEEFloat}
        return NDualMemoryRef{T,N,Memory{T}}(p)
    end
    # SoA `MemoryRef{IEEEFloat}` uninit/randn: per-lane partial MemoryRefs into
    # fresh `Memory{T}` at `p`'s offset (mirrors the `NDualMemoryRef` zero-init
    # constructor + `zero_dual` above). The general `@generated` below is for
    # non-float elements; it `memoryref`s an AoS Memory V, which would fail on the
    # SoA `NDualArray` that `uninit_dual(Memory{IEEEFloat})` returns.
    # (Open: `MemoryRef{Complex}` lacks the analogous SoA overload — see plan note.)
    @inline function uninit_dual(::Val{N}, p::MemoryRef{T}) where {N,T<:IEEEFloat}
        offset = Core.memoryrefoffset(p)
        len = length(p.mem)
        return NDualMemoryRef{T,N,Memory{T}}(
            p, ntuple(_ -> _memoryref_at(Memory{T}(undef, len), offset), Val(N))
        )
    end
    @inline function randn_dual(
        ::Val{N}, rng::AbstractRNG, p::MemoryRef{T}
    ) where {N,T<:IEEEFloat}
        offset = Core.memoryrefoffset(p)
        len = length(p.mem)
        return NDualMemoryRef{T,N,Memory{T}}(
            p, ntuple(_ -> _memoryref_at(Memory{T}(randn(rng, T, len)), offset), Val(N))
        )
    end
    # Memory{T} is `<: AbstractArray{T, 1}`, so its canonical V is the
    # standard NDualArray. `zero(::Memory{T})` returns a fresh Memory{T},
    # so the existing `NDualArray{T, N, 1, Memory{T}}(p)` constructor works
    # via `ntuple(_ -> zero(p), Val(N))`.
    @inline function zero_dual(::Val{N}, m::Memory{T}) where {N,T<:IEEEFloat}
        return NDualArray{T,N,1,Memory{T}}(m)
    end
    @inline function zero_dual(::Val{N}, m::Memory{Complex{R}}) where {N,R<:IEEEFloat}
        return NDualArray{Complex{R},N,1,Memory{Complex{R}}}(m)
    end
    # `uninit_dual` / `randn_dual` mirror `zero_dual` above (SoA for scalar-float
    # elements). `similar(::Memory)` is a fresh same-eltype Memory; `randn` content
    # comes via the `Memory{T}(::Vector)` constructor (`randn` itself returns an
    # Array). Without these, a `Memory{T}` slot falls through to the generic-struct
    # seed and tries to build `Memory`/`Int` from a tuple of field seeds.
    @inline function uninit_dual(::Val{N}, m::Memory{T}) where {N,T<:IEEEFloat}
        return NDualArray{T,N,1,Memory{T}}(m, ntuple(_ -> similar(m), Val(N)))
    end
    @inline function uninit_dual(::Val{N}, m::Memory{Complex{R}}) where {N,R<:IEEEFloat}
        return NDualArray{Complex{R},N,1,Memory{Complex{R}}}(
            m, ntuple(_ -> similar(m), Val(N))
        )
    end
    @inline function randn_dual(
        ::Val{N}, rng::AbstractRNG, m::Memory{T}
    ) where {N,T<:IEEEFloat}
        return NDualArray{T,N,1,Memory{T}}(
            m, ntuple(_ -> Memory{T}(randn(rng, T, length(m))), Val(N))
        )
    end
    @inline function randn_dual(
        ::Val{N}, rng::AbstractRNG, m::Memory{Complex{R}}
    ) where {N,R<:IEEEFloat}
        return NDualArray{Complex{R},N,1,Memory{Complex{R}}}(
            m, ntuple(_ -> Memory{Complex{R}}(randn(rng, Complex{R}, length(m))), Val(N))
        )
    end
    # AoS (non-float differentiable element) `Memory` / `MemoryRef` seeds,
    # mirroring the AoS array factory and `dual_type`: a per-element V memory,
    # built element-wise; a non-diff element gives `NoDual`. The IEEEFloat
    # overloads above are more specific and provide the SoA optimisation.
    @generated function zero_dual(::Val{N}, m::Memory{T}) where {N,T}
        dual_type(Val(N), Memory{T}) === NoDual && return :(NoDual())
        return quote
            v = Memory{dual_type(Val($N), T)}(undef, length(m))
            @inbounds for i in eachindex(m)
                isassigned(m, i) && (v[i] = zero_dual(Val($N), m[i]))
            end
            return v
        end
    end
    @generated function uninit_dual(::Val{N}, m::Memory{T}) where {N,T}
        dual_type(Val(N), Memory{T}) === NoDual && return :(NoDual())
        return quote
            v = Memory{dual_type(Val($N), T)}(undef, length(m))
            @inbounds for i in eachindex(m)
                isassigned(m, i) && (v[i] = uninit_dual(Val($N), m[i]))
            end
            return v
        end
    end
    @generated function randn_dual(::Val{N}, rng::AbstractRNG, m::Memory{T}) where {N,T}
        dual_type(Val(N), Memory{T}) === NoDual && return :(NoDual())
        return quote
            v = Memory{dual_type(Val($N), T)}(undef, length(m))
            @inbounds for i in eachindex(m)
                isassigned(m, i) && (v[i] = randn_dual(Val($N), rng, m[i]))
            end
            return v
        end
    end
    @generated function zero_dual(::Val{N}, p::MemoryRef{T}) where {N,T}
        dual_type(Val(N), MemoryRef{T}) === NoDual && return :(NoDual())
        return :(_memoryref_at(zero_dual(Val($N), p.mem), Core.memoryrefoffset(p)))
    end
    # `MemoryRef`'s V is built via `memoryref` over the `.mem`'s V (a plain
    # `MemoryRef` can't be constructed field-wise from a raw `Ptr`), so mirror
    # `zero_dual` rather than fall through to the generic struct seed.
    @generated function uninit_dual(::Val{N}, p::MemoryRef{T}) where {N,T}
        dual_type(Val(N), MemoryRef{T}) === NoDual && return :(NoDual())
        return :(_memoryref_at(uninit_dual(Val($N), p.mem), Core.memoryrefoffset(p)))
    end
    @generated function randn_dual(::Val{N}, rng::AbstractRNG, p::MemoryRef{T}) where {N,T}
        dual_type(Val(N), MemoryRef{T}) === NoDual && return :(NoDual())
        return :(_memoryref_at(randn_dual(Val($N), rng, p.mem), Core.memoryrefoffset(p)))
    end
end
