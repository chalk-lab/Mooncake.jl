# ── NTangent: width-aware tangent container ───────────────────────────────────

"""
    NTangent(lanes::Tuple)

Width-aware tangent container for forward-mode AD. Holds one tangent per basis
direction (lane). Each element of `lanes` must be a valid width-1 tangent for the
corresponding primal type.

Width 1 is ordinary forward mode; width N > 1 is chunked forward mode. The canonical
public forward tangent at any width is always an `NTangent`, never a bare tangent type.
"""
struct NTangent{L<:Tuple}
    lanes::L
end

Base.length(x::NTangent) = length(x.lanes)
Base.getindex(x::NTangent, i::Int) = x.lanes[i]
Base.iterate(x::NTangent, st...) = iterate(x.lanes, st...)

# ── Width-aware tangent_type ──────────────────────────────────────────────────

"""
    tangent_type(::Val{N}, ::Type{P})

Width-aware tangent type query. Returns the canonical tangent representation for
primal type `P` at width `N`.

- `Val(0)` → `NoTangent` (primal passthrough, no tangent needed)
- `Val(N)` where `tangent_type(P) == NoTangent` → `NoTangent`
- `Val(N)` otherwise → `NTangent{NTuple{N, tangent_type(P)}}`
"""
tangent_type(::Val{0}, ::Type{P}) where {P} = NoTangent
function tangent_type(::Val{N}, ::Type{P}) where {N,P}
    T = tangent_type(P)
    T === NoTangent && return NoTangent
    # Width 1 still collapses to bare `T`: the audit's full step-5 migration
    # would wrap to `NTangent{Tuple{T}}` here, but that produces a slot/value
    # mismatch for generic `Dual{P,T}` runtime constructions (the OC slot
    # would expect `Dual{P, NTangent{Tuple{T}}}` while every `zero_dual` /
    # `Dual(p,t)` site still emits the bare form). Migrating those sites is
    # the remaining bulk of audit step 5; this collapse stays as the
    # transitional shim until the rule/transform side is updated.
    N == 1 && return T
    return NTangent{NTuple{N,T}}
end

# ── Width-aware dual_type ─────────────────────────────────────────────────────

"""
    dual_type(::Val{N}, ::Type{P})

Width-aware forward value type query.

- `Val(0)` → `P` (primal passthrough)
- `Val(N)`, concrete `P` → `Dual{P, tangent_type(Val(N), P)}`
- abstract/union `P` → `Dual` (bare, for compiler flexibility)
"""
# `@unstable`: return type depends on the type-domain shape of `P` (Union
# splitting, Tuple field concreteness). Callers force-specialise via
# `Val(N)` constants or accept a bare `Dual`.
@unstable function dual_type(::Val{N}, ::Type{P}) where {N,P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(Val(N), P.a),dual_type(Val(N), P.b)}
    (P isa UnionAll || P == UnionAll) && return Dual

    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{dual_type(Val(N), P_split.a),dual_type(Val(N), P_split.b)}
        end
    end

    # Concrete Tuple: element-wise lifting — each field type is individually lifted.
    if isconcretetype(P) && P <: Tuple
        return Tuple{(dual_type(Val(N), fieldtype(P, i)) for i in 1:fieldcount(P))...}
    end

    # Concrete NamedTuple: element-wise lifting symmetric to Tuple. The inner
    # tuple of field types lifts via the Tuple branch; the outer NamedTuple
    # preserves field names.
    if isconcretetype(P) && P <: NamedTuple
        names = fieldnames(P)
        InnerTup = Tuple{(dual_type(Val(N), fieldtype(P, i)) for i in 1:fieldcount(P))...}
        return NamedTuple{names,InnerTup}
    end

    # Concrete IMMUTABLE struct with `tangent_type(P) <: Tangent`, all fields
    # always initialised, and only "lift-safe" field types: recursive
    # NamedTuple lift. Each field's `dual_type` is the canonical V for that
    # field's primal type; the inner V mirrors the struct's field structure
    # as a `NamedTuple{names, Tuple{Vᵢ…}}`. This generalises the per-wrapper
    # structural lift (Diagonal/Adjoint/SubArray) to arbitrary immutable
    # structs and closes the silent-corruption gap for in-place mutation
    # through struct fields. See `notes/mooncake/dual-types.md` §13.
    #
    # Excluded:
    # - Mutable structs (`MutableTangent` tangent_type): their `lsetfield!`
    #   rules need a mutable inner V to support `s.field = x`, but
    #   `NamedTuple` is immutable. Keep the existing parallel
    #   `Dual{P, MutableTangent}` form.
    # - Structs with potentially-undef fields (`PossiblyUninitTangent` in
    #   their `tangent_field_types`): the lift would call
    #   `getfield(primal, name)` on undef fields. Keep the legacy form.
    # - Structs with nested-struct fields (e.g. `Broadcasted`'s `args` is a
    #   `Tuple` containing an `Extruded` struct, which would itself recurse
    #   through this lift, losing its struct identity inside the Tuple V).
    #   `_is_lift_safe_field_type` walks Tuple/NamedTuple containers to
    #   detect any non-flat struct lurking inside. Keep the legacy form.
    #
    # Specific per-wrapper `dual_type` overloads (e.g. `Diagonal{T,Vector{T}}`
    # in `nfwd/NfwdMooncake.jl`) are more specific and dispatch first, so
    # this branch only fires for immutable structs without an explicit lift.
    if N >= 1 && _uses_structural_dual_type(P)
        names = fieldnames(P)
        InnerTup = Tuple{(dual_type(Val(N), fieldtype(P, i)) for i in 1:fieldcount(P))...}
        return NamedTuple{names,InnerTup}
    end

    return isconcretetype(P) ? Dual{P,tangent_type(Val(N), P)} : Dual
end

@inline function _uses_structural_dual_type(::Type{P}) where {P}
    # Audit step 4: structural `dual_type` mirrors structural `tangent_type`.
    # Previously this gate also required `all(_is_lift_safe_field_type,
    # fieldtypes(P))` — a narrow allowlist that forced struct fields to be
    # IEEEFloat / Complex{IEEEFloat} / NoTangent / Array{IEEEFloat} / a few
    # specialised wrappers. Nested user-defined structs (`OuterAudit`
    # containing `InnerAudit`) fell back to the parallel `Dual{P, Tangent}`
    # form even when their tangent_type was already structurally recursive.
    # Per `~/notes/mooncake/primal-mode-branch-audit.md`:
    # "There should be no broad generic exclusion gate such as
    # `_is_lift_safe_field_type`. Non-generic representations must be
    # explicit `dual_type` overloads for chosen types."
    # Field-type-specific exceptions live as `dual_type` overloads (e.g.
    # CuArray-wrappers in MooncakeCUDAExt) and dispatch first by Julia's
    # method specificity, so this generic branch only fires when the
    # recursive descent is structurally correct.
    return isconcretetype(P) &&
           !ismutabletype(P) &&
           fieldcount(P) > 0 &&
           _is_user_definable_type(P) &&
           tangent_type(P) <: Tangent &&
           _uses_structural_tangent_type(P) &&
           all(always_initialised(P))
end

# `_is_user_definable_type(P)` — heuristic gate for the recursive struct lift.
# Returns `true` for types defined outside `Base` and `Core` (and their
# submodules), `false` for built-in types like `Base.Broadcast.Extruded`.
# Built-in types are constructed by the IR through `_new_(::Type{T}, …)`
# call sites that expect the original struct identity to be preserved;
# recursive lifting would replace them with a `NamedTuple` and break the
# construction. User-defined types stay within the lifted form throughout.
@inline function _is_user_definable_type(::Type{P}) where {P}
    isconcretetype(P) || return false
    m = parentmodule(P)
    while true
        (m === Base || m === Core || m === LinearAlgebra) && return false
        m === parentmodule(m) && return true
        m = parentmodule(m)
    end
end

@inline function _uses_structural_tangent_type(::Type{P}) where {P}
    # Generated callers use this predicate, so avoid reflection (`which` is
    # forbidden there). Match the default struct tangent by shape instead.
    names = fieldnames(P)
    field_tangent_types = Tuple{
        ntuple(i -> tangent_type(fieldtype(P, i)), Val(fieldcount(P)))...
    }
    return tangent_type(P) === Tangent{NamedTuple{names,field_tangent_types}}
end

dual_type(::Val{0}, ::Type{P}) where {P} = P

dual_type(::Val{0}, ::Type{Type{P}}) where {P} = Type{P}

function dual_type(::Val{N}, p::Type{Type{P}}) where {N,P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

# ── Dual ──────────────────────────────────────────────────────────────────────

"""
    Dual(primal::P, tangent::T)

Used to pair together a `primal` value and a `tangent` to it. In the context of foward mode
AD (aka computing Frechet derivatives), `primal` governs the point at which the derivative
is computed, and `tangent` the direction in which it is computed.

Must satisfy `tangent_type(P) == T`.
"""
struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent

# `tangent(x, dir)` — per-lane tangent accessor. Delegates to the
# specialised `_tangent_dir` fast paths (NDual, Complex{NDual}, NTangent,
# Array{<:NDual}, etc.) defined in `nfwd/NfwdMooncake.jl`. The two-argument
# overload must not materialise all lanes before selecting one — extracting
# a single direction from a width-N container should remain O(container
# size), not O(N × container size).
@inline tangent(x, dir::Integer) = _tangent_dir(x, dir)

# `primal` / `tangent` on a bare element-wise tuple of inner duals (the inner
# `V` of a `Lifted{<:Tuple, N}`). Recursive map so nested Tuple-of-Dual works.
_field_primal(x) = x
_field_primal(x::Dual) = primal(x)
_field_tangent(x) = zero_tangent(x)
_field_tangent(x::Dual) = tangent(x)
primal(t::Tuple) = map(_field_primal, t)
tangent(t::Tuple) = map(_field_tangent, t)
# Bare NamedTuple inner V (struct-primal recursive lift, see §13 of
# notes/mooncake/dual-types.md): per-field `primal` / `tangent`. Mirrors
# the Tuple bare-V conventions above.
primal(t::NamedTuple) = map(_field_primal, t)
tangent(t::NamedTuple) = map(_field_tangent, t)
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
# Dual can be safely shared without copying
_copy(x::P) where {P<:Dual} = x

"""
    extract(x::Dual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::Dual) = primal(x), tangent(x)

zero_dual(x) = Dual(x, zero_tangent(x))
randn_dual(rng::AbstractRNG, x) = Dual(x, randn_tangent(rng, x))

# Generic width-N fallback. For NoTangent primals, returns `Dual(x, NoTangent())`
# (matches `dual_type(Val(N), P) == Dual{P,NoTangent}`). For non-trivial
# tangent_type without a width-N specialisation (e.g. Task, StepRangeLen),
# fall back to the no-width zero_dual — at canonical V the inner shape is
# `Dual{P, tangent_type(P)}`, which matches what `zero_dual(x)` returns.
# Specialised IEEEFloat / Complex / array / Memory overloads live in
# `nfwd/NfwdMooncake.jl`.
@inline function zero_dual(w::Val, x)
    if tangent_type(_typeof(x)) === NoTangent
        return Dual(x, NoTangent())
    end
    return zero_dual(x)
end

@unstable function dual_type(::Type{P}) where {P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
    # Use `isa` not `<:`: generators like `NTuple{N,Int} where N` are instances of
    # UnionAll but not subtypes of it (`NTuple{N,Int} where N <: UnionAll` is false).
    # `P == UnionAll` handles the UnionAll metatype itself (`UnionAll isa UnionAll` is false).
    (P isa UnionAll || P == UnionAll) && return Dual # P is abstract, tangent type unknown.

    # Union Splitting
    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)

        # If there is exactly one Union field, split it to help the compiler
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{dual_type(P_split.a),dual_type(P_split.b)}
        end
    end

    return isconcretetype(P) ? Dual{P,tangent_type(P)} : Dual
end

function dual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

"""
    verify_dual_type(x::Dual)

Check that the type of `tangent(x)` is the tangent type of the type of `primal(x)`.
Accepts both legacy bare tangents and width-aware NTangent-wrapped tangents.
"""
function verify_dual_type(x::Dual)
    P = typeof(primal(x))
    T = typeof(tangent(x))
    T === NoTangent && return tangent_type(P) === NoTangent
    if T <: NTangent
        N = fieldcount(T.parameters[1])
        return T === tangent_type(Val(N), P) ||
               (N == 1 && fieldtype(T.parameters[1], 1) === tangent_type(P))
    end
    # Legacy width-1 path: bare tangent without NTangent wrapper
    return tangent_type(P) == T
end

function error_if_incorrect_dual_types(duals::Vararg{Dual,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primals = map(primal, duals)
        tangents = map(tangent, duals)
        throw(ArgumentError("""
        Tangent types do not match primal types:
          - primal types:           $(map(typeof, primals))
          - provided tangent types: $(map(typeof, tangents))
          - required tangent types: $(map(tangent_type, map(typeof, primals)))
        """))
    end
end

@inline uninit_dual(x::P) where {P} = Dual(x, uninit_tangent(x))

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

# ── Inner-type constructors (Layer-2 dual targets) ────────────────────────────
# These let the `Lifted{P, N}(primal, tangent)` 2-arg constructor delegate to
# `dual_type(Val(N), P)(primal, tangent)` without rule bodies needing to choose
# the inner shape. Each inner type accepts the canonical tangent shapes a rule
# body may produce: scalar broadcast, pre-computed lanes, or NTangent-wrapped.

# 1-tuple convenience for the bare `Dual{P, T}` width-1 inner: lets per-lane rule
# bodies use the same `ntuple(closure, Val(N))` pattern at any width — at width 1
# the closure produces a 1-tuple `(t,)` which this method unwraps to scalar `t`.
Dual{P,T}(value, t::NTuple{1,T}) where {P,T} = Dual{P,T}(value, t[1])

# Chunked structured `Dual{P, NTangent{NTuple{N, T}}}`: pre-computed lanes wrap
# in NTangent; scalar tangent broadcasts across N lanes.
function (::Type{Dual{P,NTangent{NTuple{N,T}}}})(value::P, lanes::NTuple{N,T}) where {P,N,T}
    return Dual(value, NTangent(lanes))
end
function (::Type{Dual{P,NTangent{NTuple{N,T}}}})(value::P, tangent::T) where {P,N,T}
    return Dual(value, NTangent(ntuple(_ -> tangent, Val(N))))
end

# ── Lifted: Layer-3 wrapper struct ───────────────────────────────────────────

"""
    Lifted{P, N, V}

Forward-mode slot wrapper. `P` is the primal type, `N` is the AD width
(`1` ordinary, `N >= 2` chunked), and `V === dual_type(Val(N), P)` is the
inner dual shape. Slot identity `(P, N)` is dispatch-visible at the wrapper
level; the inner shape `V` is hidden.

Mirrors `CoDual{Tx, Tdx}` for forward mode. The extra `N` parameter and the
fact that `V` varies (NDual, Complex{<:NDual}, Array{<:NDual}, …) reflect
forward mode supporting multiple inner shapes; `V` is fully determined by
`(P, N)` via `dual_type`.

`Lifted` exists for every `P` at `N >= 1`, including `Tuple` primals (which
lift to a single outer `Lifted` whose `V` is a `Tuple` of bare inner duals —
`Lifted` never nests inside another `Lifted`'s `V`). `Val(0)` slots are
unwrapped (primal passthrough) and are not represented by `Lifted`.
"""
struct Lifted{P,N,V}
    value::V
end

"""
    verify_lifted_type(x::Lifted{P, N, V})

Check the canonical Lifted-slot invariant: `V === dual_type(Val(N), P)`, the
stored `value::V` matches that type, and the inner dual value itself
validates via `verify_dual_type`. Rejects nested `Lifted` inside `V`.

Returns `true` if `x` is canonical, `false` otherwise. Companion to
`verify_dual_type`, which validates only the inner dual value; this
function additionally validates the outer slot wrapper.
"""
function verify_lifted_type(x::Lifted{P,N,V}) where {P,N,V}
    # Reject nested `Lifted` — slot wrappers must not appear inside V.
    _contains_lifted(V) && return false
    # Canonical V for this (P, N) must match.
    V === dual_type(Val(N), P) || return false
    # Stored value must actually be of declared V.
    x.value isa V || return false
    # And the inner value must itself be a valid dual.
    return verify_dual_type(x.value)
end

# Type-level walker: detect any `Lifted` nested in T (including inside Tuple /
# NamedTuple V shapes). Fully type-domain, no runtime cost in the foldable
# generic case.
@inline _contains_lifted(::Type{<:Lifted}) = true
@inline _contains_lifted(::Type{T}) where {T<:Tuple} = any(_contains_lifted, fieldtypes(T))
@inline _contains_lifted(::Type{NamedTuple{names,T}}) where {names,T} = _contains_lifted(T)
@inline _contains_lifted(::Type) = false

# Lifted slot wrappers: the canonical V invariant guarantees
# `V === dual_type(Val(N), P)`, so the slot is well-typed by construction.
verify_dual_type(::Lifted) = true
# Bare canonical-V Tuple (e.g. `Tuple{Dual{Int,NoTangent}, Dual{Float64,...}}`)
# from `_unlift(::Lifted{<:Tuple, 1})` — verify each element.
verify_dual_type(t::Tuple) = all(verify_dual_type, t)
verify_dual_type(t::NamedTuple) = all(verify_dual_type, values(t))
# Bare canonical-V leaf-scalar shapes (`NDual`, `Complex{<:NDual}`,
# `<:NTangent`) leak through helper-API boundaries but still represent valid
# inner dual values. Their concrete overloads are added in `nfwd/NfwdMooncake.jl`
# once `NDual` is defined; the fallback below accepts everything else as
# user-typed primitive values flowing through outside the strict slot path.

# 1-arg: wrap an already-built inner. V is inferred from typeof(value).
@inline Lifted{P,N}(value) where {P,N} = Lifted{P,N,typeof(value)}(value)

# 2-arg: build the inner via the inner type's own constructor methods. Mirrors
# `CoDual(x, dx)` — pass `(primal, tangent)` and the wrapper takes care of the
# rest. The dispatch on inner shape lives in the inner type's constructors.
@inline function Lifted{P,N}(primal, tangent) where {P,N}
    InnerT = dual_type(Val(N), P)
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end

# Type-slot specialisation: `dual_type(Val(N), Type{P_user})` may substitute the
# inner type parameter (e.g. `Type{Memory{Float64}}` → V-primal
# `Type{Memory{NDual{Float64,N}}}` per the override in `nfwd/NfwdMooncake.jl`)
# so that the OC slot's inner V matches what IR-emit produces at runtime. The
# user-facing path (test framework, direct callers) still passes the
# unsubstituted `P_user`, which would fail the auto-generated
# `Dual{Type{P_lifted}, NoTangent}` constructor. Detect the Type-slot shape and
# substitute to `P_lifted` here so both paths converge. Safe when `P_lifted ==
# P_user` (e.g. `Type{Float64}` lifts to itself); the substitution is a no-op.
@inline function Lifted{P,N}(primal::Type, tangent::NoTangent) where {P<:Type,N}
    InnerT = dual_type(Val(N), P)
    if InnerT isa DataType && InnerT <: Dual && InnerT.parameters[1] <: Type
        P_lifted = InnerT.parameters[1].parameters[1]
        return Lifted{P,N,InnerT}(InnerT(P_lifted, NoTangent()))
    end
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end

# 3-param 2-arg ctor: callers that obtained `T = lifted_type(Val(N), P)` (the
# fully-parameterised slot type) and want to build it with `(primal, tangent)`.
# Forwards to the 2-param 2-arg form.
@inline Lifted{P,N,V}(primal, tangent) where {P,N,V} = Lifted{P,N}(primal, tangent)

# Tuple-primal special case: `dual_type(Val(N), P<:Tuple)` is a bare
# element-wise `Tuple{...}` of inner duals, which has no user-defined 2-arg
# constructor. Build the inner tuple here element-wise instead. Nested tuple
# fields recurse via `_inner_dual_for_field` so that
# `Tuple{NDual, Tuple{NDual, Vector{NDual}}, ...}` is built leaf-by-leaf.
#
# `@generated` for the same reason as `_wrap_oc_args` / `_wrap_rule_result`:
# a runtime `ntuple` with a closure body indexing `primal[i]` / `tangent[i]`
# leaves the inner result Union-typed (each Vᵢ varies), forcing a heap alloc
# for the resulting Tuple. Unrolling per-field gives static dispatch on each
# `_inner_dual_for_field(Vᵢ, ...)` call.
@inline @generated function Lifted{P,N}(
    primal::P, tangent::Tup
) where {P<:Tuple,N,Tup<:Tuple}
    n = fieldcount(P)
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: Tuple)
        # Fall back to the runtime path when InnerT isn't a concrete Tuple of
        # known inner V's (e.g. abstract P).
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, tangent))
    end
    inner_exprs = map(1:n) do i
        :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], tangent[$i]))
    end
    return quote
        return Lifted{$P,$N,$InnerT}(($(inner_exprs...),))
    end
end

# Build a single field's inner dual value. For non-Tuple fields, defer to the
# field type's own 2-arg constructor (NDual, Vector{NDual}, Dual, etc.). For
# Tuple-typed fields, recurse element-wise so a nested Tuple-of-Dual builds
# without trying `Tuple{...}(::Tuple, ::Tuple)` (which has no ctor).
@inline function _inner_dual_for_field(::Type{V}, primal::P, tangent::T) where {V,P,T}
    if V isa DataType && V <: Dual && V.parameters[1] === P && !(T <: V.parameters[2])
        return Dual(primal, tangent)
    end
    return V(primal, tangent)
end
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Base.Broadcast.Extruded, tangent::Tangent
) where {V<:Base.Broadcast.Extruded}
    inner_x = _inner_dual_for_field(fieldtype(V, 1), primal.x, val(tangent.fields.x))
    # Broadcast metadata is non-differentiable; only `x` carries tangent data.
    return V(inner_x, primal.keeps, primal.defaults)
end
# When the tangent is already the canonical V (e.g. test_rule passes a `Dual`
# whose tangent slot carries an NDual or Array{NDual} directly), pass it
# through — re-wrapping would invoke a non-existent 2-arg ctor.
@inline _inner_dual_for_field(::Type{V}, primal, tangent::V) where {V} = tangent
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Tuple, tangent::Tuple
) where {V<:Tuple}
    return ntuple(
        i -> _inner_dual_for_field(fieldtype(V, i), primal[i], tangent[i]),
        Val(fieldcount(V)),
    )
end
# NoTangent broadcast for Tuple/NamedTuple V — recurse element-wise with NoTangent
# at each leaf. Required when a parent `Lifted{<:Tuple, N}(primal, ::NoTangent)`
# constructor visits a nested Tuple field.
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Tuple, ::NoTangent
) where {V<:Tuple}
    return ntuple(
        i -> _inner_dual_for_field(fieldtype(V, i), primal[i], NoTangent()),
        Val(fieldcount(V)),
    )
end
@inline function _inner_dual_for_field(
    ::Type{V}, primal::NamedTuple, ::NoTangent
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(fieldtype(InnerTup, i), values(primal)[i], NoTangent()),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end
# NamedTuple-typed nested fields recurse element-wise like Tuple.
@inline function _inner_dual_for_field(
    ::Type{V}, primal::NamedTuple, tangent::NamedTuple
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(
            fieldtype(InnerTup, i), values(primal)[i], values(tangent)[i]
        ),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end
# Nested-struct case: V is the recursive NamedTuple lift of an immutable
# struct field whose tangent is `Tangent`. Recurse element-wise, extracting
# per-field primals from the struct and per-field tangents from the
# Tangent's fields NamedTuple (with `_get_tangent_field_for_lift` unwrapping
# `PossiblyUninitTangent`).
@inline function _inner_dual_for_field(
    ::Type{V}, primal, tangent::Tangent
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(
            fieldtype(InnerTup, i),
            getfield(primal, names[i]),
            _get_tangent_field_for_lift(tangent, names[i]),
        ),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end

# Tuple-primal with `NoTangent` (whole-tuple) tangent — common when a `Tuple`
# slot holds non-differentiable elements (e.g. `Tuple{Int}`). Build the inner
# tuple element-wise with `NoTangent` for each field. Use
# `_inner_dual_for_field` so nested Tuple fields (e.g. `Tuple{Tuple{}, Int}`)
# recurse properly instead of trying to call `Tuple{}(::Tuple, ::NoTangent)`
# which has no method.
@inline @generated function Lifted{P,N}(primal::P, ::NoTangent) where {P<:Tuple,N}
    n = fieldcount(P)
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: Tuple)
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, NoTangent()))
    end
    inner_exprs = map(1:n) do i
        :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], NoTangent()))
    end
    return quote
        return Lifted{$P,$N,$InnerT}(($(inner_exprs...),))
    end
end

# Tuple-primal with `NTangent` tangent — width-N chunked vararg case where
# `_group_vararg_dual` produces an `NTangent` lane structure. Extract per
# element, per direction.
@inline _all_notangent_lanes(t::NTangent) = all(_all_notangent_lane, t.lanes)
@inline _all_notangent_lane(t::NoTangent) = true
@inline _all_notangent_lane(t::Tuple) = all(_all_notangent_lane, t)
@inline _all_notangent_lane(t::NamedTuple) = all(_all_notangent_lane, values(t))
@inline _all_notangent_lane(_) = false

@inline function Lifted{P,N}(primal::P, tangent::NTangent) where {P<:Tuple,N}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: Tuple)
        _all_notangent_lanes(tangent) && return Lifted{P,N}(primal, NoTangent())
        return Lifted{P,N,InnerT}(InnerT(primal, tangent))
    end
    lanes = tangent.lanes
    inner = ntuple(Val(fieldcount(P))) do i
        Vi = fieldtype(InnerT, i)
        if N == 1
            Vi(primal[i], lanes[1][i])
        else
            partials = ntuple(d -> lanes[d][i], Val(N))
            Vi(primal[i], _all_notangent_lane(partials) ? NoTangent() : partials)
        end
    end
    return Lifted{P,N,InnerT}(inner)
end

# NamedTuple-primal: parallel to the Tuple ctor. Inner V is a
# `NamedTuple{names, Tuple{V_i...}}` of bare inner duals; build element-wise.
@inline function Lifted{P,N}(primal::P, tangent::NTangent) where {P<:NamedTuple,N}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        _all_notangent_lanes(tangent) && return Lifted{P,N}(primal, NoTangent())
        return Lifted{P,N,InnerT}(InnerT(primal, tangent))
    end
    names = fieldnames(P)
    InnerTup = InnerT.parameters[2]
    lanes = tangent.lanes
    inner = ntuple(Val(fieldcount(P))) do i
        Vi = fieldtype(InnerTup, i)
        if N == 1
            Vi(primal[i], lanes[1][i])
        else
            partials = ntuple(d -> lanes[d][i], Val(N))
            Vi(primal[i], _all_notangent_lane(partials) ? NoTangent() : partials)
        end
    end
    return Lifted{P,N,InnerT}(NamedTuple{names}(inner))
end
@inline @generated function Lifted{P,N}(
    primal::P, tangent::NamedTuple{names}
) where {P<:NamedTuple{names},N} where {names}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, tangent))
    end
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = map(1:n) do i
        :(_inner_dual_for_field($(fieldtype(InnerTup, i)), primal[$i], values(tangent)[$i]))
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end
@inline @generated function Lifted{P,N}(
    primal::P, ::NoTangent
) where {P<:NamedTuple{names},N} where {names}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, NoTangent()))
    end
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = map(1:n) do i
        :($(fieldtype(InnerTup, i))(primal[$i], NoTangent()))
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end

# Struct-primal with `Tangent` tangent — recursive lift: the inner V is a
# `NamedTuple{fieldnames(S), Tuple{Vᵢ…}}` mirroring the struct's field
# structure with each field already in canonical V form. Each field is built
# via `_inner_dual_for_field`, which routes nested Tuple / NamedTuple /
# struct fields to their respective recursive constructors and leaf
# canonical-V types (`NDual`, `Vector{NDual}`, …) to their 2-arg ctors.
# `_get_tangent_field_for_lift` unwraps `PossiblyUninitTangent` slots so
# field tangents pass through unwrapped. Mutable structs keep the existing
# parallel-Dual path (their `dual_type` does not return `<: NamedTuple`).
@inline @generated function Lifted{P,N}(primal::P, tangent::Tangent) where {P,N}
    # `try` to resolve InnerT at expansion time. The static path emits a
    # tight ctor (recursive struct lift). On expansion failure (ext-typed
    # primitive-leaf world-age error inside `tangent_type`), defer the
    # whole construction to runtime.
    InnerT = try
        dual_type(Val(N), P)
    catch
        return :(_lifted_struct_runtime_fallback($P, Val($N), primal, tangent))
    end
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        # Non-struct lift: defer to the inner type's own 2-arg constructor.
        return :(Lifted{$P,$N,$InnerT}($InnerT(primal, tangent)))
    end
    names = fieldnames(P)
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = map(1:n) do i
        :(_inner_dual_for_field(
            $(fieldtype(InnerTup, i)),
            getfield(primal, $(QuoteNode(names[i]))),
            _get_tangent_field_for_lift(tangent, $(QuoteNode(names[i]))),
        ))
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end

@inline function Lifted{P,N}(
    primal::P, tangent::Tangent
) where {P<:Base.Broadcast.Extruded,N}
    InnerT = dual_type(Val(N), P)
    XInnerT = fieldtype(InnerT, 1)
    inner_x = _inner_dual_for_field(XInnerT, getfield(primal, :x), val(tangent.fields.x))
    # Broadcast metadata is non-differentiable; only `x` carries tangent data.
    return Lifted{P,N,InnerT}(InnerT(inner_x, primal.keeps, primal.defaults))
end

# Helper: extract a field's tangent value from a `Tangent`, unwrapping
# `PossiblyUninitTangent` wrappers so the value passes to the leaf constructor
# in its bare form. Mirrors `_get_tangent_field` in `rules/misc.jl` but lives
# here to avoid a load-order dependency.
@inline _get_tangent_field_for_lift(t::Tangent, name) = val(getfield(t.fields, name))

# Runtime fallback for `Lifted{P, N}(primal::P, tangent::Tangent)` when the
# `@generated` expansion can't resolve `InnerT = dual_type(Val(N), P)` —
# typically because the recursive `tangent_type` descent through `P`'s
# fields hits a primitive-leaf world-age boundary (e.g. CuArray's nested
# `CuPtr{Nothing}` chain). At runtime the call uses the latest world.
@inline function _lifted_struct_runtime_fallback(
    ::Type{P}, ::Val{N}, primal, tangent
) where {P,N}
    InnerT = dual_type(Val(N), P)
    if InnerT isa DataType && InnerT <: NamedTuple
        names = fieldnames(P)
        InnerTup = InnerT.parameters[2]
        fields = ntuple(Val(fieldcount(P))) do i
            _inner_dual_for_field(
                fieldtype(InnerTup, i),
                getfield(primal, names[i]),
                _get_tangent_field_for_lift(tangent, names[i]),
            )
        end
        return Lifted{P,N,InnerT}(NamedTuple{names}(fields))
    end
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end

# Accessors: delegate to the inner's own primal/tangent. Tuple primals need
# field-type-aware reconstruction: a field can have primal type `CoDual` while
# its canonical inner V is a structural `NamedTuple`, so a plain `map(primal, ...)`
# would erase the field's original type.
primal(d::Lifted) = primal(d.value)
tangent(d::Lifted) = tangent(d.value)
@generated function primal(d::Lifted{P,N,V}) where {P<:Tuple,N,V<:Tuple}
    isconcretetype(P) || return :(map(primal, d.value))
    exprs = map(1:fieldcount(P)) do i
        Pi = fieldtype(P, i)
        Vi = fieldtype(V, i)
        if (Pi <: Tuple && Vi <: Tuple) || (Vi <: NamedTuple && !(Pi <: NamedTuple))
            :(primal(Lifted{$Pi,$N,$Vi}(d.value[$i])))
        else
            :(primal(d.value[$i]))
        end
    end
    return :(($(exprs...),))
end
@generated function tangent(d::Lifted{P,N,V}) where {P<:Tuple,N,V<:Tuple}
    isconcretetype(P) || return :(map(tangent, d.value))
    exprs = map(1:fieldcount(P)) do i
        Pi = fieldtype(P, i)
        Vi = fieldtype(V, i)
        if (Pi <: Tuple && Vi <: Tuple) || (Vi <: NamedTuple && !(Pi <: NamedTuple))
            :(tangent(Lifted{$Pi,$N,$Vi}(d.value[$i])))
        else
            :(tangent(d.value[$i]))
        end
    end
    return :(($(exprs...),))
end
function primal(d::Lifted{P,N,V}) where {names,P<:NamedTuple,N,V<:NamedTuple{names}}
    return map(primal, d.value)
end
function tangent(d::Lifted{P,N,V}) where {names,P<:NamedTuple,N,V<:NamedTuple{names}}
    return map(tangent, d.value)
end

# Struct-primal accessors: the inner V is a `NamedTuple{fieldnames(P), Tuple{Vᵢ…}}`
# (recursive lift), but `P` itself is a struct, not a NamedTuple. Reconstruct
# the struct via `_new_` (Mooncake's bypass-constructor primitive) from the
# per-field primals; build a `Tangent` / `MutableTangent` whose fields carry
# the per-field tangent (NTangent-bearing for IEEEFloat-leaf fields, mirroring
# the existing `tangent(::Array{<:NDual})` convention). This shape is used
# for address-map tracking; `_tangent_dir(slot, i)` produces the bare-tangent
# shape used by `_dot` for FD comparison.
@generated function primal(d::Lifted{P,N,V}) where {P,N,V<:NamedTuple{names}} where {names}
    P <: NamedTuple && return :(map(primal, d.value))   # earlier method handles this
    field_exprs = [:(primal(d.value.$n)) for n in names]
    return :(_new_($P, $(field_exprs...)))
end
@generated function tangent(d::Lifted{P,N,V}) where {P,N,V<:NamedTuple{names}} where {names}
    P <: NamedTuple && return :(map(tangent, d.value))   # earlier method handles this
    pairs = [Expr(:kw, n, :(tangent(d.value.$n))) for n in names]
    return :(Tangent((; $(pairs...))))
end

"""
    extract(d::Lifted)

Return the 2-tuple `(primal(d), tangent(d))`.
"""
extract(d::Lifted) = (primal(d), tangent(d))

# Wrap / unwrap mechanics: `_lift(Val(N), P, _unlift(d))` is a typed identity.
# `Val(0)` is the primal passthrough — `lifted_type(Val(0), P) = P`, so the
# value passes through without wrapping.
_lift(::Val{0}, ::Type{P}, value) where {P} = value
_lift(::Val{N}, ::Type{P}, inner) where {N,P} = Lifted{P,N}(inner)
_unlift(d::Lifted) = d.value

Base.copy(d::Lifted{P,N}) where {P,N} = Lifted{P,N}(copy(d.value))
# Lifted can be safely shared without copying (same as Dual).
_copy(d::L) where {L<:Lifted} = d

# ── lifted_type: Layer-3 slot type query ─────────────────────────────────────

"""
    lifted_type(::Val{N}, ::Type{P})

Width-aware Layer-3 slot type query. Returns the wrapped slot type for primal
`P` at width `N`.

- `Val(0)` → `P` (primal passthrough)
- `Val(N)` → `Lifted{P, N, dual_type(Val(N), P)}`

Companion to `tangent_type` (Layer 1) and `dual_type` (Layer 2). The single
top-level slot-type query — used both by rule bodies (computing the type of
a result slot) and by IR-emit at lift sites (computing the type of an OC
slot). Symmetric with reverse mode's `codual_type`.
"""
lifted_type(::Val{0}, ::Type{P}) where {P} = P
function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    V = dual_type(Val(N), P)
    # Concrete `(P, V)`: produce the fully-parameterised slot type.
    isconcretetype(V) && return Lifted{P,N,V}
    # Abstract `P` (e.g. `Any`, `Real`): keep the width `N` bound and let `P`
    # widen via the existing `Q<:P` constraint and `V` widen freely. Returning
    # the bare `Lifted` UnionAll (the previous behaviour) was unsound — a
    # width-2 abstract slot would accept a width-1 lifted runtime value, since
    # `Lifted{Float64, 1, NDual{Float64, 1}} <: Lifted`. Preserving `N`
    # rejects cross-width substitution while still accepting any concrete
    # subtype of `P` at the bound width.
    return Lifted{Q,N,V_inner} where {Q<:P,V_inner}
end

# ── Layer-3 seed factories: return wrapped Lifted slot ───────────────────────
# One-line delegations to the Layer-2 factories. Both routes consult the same
# `dual_type` table for the inner shape; Layer 3 just adds the outer wrap.
# `Val(0)` is the primal passthrough — `lifted_type(Val(0), P) === P`.

"""
    zero_lifted(::Val{N}, x)

Width-aware Layer-3 seed factory. Returns a `Lifted{typeof(x), N}` wrapping
the canonical zero dual at width `N`, or the bare primal `x` at `Val(0)`.

Companion to `zero_dual` (Layer 2). The result type matches
`lifted_type(Val(N), typeof(x))`.
"""
# Val(0) is the primal passthrough — bare `x` flows unchanged at every
# primal shape. Explicit Tuple/NamedTuple/Type overloads disambiguate
# against the matching width-N variants below (Aqua flagged these as
# `zero_lifted(::Val{0}, ::Tuple)` etc.). Keep the generic `Val(0)` last
# so it acts as the catch-all.
@inline zero_lifted(::Val{0}, x::Tuple) = x
@inline zero_lifted(::Val{0}, x::NamedTuple) = x
@inline zero_lifted(::Val{0}, x::Type) = x
@inline zero_lifted(::Val{0}, x) = x
@inline zero_lifted(w::Val{N}, x) where {N} = Lifted{typeof(x),N}(zero_dual(w, x))
@inline function zero_lifted(w::Val{N}, x::Type{P}) where {N,P}
    P_slot = @isdefined(P) ? Type{P} : typeof(x)
    return Lifted{P_slot,N}(zero_dual(w, x))
end
# Tuple/NamedTuple primal: produce canonical V (a bare element-wise tuple of
# inner duals) per AGENTS.md tuple-lifting. Without this overload, the generic
# `zero_dual(w, ::Tuple)` returns `Dual(tuple, NoTangent)` which violates the
# canonical V invariant and breaks `tangent(::Lifted{<:Tuple})` (which calls
# `map(tangent, .value)` expecting a bare Tuple).
@inline function zero_lifted(w::Val{N}, x::Tuple) where {N}
    inner = ntuple(i -> zero_lifted_inner(w, x[i]), Val(length(x)))
    return Lifted{typeof(x),N}(inner)
end
@inline function zero_lifted(w::Val{N}, x::NamedTuple{names}) where {N,names}
    inner = ntuple(i -> zero_lifted_inner(w, values(x)[i]), Val(length(x)))
    return Lifted{typeof(x),N}(NamedTuple{names}(inner))
end
# Same disambiguation pattern for the recursive helper.
@inline zero_lifted_inner(::Val{0}, x::Tuple) = x
@inline zero_lifted_inner(::Val{0}, x::NamedTuple) = x
@inline zero_lifted_inner(::Val{0}, x) = x
@inline zero_lifted_inner(w::Val{N}, x) where {N} = zero_dual(w, x)
@inline zero_lifted_inner(w::Val{N}, x::Tuple) where {N} = ntuple(
    i -> zero_lifted_inner(w, x[i]), Val(length(x))
)
@inline zero_lifted_inner(w::Val{N}, x::NamedTuple{names}) where {N,names} = NamedTuple{
    names
}(
    ntuple(i -> zero_lifted_inner(w, values(x)[i]), Val(length(x)))
)

"""
    uninit_lifted(::Val{N}, x)

Layer-3 seed factory for uninitialised slots. See [`zero_lifted`](@ref).
"""
@inline uninit_lifted(::Val{0}, x) = x
@inline uninit_lifted(w::Val{N}, x) where {N} = Lifted{typeof(x),N}(uninit_dual(w, x))

"""
    randn_lifted(::Val{N}, rng, x)

Layer-3 seed factory with random partials. See [`zero_lifted`](@ref).
"""
@inline randn_lifted(::Val{0}, ::AbstractRNG, x) = x
@inline randn_lifted(w::Val{N}, rng::AbstractRNG, x) where {N} = Lifted{typeof(x),N}(
    randn_dual(w, rng, x)
)
