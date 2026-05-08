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
    # At width=1 keep the bare tangent type (no NTangent wrap). Chunked
    # `NTangent{NTuple{N, T}}` is only meaningful at N>=2; using it at N=1
    # would break the many existing rules whose signature constrains
    # `Dual{P, T<:StandardTangentType}` (e.g. the structured-getfield rule),
    # since `NTangent` is not `<:StandardTangentType`.
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

    return isconcretetype(P) ? Dual{P,tangent_type(Val(N), P)} : Dual
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

# `primal` / `tangent` on a bare element-wise tuple of inner duals (the inner
# `V` of a `Lifted{<:Tuple, N}`). Recursive map so nested Tuple-of-Dual works.
primal(t::Tuple) = map(primal, t)
tangent(t::Tuple) = map(tangent, t)
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

# Generic width-N fallback for non-differentiable primals (`tangent_type ===
# NoTangent`); matches `dual_type(Val(N), P) == Dual{P,NoTangent}`. Specialised
# IEEEFloat / Complex / array / Memory overloads live in `nfwd/NfwdMooncake.jl`.
# Fails loudly if a non-trivial `tangent_type` reaches here without a width-N
# overload — silently downgrading to width-1 would be wrong.
@inline function zero_dual(w::Val, x)
    if tangent_type(_typeof(x)) === NoTangent
        return Dual(x, NoTangent())
    end
    throw(
        ArgumentError(
            "zero_dual(::Val, ::$(_typeof(x))): missing width-N overload for a " *
            "type with non-trivial tangent_type. Add a method to NfwdMooncake.jl " *
            "matching `dual_type(Val(N), $(_typeof(x)))`.",
        ),
    )
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
        return T === tangent_type(Val(N), P)
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

# 1-arg: wrap an already-built inner. V is inferred from typeof(value).
@inline Lifted{P,N}(value) where {P,N} = Lifted{P,N,typeof(value)}(value)

# 2-arg: build the inner via the inner type's own constructor methods. Mirrors
# `CoDual(x, dx)` — pass `(primal, tangent)` and the wrapper takes care of the
# rest. The dispatch on inner shape lives in the inner type's constructors.
@inline function Lifted{P,N}(primal, tangent) where {P,N}
    InnerT = dual_type(Val(N), P)
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
@inline function Lifted{P,N}(primal::P, tangent::Tup) where {P<:Tuple,N,Tup<:Tuple}
    InnerT = dual_type(Val(N), P)
    inner = ntuple(
        i -> _inner_dual_for_field(fieldtype(InnerT, i), primal[i], tangent[i]),
        Val(fieldcount(P)),
    )
    return Lifted{P,N,InnerT}(inner)
end

# Build a single field's inner dual value. For non-Tuple fields, defer to the
# field type's own 2-arg constructor (NDual, Vector{NDual}, Dual, etc.). For
# Tuple-typed fields, recurse element-wise so a nested Tuple-of-Dual builds
# without trying `Tuple{...}(::Tuple, ::Tuple)` (which has no ctor).
@inline _inner_dual_for_field(::Type{V}, primal, tangent) where {V} = V(primal, tangent)
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

# Tuple-primal with `NoTangent` (whole-tuple) tangent — common when a `Tuple`
# slot holds non-differentiable elements (e.g. `Tuple{Int}`). Build the inner
# tuple element-wise with `NoTangent` for each field. Use
# `_inner_dual_for_field` so nested Tuple fields (e.g. `Tuple{Tuple{}, Int}`)
# recurse properly instead of trying to call `Tuple{}(::Tuple, ::NoTangent)`
# which has no method.
@inline function Lifted{P,N}(primal::P, ::NoTangent) where {P<:Tuple,N}
    InnerT = dual_type(Val(N), P)
    inner = ntuple(
        i -> _inner_dual_for_field(fieldtype(InnerT, i), primal[i], NoTangent()),
        Val(fieldcount(P)),
    )
    return Lifted{P,N,InnerT}(inner)
end

# Tuple-primal with `NTangent` tangent — width-N chunked vararg case where
# `_group_vararg_dual` produces an `NTangent` lane structure. Extract per
# element, per direction.
@inline function Lifted{P,N}(primal::P, tangent::NTangent) where {P<:Tuple,N}
    InnerT = dual_type(Val(N), P)
    lanes = tangent.lanes
    inner = ntuple(Val(fieldcount(P))) do i
        Vi = fieldtype(InnerT, i)
        if N == 1
            Vi(primal[i], lanes[1][i])
        else
            partials = ntuple(d -> lanes[d][i], Val(N))
            Vi(primal[i], partials)
        end
    end
    return Lifted{P,N,InnerT}(inner)
end

# NamedTuple-primal: parallel to the Tuple ctor. Inner V is a
# `NamedTuple{names, Tuple{V_i...}}` of bare inner duals; build element-wise.
@inline function Lifted{P,N}(
    primal::P, tangent::NamedTuple{names}
) where {P<:NamedTuple{names},N} where {names}
    InnerT = dual_type(Val(N), P)
    InnerTup = fieldtype(InnerT, 1) === Nothing ? Tuple{} : InnerT.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(fieldtype(InnerTup, i), primal[i], values(tangent)[i]),
        Val(fieldcount(P)),
    )
    return Lifted{P,N,InnerT}(NamedTuple{names}(inner_tup))
end
@inline function Lifted{P,N}(
    primal::P, ::NoTangent
) where {P<:NamedTuple{names},N} where {names}
    InnerT = dual_type(Val(N), P)
    InnerTup = InnerT.parameters[2]
    inner_tup = ntuple(
        i -> fieldtype(InnerTup, i)(primal[i], NoTangent()), Val(fieldcount(P))
    )
    return Lifted{P,N,InnerT}(NamedTuple{names}(inner_tup))
end

# Accessors: delegate to the inner's own primal/tangent. For Tuple/NamedTuple
# primals the inner is a bare element-wise (named) tuple, so map over it.
primal(d::Lifted) = primal(d.value)
tangent(d::Lifted) = tangent(d.value)
primal(d::Lifted{<:Tuple}) = map(primal, d.value)
tangent(d::Lifted{<:Tuple}) = map(tangent, d.value)
primal(d::Lifted{<:NamedTuple}) = map(primal, d.value)
tangent(d::Lifted{<:NamedTuple}) = map(tangent, d.value)

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
    # Abstract `P` (e.g. `Any`, `Real`): widen to the bare `Lifted` UnionAll —
    # mirrors reverse mode's `fcodual_type(Any) = CoDual` UnionAll. Both `P`
    # and `V` are universally quantified so concrete runtime values
    # `Lifted{Q, N, V'}` for any `Q` and `V'` fit the slot without parametric-
    # invariance coercion. We can't use `Lifted{P, N, V} where {P', V'}` for a
    # fixed `N` because `N` would still need to be free for cross-width
    # specialisation; the bare UnionAll is the simplest correct shape.
    return Lifted
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
@inline zero_lifted(::Val{0}, x) = x
@inline zero_lifted(w::Val{N}, x) where {N} = Lifted{typeof(x),N}(zero_dual(w, x))

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
