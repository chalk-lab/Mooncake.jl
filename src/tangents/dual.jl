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
        N = length(T.parameters[1].parameters)
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
