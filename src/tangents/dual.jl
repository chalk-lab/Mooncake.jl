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
"""
verify_dual_type(x::Dual) = tangent_type(typeof(primal(x))) == typeof(tangent(x))

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

# ── Structural-lift wrappers ─────────────────────────────────────────────────
#
# `ImmutableDual{T<:NamedTuple}` and `MutableDual{T<:NamedTuple}` are
# single-field wrappers used as the canonical V for concrete struct primals
# under the forward-mode structural lift. They mirror reverse mode's
# `Tangent{Tfields<:NamedTuple}` / `MutableTangent{Tfields<:NamedTuple}` split:
# `ImmutableDual` is the immutable variant (for immutable struct primals),
# `MutableDual` is the mutable variant (for mutable struct primals; the
# mutability is load-bearing for the `MutableDualTangentView` proxy that
# writes back to the parent via `setfield!`).
#
# Both wrappers hold the recursive NamedTuple of canonical field Vs in their
# `value` field. The slot-level primal back-reference lives in the outer
# `Lifted{P, N, V}` wrapper (defined elsewhere), not inside these V wrappers.
#
# See AGENTS.md (Working Conventions) and the design notes referenced under
# the Documentation section there for the full specification.

"""
    ImmutableDual{T<:NamedTuple}

Single-field immutable wrapper used as the canonical V for *immutable struct*
primals under the forward-mode structural lift. Its `value::T` field holds
the recursive `NamedTuple{fieldnames(P), Tuple{V_i...}}` of canonical field
Vs, where each `V_i = dual_type(Val(N), fieldtype(P, i))`.

Mirrors reverse-mode `Tangent{Tfields<:NamedTuple}`: same single-field shape,
immutable variant. The slot-level primal back-reference lives in the outer
`Lifted{P, N, V}` wrapper.
"""
struct ImmutableDual{T<:NamedTuple}
    value::T
end

Base.:(==)(x::ImmutableDual, y::ImmutableDual) = x.value == y.value

"""
    MutableDual{T<:NamedTuple}

Single-field mutable wrapper used as the canonical V for *mutable struct*
primals under the forward-mode structural lift. Counterpart to
`ImmutableDual`; its mutability is load-bearing for the
`MutableDualTangentView` proxy that writes back to `value` via `setfield!`.

Mirrors reverse-mode `MutableTangent{Tfields<:NamedTuple}`: same single-field
shape, mutable variant. The slot-level primal back-reference lives in the
outer `Lifted{P, N, V}` wrapper.
"""
mutable struct MutableDual{T<:NamedTuple}
    value::T
end

Base.:(==)(x::MutableDual, y::MutableDual) = x.value == y.value
