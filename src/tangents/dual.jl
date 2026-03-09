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

function dual_type(::Type{P}) where {P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
    # P is abstract (a UnionAll generator or the UnionAll type itself), so we dont know its tangent type.
    (P isa UnionAll || P == UnionAll) && return Dual

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
