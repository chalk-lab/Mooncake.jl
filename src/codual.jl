struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type, in order to preserve dispatch possibility.
function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x), NoTangent}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    uninit_codual(x)

See implementation for details, as this function is subject to change.
"""
@inline uninit_codual(x::P) where {P} = CoDual(x, uninit_tangent(x))

"""
    codual_type(P::Type)

Shorthand for `CoDual{P, tangent_type(P}}` when `P` is concrete, equal to `CoDual` if not.
"""
function codual_type(::Type{P}) where {P}
    P == DataType && return CoDual
    P isa Union && return Union{codual_type(P.a), codual_type(P.b)}
    return isconcretetype(P) ? CoDual{P, tangent_type(P)} : CoDual
end

codual_type(::Type{Type{P}}) where {P} = CoDual{Type{P}, NoTangent}

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

might_be_active(args) = any(might_be_active ∘ _typeof, args)
