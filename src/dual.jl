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
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
    P <: UnionAll && return Dual # P is abstract, so we don't know its tangent type.
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

@inline uninit_dual(x::P) where {P} = Dual(x, uninit_tangent(x))

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

# Dual of numeric types is self-tangent
@inline tangent_type(::Type{Dual{P,T}}) where {P<:IEEEFloat,T<:IEEEFloat} = Dual{P,T}

@inline zero_tangent_internal(
    x::Dual{P,T}, ::MaybeCache
) where {P<:IEEEFloat,T<:IEEEFloat} = Dual(zero(P), zero(T))

@inline function randn_tangent_internal(
    rng::AbstractRNG, x::Dual{P,T}, ::MaybeCache
) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(randn(rng, P), randn(rng, T))
end

@inline function increment!!(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) + primal(y), tangent(x) + tangent(y))
end

@inline set_to_zero_internal!!(
    ::SetToZeroCache, x::Dual{P,T}
) where {P<:IEEEFloat,T<:IEEEFloat} = Dual(zero(P), zero(T))

@inline function increment_internal!!(
    ::IncCache, x::Dual{P,T}, y::Dual{P,T}
) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) + primal(y), tangent(x) + tangent(y))
end

Base.one(::Type{Dual{P,T}}) where {P<:IEEEFloat,T<:IEEEFloat} = Dual(one(P), zero(T))
function Base.one(x::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(one(primal(x)), zero(tangent(x)))
end

# Arithmetic operations
function Base.:+(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) + y, tangent(x))
end
function Base.:+(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(x + primal(y), tangent(y))
end
function Base.:+(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) + primal(y), tangent(x) + tangent(y))
end

# Subtraction
Base.:-(x::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = Dual(-primal(x), -tangent(x))
function Base.:-(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) - y, tangent(x))
end
function Base.:-(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(x - primal(y), -tangent(y))
end
function Base.:-(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) - primal(y), tangent(x) - tangent(y))
end

# Multiplication (product rule)
function Base.:*(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) * y, tangent(x) * y)
end
function Base.:*(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(x * primal(y), x * tangent(y))
end
function Base.:*(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) * primal(y), primal(x) * tangent(y) + tangent(x) * primal(y))
end
function Base.:*(x::Dual{P,T}, y::Integer) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) * y, tangent(x) * y)
end
function Base.:*(x::Integer, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(x * primal(y), x * tangent(y))
end

# Division (quotient rule)
function Base.:/(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x) / y, tangent(x) / y)
end
function Base.:/(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(x / primal(y), -x * tangent(y) / primal(y)^2)
end
function Base.:/(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(
        primal(x) / primal(y),
        (tangent(x) * primal(y) - primal(x) * tangent(y)) / primal(y)^2,
    )
end

# Power (chain rule)
function Base.:^(x::Dual{P,T}, n::Integer) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x)^n, n * primal(x)^(n - 1) * tangent(x))
end
function Base.:^(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual(primal(x)^y, y * primal(x)^(y - 1) * tangent(x))
end

# Comparison (use primal for comparisons)
Base.:<(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat} = primal(x) < y
Base.:<(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x < primal(y)
function Base.:<(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return primal(x) < primal(y)
end
Base.:>(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat} = primal(x) > y
Base.:>(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x > primal(y)
function Base.:>(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return primal(x) > primal(y)
end
Base.:<=(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat} = primal(x) <= y
Base.:<=(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x <= primal(y)
function Base.:<=(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return primal(x) <= primal(y)
end
Base.:>=(x::Dual{P,T}, y::P) where {P<:IEEEFloat,T<:IEEEFloat} = primal(x) >= y
Base.:>=(x::P, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x >= primal(y)
function Base.:>=(x::Dual{P,T}, y::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat}
    return primal(x) >= primal(y)
end

# Conversion and promotion
Base.convert(::Type{Dual{P,T}}, x::P) where {P<:IEEEFloat,T<:IEEEFloat} = Dual(x, zero(T))
function Base.promote_rule(::Type{Dual{P,T}}, ::Type{P}) where {P<:IEEEFloat,T<:IEEEFloat}
    return Dual{P,T}
end

LinearAlgebra.transpose(x::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x
LinearAlgebra.adjoint(x::Dual{P,T}) where {P<:IEEEFloat,T<:IEEEFloat} = x
