module MooncakeBFloat16sExt
# BFloat16 arithmetic is supplied by BFloat16s.jl.

using Mooncake
using LinearAlgebra
using Random: AbstractRNG

#! format: off
using BFloat16s: BFloat16s

using Mooncake: @foldable, CRC

import Mooncake:
    MaybeCache,
    IncCache,
    SetToZeroCache,
    zero_tangent_internal,
    randn_tangent_internal,
    increment_internal!!,
    set_to_zero_internal!!,
    _scale_internal,
    _dot_internal,
    _add_to_primal_internal,
    tangent_to_primal_internal!!,
    primal_to_tangent_internal!!,
    zero_rdata,
    zero_rdata_from_type,
    can_produce_zero_rdata_from_type,
    nan_tangent_guard,
    NoFData,
    NoRData,
    CoDual,
    Dual,
    primal,
    tangent,
    extract,
    zero_fcodual,
    MinimalCtx

const P = BFloat16s.BFloat16

# These P specialisations track core's HalfFloat/IEEEOrBFloat methods for first-order AD.
# __verify_fdata_value uses the NoFData fallback; the ForwardMode nan_tangent_guard
# primitive for forward-over-reverse is not mirrored here.
@static if !isdefined(Core, :BFloat16) || P !== Core.BFloat16
@foldable Mooncake.tangent_type(::Type{P}) = P
Mooncake.fdata_type(::Type{P}) = NoFData
Mooncake.rdata_type(::Type{P}) = P
@foldable Mooncake.tangent_type(::Type{NoFData}, ::Type{P}) = P
tangent(::NoFData, r::P) = r
Mooncake._verify_rdata_value(::P, ::P) = nothing

Mooncake.arrayify(x::Union{Dual{<:AbstractArray{P}},CoDual{<:AbstractArray{P}}}) =
    Mooncake.arrayify(primal(x), tangent(x))
Mooncake.arrayify(x::A, dx::A) where {A<:Array{P}} = (x, dx)
function Mooncake.arrayify(x::SubArray{P,B,C,D,E}, dx::Mooncake.TangentOrFData) where {B,C,D,E}
    _, dt = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
    return x, SubArray{P,B,typeof(dt),D,E}(dt, x.indices, x.offset1, x.stride1)
end
function Mooncake.arrayify(x::Base.ReshapedArray{P,B,C,D}, dx::Mooncake.TangentOrFData) where {B,C,D}
    _, dt = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
    return x, Base.ReshapedArray{P,B,typeof(dt),D}(dt, x.dims, x.mi)
end
function Mooncake.arrayify(x::Base.ReinterpretArray{P,N,S,A,R}, dx::Mooncake.TangentOrFData) where {N,S,A,R}
    _, dt = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
    return x, R ? reinterpret(reshape, P, dt) : reinterpret(P, dt)
end
for (Wrapper, wrap) in ((LinearAlgebra.Adjoint, adjoint), (LinearAlgebra.Transpose, transpose))
    @eval function Mooncake.arrayify(x::$Wrapper{P,<:AbstractArray{P}}, dx::Mooncake.TangentOrFData)
        _, dt = Mooncake.arrayify(x.parent, Mooncake._fields(dx).parent)
        return x, $wrap(dt)
    end
end

Mooncake._is_ieee_or_bfloat(::Type{P}) = true

Mooncake.@is_primitive MinimalCtx Mooncake._CPUStridedMulSig{P}

@inline function Mooncake.frule!!(
    ::Dual{typeof(mul!)},
    c::Dual{<:Mooncake._CPUStridedOrAdjTransVecOrMat{P}},
    a::Dual{<:Mooncake._CPUStridedOrAdjTransMatrix{P}},
    b::Dual{<:Mooncake._CPUStridedOrAdjTransVecOrMat{P}},
    alpha::Dual{<:Union{Bool,Base.IEEEFloat,P}},
    beta::Dual{<:Union{Bool,Base.IEEEFloat,P}},
)
    return Mooncake._mul_frule!!(c, a, b, alpha, beta)
end

@inline function Mooncake.rrule!!(
    ::CoDual{typeof(mul!)},
    c::CoDual{<:Mooncake._CPUStridedOrAdjTransVecOrMat{P}},
    a::CoDual{<:Mooncake._CPUStridedOrAdjTransMatrix{P}},
    b::CoDual{<:Mooncake._CPUStridedOrAdjTransVecOrMat{P}},
    alpha::CoDual{S},
    beta::CoDual{U},
) where {S<:Union{Bool,Base.IEEEFloat,P},U<:Union{Bool,Base.IEEEFloat,P}}
    return Mooncake._mul_rrule!!(c, a, b, alpha, beta)
end

end

@static if VERSION >= v"1.11-rc4"
    Mooncake.arrayify(x::A, dx::A) where {A<:Memory{P}} = (x, dx)
end

# Constructing BFloat16 zeros requires the arithmetic supplied by BFloat16s.jl.
@noinline _scalar_float64(x::Union{P,Float32}) = Float64(x)

zero_tangent_internal(::P, ::MaybeCache) = zero(P)

randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) = P(randn(rng, Float32))

increment_internal!!(::IncCache, x::P, y::P) = x + y

set_to_zero_internal!!(::SetToZeroCache, ::P) = zero(P)

_scale_internal(::MaybeCache, a::Float64, t::P) = P(a * _scalar_float64(t))

# Julia 1.11 on x86_64: LLVM 16 may abort if tuple tangent maths vectorises this widening.
# Must return Float64: _dot_internal is always accumulated into a Float64 scalar.
@noinline _dot_internal(::MaybeCache, t::P, s::P) = Float64(t) * Float64(s)

_add_to_primal_internal(::MaybeCache, x::P, t::P, ::Bool) = x + t

tangent_to_primal_internal!!(::P, tx, ::MaybeCache) = tx

primal_to_tangent_internal!!(tx, x::P, ::MaybeCache) = x

zero_rdata(::P) = zero(P)

zero_rdata_from_type(::Type{P}) = zero(P)

@foldable can_produce_zero_rdata_from_type(::Type{P}) = true

@inline nan_tangent_guard(dy::P, t::P) = iszero(dy) ? zero(P) : t

# Conversions

Mooncake.to_cr_tangent(t::P) = t
Mooncake.to_cr_tangent(t::Array{P}) = t
@static if VERSION >= v"1.11-rc4"
    Mooncake.to_cr_tangent(t::Memory{P}) = t
end
Mooncake.mooncake_tangent(p, t::P) = t
Mooncake.mooncake_tangent(p::Array, t::Array{P}) = t
Mooncake.mooncake_tangent(::P, ::CRC.NotImplemented) = P(NaN)
Mooncake.increment_and_get_rdata!(::NoFData, r::P, t::P) = r + t
function Mooncake.increment_and_get_rdata!(f::Array{P}, ::NoRData, t::Array{P})
    Mooncake.increment!!(f, t)
    return NoRData()
end
Mooncake.increment_and_get_rdata!(f, r::P, ::CRC.NotImplemented) = P(NaN)

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float32},P}
function Mooncake.frule!!(::Dual{Type{Float32}}, x::Dual{P})
    return Dual(Float32(primal(x)), Float32(tangent(x)))
end
function Mooncake.rrule!!(::CoDual{Type{Float32}}, x::CoDual{P})
    pb(dy::Float32) = NoRData(), P(dy)
    return zero_fcodual(Float32(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float64},P}
function Mooncake.frule!!(::Dual{Type{Float64}}, x::Dual{P})
    return Dual(_scalar_float64(primal(x)), _scalar_float64(tangent(x)))
end
function Mooncake.rrule!!(::CoDual{Type{Float64}}, x::CoDual{P})
    pb(dy::Float64) = NoRData(), P(Float32(dy))
    return zero_fcodual(_scalar_float64(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float32}
function Mooncake.frule!!(::Dual{Type{P}}, x::Dual{Float32})
    return Dual(P(primal(x)), P(tangent(x)))
end
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float32})
    pb(dy::P) = NoRData(), Float32(dy)
    return zero_fcodual(P(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float64}
function Mooncake.frule!!(::Dual{Type{P}}, x::Dual{Float64})
    return Dual(P(Float32(primal(x))), P(Float32(tangent(x))))
end
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float64})
    pb(dy::P) = NoRData(), _scalar_float64(Float32(dy))
    return zero_fcodual(P(Float32(primal(x)))), pb
end

# Math rules

Mooncake.@is_primitive MinimalCtx Tuple{typeof(-),P}
function Mooncake.frule!!(::Dual{typeof(-)}, x::Dual{P})
    return Dual(-primal(x), -tangent(x))
end
function Mooncake.rrule!!(::CoDual{typeof(-)}, x::CoDual{P})
    pb(dy::P) = NoRData(), -dy
    return zero_fcodual(-primal(x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sqrt),P}
function Mooncake.frule!!(::Dual{typeof(sqrt)}, x::Dual{P})
    _x, dx = extract(x)
    y = sqrt(_x)
    return Dual(y, nan_tangent_guard(dx, dx / (2 * y)))
end
function Mooncake.rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P})
    y = sqrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (2 * y))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cbrt),P}
function Mooncake.frule!!(::Dual{typeof(cbrt)}, x::Dual{P})
    _x, dx = extract(x)
    y = cbrt(_x)
    return Dual(y, nan_tangent_guard(dx, dx / (3 * y^2)))
end
function Mooncake.rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P})
    y = cbrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (3 * y^2))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp),P}
function Mooncake.frule!!(::Dual{typeof(exp)}, x::Dual{P})
    y = exp(primal(x))
    return Dual(y, tangent(x) * y)
end
function Mooncake.rrule!!(::CoDual{typeof(exp)}, x::CoDual{P})
    y = exp(primal(x))
    pb(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp2),P}
function Mooncake.frule!!(::Dual{typeof(exp2)}, x::Dual{P})
    y = exp2(primal(x))
    return Dual(y, tangent(x) * y * P(log(2.0f0)))
end
function Mooncake.rrule!!(::CoDual{typeof(exp2)}, x::CoDual{P})
    y = exp2(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(2.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp10),P}
function Mooncake.frule!!(::Dual{typeof(exp10)}, x::Dual{P})
    y = exp10(primal(x))
    return Dual(y, tangent(x) * y * P(log(10.0f0)))
end
function Mooncake.rrule!!(::CoDual{typeof(exp10)}, x::CoDual{P})
    y = exp10(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(10.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(expm1),P}
function Mooncake.frule!!(::Dual{typeof(expm1)}, x::Dual{P})
    y = expm1(primal(x))
    return Dual(y, tangent(x) * (y + one(P)))
end
function Mooncake.rrule!!(::CoDual{typeof(expm1)}, x::CoDual{P})
    y = expm1(primal(x))
    pb(dy::P) = NoRData(), dy * (y + one(P))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log),P}
function Mooncake.frule!!(::Dual{typeof(log)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(log(_x), nan_tangent_guard(dx, dx / _x))
end
function Mooncake.rrule!!(::CoDual{typeof(log)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / _x)
    return zero_fcodual(log(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log2),P}
function Mooncake.frule!!(::Dual{typeof(log2)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(log2(_x), nan_tangent_guard(dx, dx / (_x * P(log(2.0f0)))))
end
function Mooncake.rrule!!(::CoDual{typeof(log2)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(2.0f0))))
    return zero_fcodual(log2(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log10),P}
function Mooncake.frule!!(::Dual{typeof(log10)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(log10(_x), nan_tangent_guard(dx, dx / (_x * P(log(10.0f0)))))
end
function Mooncake.rrule!!(::CoDual{typeof(log10)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(10.0f0))))
    return zero_fcodual(log10(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log1p),P}
function Mooncake.frule!!(::Dual{typeof(log1p)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(log1p(_x), nan_tangent_guard(dx, dx / (one(P) + _x)))
end
function Mooncake.rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) + _x))
    return zero_fcodual(log1p(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sin),P}
function Mooncake.frule!!(::Dual{typeof(sin)}, x::Dual{P})
    # Use separate sin/cos calls: sincos(::BFloat16) is broken (infinitely recursive) in Julia 1.12.
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    return Dual(s, tangent(x) * c)
end
function Mooncake.rrule!!(::CoDual{typeof(sin)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), dy * c
    return zero_fcodual(s), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cos),P}
function Mooncake.frule!!(::Dual{typeof(cos)}, x::Dual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    return Dual(c, -tangent(x) * s)
end
function Mooncake.rrule!!(::CoDual{typeof(cos)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), -dy * s
    return zero_fcodual(c), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tan),P}
function Mooncake.frule!!(::Dual{typeof(tan)}, x::Dual{P})
    y = tan(primal(x))
    return Dual(y, tangent(x) * (one(P) + y^2))
end
function Mooncake.rrule!!(::CoDual{typeof(tan)}, x::CoDual{P})
    y = tan(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) + y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asin),P}
function Mooncake.frule!!(::Dual{typeof(asin)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(asin(_x), nan_tangent_guard(dx, dx / sqrt(one(P) - _x^2)))
end
function Mooncake.rrule!!(::CoDual{typeof(asin)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(one(P) - _x^2))
    return zero_fcodual(asin(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acos),P}
function Mooncake.frule!!(::Dual{typeof(acos)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(acos(_x), nan_tangent_guard(dx, -dx / sqrt(one(P) - _x^2)))
end
function Mooncake.rrule!!(::CoDual{typeof(acos)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, -dy / sqrt(one(P) - _x^2))
    return zero_fcodual(acos(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atan),P}
function Mooncake.frule!!(::Dual{typeof(atan)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(atan(_x), dx / (one(P) + _x^2))
end
function Mooncake.rrule!!(::CoDual{typeof(atan)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / (one(P) + _x^2)
    return zero_fcodual(atan(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sinh),P}
function Mooncake.frule!!(::Dual{typeof(sinh)}, x::Dual{P})
    _x = primal(x)
    return Dual(sinh(_x), tangent(x) * cosh(_x))
end
function Mooncake.rrule!!(::CoDual{typeof(sinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * cosh(_x)
    return zero_fcodual(sinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cosh),P}
function Mooncake.frule!!(::Dual{typeof(cosh)}, x::Dual{P})
    _x = primal(x)
    return Dual(cosh(_x), tangent(x) * sinh(_x))
end
function Mooncake.rrule!!(::CoDual{typeof(cosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * sinh(_x)
    return zero_fcodual(cosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tanh),P}
function Mooncake.frule!!(::Dual{typeof(tanh)}, x::Dual{P})
    y = tanh(primal(x))
    return Dual(y, tangent(x) * (one(P) - y^2))
end
function Mooncake.rrule!!(::CoDual{typeof(tanh)}, x::CoDual{P})
    y = tanh(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) - y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asinh),P}
function Mooncake.frule!!(::Dual{typeof(asinh)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(asinh(_x), dx / sqrt(one(P) + _x^2))
end
function Mooncake.rrule!!(::CoDual{typeof(asinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / sqrt(one(P) + _x^2)
    return zero_fcodual(asinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acosh),P}
function Mooncake.frule!!(::Dual{typeof(acosh)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(acosh(_x), nan_tangent_guard(dx, dx / sqrt(_x^2 - one(P))))
end
function Mooncake.rrule!!(::CoDual{typeof(acosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(_x^2 - one(P)))
    return zero_fcodual(acosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atanh),P}
function Mooncake.frule!!(::Dual{typeof(atanh)}, x::Dual{P})
    _x, dx = extract(x)
    return Dual(atanh(_x), nan_tangent_guard(dx, dx / (one(P) - _x^2)))
end
function Mooncake.rrule!!(::CoDual{typeof(atanh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) - _x^2))
    return zero_fcodual(atanh(_x)), pb
end


Mooncake.@is_primitive MinimalCtx Tuple{typeof(hypot),P,P}
function Mooncake.frule!!(::Dual{typeof(hypot)}, x::Dual{P}, y::Dual{P})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    dh =
        nan_tangent_guard(tangent(x), _x * tangent(x)) +
        nan_tangent_guard(tangent(y), _y * tangent(y))
    return Dual(h, dh / h)
end
function Mooncake.rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    pb(dh::P) = NoRData(),
    nan_tangent_guard(dh, dh * _x / h),
    nan_tangent_guard(dh, dh * _y / h)
    return zero_fcodual(h), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(^),P,P}
function Mooncake.frule!!(::Dual{typeof(^)}, x::Dual{P}, y::Dual{P})
    _x, _y = primal(x), primal(y)
    z = _x^_y
    # Note: _x^(_y-1) is computed separately from z to correctly handle _x=0:
    # z/_x = 0/0 = NaN, whereas _x^(_y-1) gives the correct value at the boundary
    # (e.g. Inf when 0 < _y < 1, since d/dx(x^y)|_{x=0} diverges).
    return Dual(z,
        nan_tangent_guard(tangent(x), _y * _x^(_y - one(P)) * tangent(x)) +
        # Guard on z (not tangent(y)): when _x=0, z=0 and log(_x)=-Inf, so
        # z*log(_x)*tangent(y) = 0*(-Inf)*tangent(y) = NaN without this guard.
        nan_tangent_guard(z, z * log(_x) * tangent(y)))
end
function Mooncake.rrule!!(::CoDual{typeof(^)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    z = _x^_y
    function pow_pb(dz::P)
        return NoRData(),
            nan_tangent_guard(dz, dz * _y * _x^(_y - one(P))),
            # Inner guard on z: prevents 0*(-Inf)=NaN when _x=0 (z=0, log(_x)=-Inf).
            # Outer guard on dz: standard upstream-zero mask (dz=0 → zero gradient).
            nan_tangent_guard(dz, nan_tangent_guard(z, dz * z * log(_x)))
    end
    return zero_fcodual(z), pow_pb
end


Mooncake.@is_primitive MinimalCtx Tuple{typeof(max),P,P}
function Mooncake.frule!!(::Dual{typeof(max)}, x::Dual{P}, y::Dual{P})
    _x, _y = primal(x), primal(y)
    return Dual(max(_x, _y), _x >= _y ? tangent(x) : tangent(y))
end
function Mooncake.rrule!!(::CoDual{typeof(max)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) >= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(max(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(min),P,P}
function Mooncake.frule!!(::Dual{typeof(min)}, x::Dual{P}, y::Dual{P})
    _x, _y = primal(x), primal(y)
    return Dual(min(_x, _y), _x <= _y ? tangent(x) : tangent(y))
end
function Mooncake.rrule!!(::CoDual{typeof(min)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) <= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(min(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(abs),P}
function Mooncake.frule!!(::Dual{typeof(abs)}, x::Dual{P})
    _x = primal(x)
    return Dual(abs(_x), _x >= zero(P) ? tangent(x) : -tangent(x))
end
function Mooncake.rrule!!(::CoDual{typeof(abs)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), _x >= zero(P) ? dy : -dy
    return zero_fcodual(abs(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(Base.eps),P}
function Mooncake.frule!!(::Dual{typeof(Base.eps)}, x::Dual{P})
    return Dual(eps(primal(x)), zero(P))
end
function Mooncake.rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P})
    pb(::P) = NoRData(), zero(P)
    return zero_fcodual(eps(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(nextfloat),P}
function Mooncake.frule!!(::Dual{typeof(nextfloat)}, x::Dual{P})
    return Dual(nextfloat(primal(x)), tangent(x))
end
function Mooncake.rrule!!(::CoDual{typeof(nextfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(nextfloat(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(prevfloat),P}
function Mooncake.frule!!(::Dual{typeof(prevfloat)}, x::Dual{P})
    return Dual(prevfloat(primal(x)), tangent(x))
end
function Mooncake.rrule!!(::CoDual{typeof(prevfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(prevfloat(primal(x))), pb
end

#! format: on

end # module MooncakeBFloat16sExt
