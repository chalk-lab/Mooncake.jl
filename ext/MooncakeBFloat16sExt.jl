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
zero_tangent_internal(::P, ::MaybeCache) = zero(P)

randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) = P(randn(rng, Float32))

increment_internal!!(::IncCache, x::P, y::P) = x + y

set_to_zero_internal!!(::SetToZeroCache, ::P) = zero(P)

_scale_internal(::MaybeCache, a::Float64, t::P) = P(a * Float64(t))

# Julia 1.11 on x86_64: LLVM 16 may abort if tuple tangent maths vectorises this widening.
# Must return Float64: _dot_internal is always accumulated into a Float64 scalar.
_dot_internal(::MaybeCache, t::P, s::P) = Float64(t) * Float64(s)

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
    return Dual(Float64(primal(x)), Float64(tangent(x)))
end
function Mooncake.rrule!!(::CoDual{Type{Float64}}, x::CoDual{P})
    pb(dy::Float64) = NoRData(), P(Float32(dy))
    return zero_fcodual(Float64(primal(x))), pb
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
    pb(dy::P) = NoRData(), Float64(Float32(dy))
    return zero_fcodual(P(Float32(primal(x)))), pb
end

# Math rules

# Each derivative is written in terms of the primal `_x`, its result `y` and the incoming
# tangent (forward) or cotangent (reverse) `d`.
for (op, derivative) in (
    (:-, :(-d)),
    (:sqrt, :(nan_tangent_guard(d, d / (2 * y)))),
    (:cbrt, :(nan_tangent_guard(d, d / (3 * y^2)))),
    (:exp, :(d * y)),
    (:exp2, :(d * y * P(log(2.0f0)))),
    (:exp10, :(d * y * P(log(10.0f0)))),
    (:expm1, :(d * (y + one(P)))),
    (:log, :(nan_tangent_guard(d, d / _x))),
    (:log2, :(nan_tangent_guard(d, d / (_x * P(log(2.0f0)))))),
    (:log10, :(nan_tangent_guard(d, d / (_x * P(log(10.0f0)))))),
    (:log1p, :(nan_tangent_guard(d, d / (one(P) + _x)))),
    # Not sincos: sincos(::BFloat16) recurses infinitely on Julia 1.12.
    (:sin, :(d * cos(_x))),
    (:cos, :(-d * sin(_x))),
    (:tan, :(d * (one(P) + y^2))),
    (:asin, :(nan_tangent_guard(d, d / sqrt(one(P) - _x^2)))),
    (:acos, :(nan_tangent_guard(d, -d / sqrt(one(P) - _x^2)))),
    (:atan, :(d / (one(P) + _x^2))),
    (:sinh, :(d * cosh(_x))),
    (:cosh, :(d * sinh(_x))),
    (:tanh, :(d * (one(P) - y^2))),
    (:asinh, :(d / sqrt(one(P) + _x^2))),
    (:acosh, :(nan_tangent_guard(d, d / sqrt(_x^2 - one(P))))),
    (:atanh, :(nan_tangent_guard(d, d / (one(P) - _x^2)))),
    (:abs, :(_x >= zero(P) ? d : -d)),
    (:eps, :(zero(P))),
    (:nextfloat, :d),
    (:prevfloat, :d),
)
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{typeof($op),P}
        function Mooncake.frule!!(::Dual{typeof($op)}, x::Dual{P})
            _x, d = extract(x)
            y = $op(_x)
            return Dual(y, $derivative)
        end
        function Mooncake.rrule!!(::CoDual{typeof($op)}, x::CoDual{P})
            _x = primal(x)
            y = $op(_x)
            pb(d::P) = NoRData(), $derivative
            return zero_fcodual(y), pb
        end
    end
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


for (op, x_wins) in ((:max, :>=), (:min, :<=))
    @eval begin
        Mooncake.@is_primitive MinimalCtx Tuple{typeof($op),P,P}
        function Mooncake.frule!!(::Dual{typeof($op)}, x::Dual{P}, y::Dual{P})
            _x, _y = primal(x), primal(y)
            return Dual($op(_x, _y), $x_wins(_x, _y) ? tangent(x) : tangent(y))
        end
        function Mooncake.rrule!!(::CoDual{typeof($op)}, x::CoDual{P}, y::CoDual{P})
            _x_wins = $x_wins(primal(x), primal(y))
            pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
            return zero_fcodual($op(primal(x), primal(y))), pb
        end
    end
end

#! format: on

end # module MooncakeBFloat16sExt
