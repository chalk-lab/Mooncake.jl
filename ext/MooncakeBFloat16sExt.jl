module MooncakeBFloat16sExt

using Mooncake
using Random: AbstractRNG

#! format: off
using BFloat16s: BFloat16s

using Mooncake: @foldable

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
    Lifted,
    NoTangent,
    NTangent,
    primal,
    tangent,
    extract,
    zero_fcodual,
    MinimalCtx,
    _unlift,
    _ndual_output_to_width1

# Core.BFloat16 requires Julia >= 1.11.
# BFloat16s.BFloat16 === Core.BFloat16 is not guaranteed on all platforms.
@static if VERSION >= v"1.11-" && BFloat16s.BFloat16 === Core.BFloat16

# On x86_64 with LLVM >= 15, BFloat16s.BFloat16 === Core.BFloat16.
# On other platforms, BFloat16s.BFloat16 is a distinct type.
# All methods below are defined on Core.BFloat16 directly (always available on Julia >= 1.11).

const P = Core.BFloat16

# zero(P) calls P(0), which requires BFloat16s.jl to define convert(Core.BFloat16, ::Int).
# These therefore live here rather than in src/rules/bfloat16.jl.
zero_tangent_internal(::P, ::MaybeCache) = zero(P)

randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) = P(randn(rng, Float32))

increment_internal!!(::IncCache, x::P, y::P) = x + y

set_to_zero_internal!!(::SetToZeroCache, ::P) = zero(P)

_scale_internal(::MaybeCache, a::Float64, t::P) = P(a * Float64(t))

# Must return Float64: _dot_internal is always accumulated into a Float64 scalar.
_dot_internal(::MaybeCache, t::P, s::P) = Float64(t) * Float64(s)

_add_to_primal_internal(::MaybeCache, x::P, t::P, ::Bool) = x + t

tangent_to_primal_internal!!(::P, tx, ::MaybeCache) = tx

primal_to_tangent_internal!!(tx, x::P, ::MaybeCache) = x

zero_rdata(::P) = zero(P)

zero_rdata_from_type(::Type{P}) = zero(P)

@foldable can_produce_zero_rdata_from_type(::Type{P}) = true

@inline nan_tangent_guard(dy::P, t::P) = iszero(dy) ? zero(P) : t

# ── Centralised bare-Dual → Lifted dispatch adapters for BFloat16 ─────────────
#
# Parallel to the IEEEFloat adapters in `src/rules/rules_via_nfwd.jl`: route
# bare-Dual `frule!!` calls with `Dual{Core.BFloat16}` args through the Lifted
# path for `_is_lifted_aware` primitives. Per-op Lifted-typed bodies (below)
# are then the single source of truth. The IEEEFloat adapters do not match
# `P=Core.BFloat16` (BFloat16 is not `<: IEEEFloat`), so these are
# extension-local.

@inline function Mooncake.frule!!(f::Dual{F}, x::Dual{P}) where {F,P<:Core.BFloat16}
    Mooncake._is_lifted_aware(Tuple{F,P}) || throw(MethodError(Mooncake.frule!!, (f, x)))
    return _ndual_output_to_width1(
        Mooncake.frule!!(
            Lifted{F,1}(primal(f), tangent(f)),
            Lifted{P,1}(primal(x), tangent(x)),
        ),
    )
end

# Binary: both args have the same BFloat16 primal type. Used by `hypot`, `^`,
# `max`, `min`.
@inline function Mooncake.frule!!(
    f::Dual{F}, a::Dual{P}, b::Dual{P}
) where {F,P<:Core.BFloat16}
    Mooncake._is_lifted_aware(Tuple{F,P,P}) ||
        throw(MethodError(Mooncake.frule!!, (f, a, b)))
    return _ndual_output_to_width1(
        Mooncake.frule!!(
            Lifted{F,1}(primal(f), tangent(f)),
            Lifted{P,1}(primal(a), tangent(a)),
            Lifted{P,1}(primal(b), tangent(b)),
        ),
    )
end

# Conversion rules are covered by existing adapters:
#   - `Type{Float32}` / `Type{Float64}` from `BFloat16` → matched by the
#     BFloat16-unary adapter above (F = `Type{Pout}`, P = `BFloat16`).
#   - `Type{BFloat16}` from `Float32` / `Float64` → matched by the IEEEFloat
#     unary adapter in `src/rules/rules_via_nfwd.jl` (F = `Type{BFloat16}`,
#     P <: IEEEFloat).
# Both delegate through to the Lifted-typed conversion bodies registered
# below via `_is_lifted_aware`.

# Conversions

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float32},P}
@inline function Mooncake.frule!!(
    ::Lifted{Type{Float32},N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = Float32(primal(inner))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> Float32(lanes[n]), Val(N))
    return Lifted{Float32,N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{Type{Float32},P}}) = true
function Mooncake.rrule!!(::CoDual{Type{Float32}}, x::CoDual{P})
    pb(dy::Float32) = NoRData(), P(dy)
    return zero_fcodual(Float32(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float64},P}
@inline function Mooncake.frule!!(
    ::Lifted{Type{Float64},N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = Float64(primal(inner))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> Float64(lanes[n]), Val(N))
    return Lifted{Float64,N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{Type{Float64},P}}) = true
function Mooncake.rrule!!(::CoDual{Type{Float64}}, x::CoDual{P})
    pb(dy::Float64) = NoRData(), P(Float32(dy))
    return zero_fcodual(Float64(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float32}
@inline function Mooncake.frule!!(
    ::Lifted{Type{P},N}, x::Lifted{Float32,N}
) where {N}
    inner = _unlift(x)
    y = P(primal(inner))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> P(lanes[n]), Val(N))
    return Lifted{P,N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{Type{P},Float32}}) = true
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float32})
    pb(dy::P) = NoRData(), Float32(dy)
    return zero_fcodual(P(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float64}
@inline function Mooncake.frule!!(
    ::Lifted{Type{P},N}, x::Lifted{Float64,N}
) where {N}
    inner = _unlift(x)
    y = P(Float32(primal(inner)))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> P(Float32(lanes[n])), Val(N))
    return Lifted{P,N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{Type{P},Float64}}) = true
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float64})
    pb(dy::P) = NoRData(), Float64(Float32(dy))
    return zero_fcodual(P(Float32(primal(x)))), pb
end

# Math rules

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sqrt),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(sqrt),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = sqrt(_x)
    inv_2y = inv(2 * y)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * inv_2y), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sqrt),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P})
    y = sqrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (2 * y))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cbrt),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(cbrt),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = cbrt(_x)
    inv_3y2 = inv(3 * y^2)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * inv_3y2), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(cbrt),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P})
    y = cbrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (3 * y^2))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp),P}
@inline function Mooncake.frule!!(::Lifted{typeof(exp),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    y = exp(primal(inner))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * y, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(exp),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(exp)}, x::CoDual{P})
    y = exp(primal(x))
    pb(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp2),P}
@inline function Mooncake.frule!!(::Lifted{typeof(exp2),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    y = exp2(primal(inner))
    d = y * P(log(2.0f0))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(exp2),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(exp2)}, x::CoDual{P})
    y = exp2(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(2.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp10),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(exp10),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = exp10(primal(inner))
    d = y * P(log(10.0f0))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(exp10),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(exp10)}, x::CoDual{P})
    y = exp10(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(10.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(expm1),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(expm1),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = expm1(primal(inner))
    d = y + one(P)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(expm1),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(expm1)}, x::CoDual{P})
    y = expm1(primal(x))
    pb(dy::P) = NoRData(), dy * (y + one(P))
    return zero_fcodual(y), pb
end

# Helper for log family: derivative `1/(_x * c)` where c is a precomputed
# constant. Common to log, log2, log10. Width-N Lifted body applies per-lane
# with `nan_tangent_guard`.
@inline function _bf16_log_family(f, c::P, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = f(_x)
    d = inv(_x * c)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log),P}
@inline function Mooncake.frule!!(::Lifted{typeof(log),N}, x::Lifted{P,N}) where {N}
    return _bf16_log_family(log, one(P), x)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(log),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(log)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / _x)
    return zero_fcodual(log(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log2),P}
@inline function Mooncake.frule!!(::Lifted{typeof(log2),N}, x::Lifted{P,N}) where {N}
    return _bf16_log_family(log2, P(log(2.0f0)), x)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(log2),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(log2)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(2.0f0))))
    return zero_fcodual(log2(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log10),P}
@inline function Mooncake.frule!!(::Lifted{typeof(log10),N}, x::Lifted{P,N}) where {N}
    return _bf16_log_family(log10, P(log(10.0f0)), x)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(log10),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(log10)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(10.0f0))))
    return zero_fcodual(log10(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log1p),P}
@inline function Mooncake.frule!!(::Lifted{typeof(log1p),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = log1p(_x)
    d = inv(one(P) + _x)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(log1p),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) + _x))
    return zero_fcodual(log1p(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sin),P}
@inline function Mooncake.frule!!(::Lifted{typeof(sin),N}, x::Lifted{P,N}) where {N}
    # Use separate sin/cos calls: sincos(::BFloat16) is broken (infinitely recursive) in Julia 1.12.
    inner = _unlift(x)
    _x = primal(inner)
    s = sin(_x)
    c = cos(_x)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * c, Val(N))
    return Lifted{typeof(s),N}(s, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sin),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(sin)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), dy * c
    return zero_fcodual(s), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cos),P}
@inline function Mooncake.frule!!(::Lifted{typeof(cos),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    s = sin(_x)
    c = cos(_x)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> -lanes[n] * s, Val(N))
    return Lifted{typeof(c),N}(c, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(cos),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(cos)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), -dy * s
    return zero_fcodual(c), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tan),P}
@inline function Mooncake.frule!!(::Lifted{typeof(tan),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    y = tan(primal(inner))
    d = one(P) + y^2
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(tan),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(tan)}, x::CoDual{P})
    y = tan(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) + y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asin),P}
@inline function Mooncake.frule!!(::Lifted{typeof(asin),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = asin(_x)
    d = inv(sqrt(one(P) - _x^2))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(asin),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(asin)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(one(P) - _x^2))
    return zero_fcodual(asin(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acos),P}
@inline function Mooncake.frule!!(::Lifted{typeof(acos),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = acos(_x)
    d = -inv(sqrt(one(P) - _x^2))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(acos),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(acos)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, -dy / sqrt(one(P) - _x^2))
    return zero_fcodual(acos(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atan),P}
@inline function Mooncake.frule!!(::Lifted{typeof(atan),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = atan(_x)
    d = inv(one(P) + _x^2)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(atan),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(atan)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / (one(P) + _x^2)
    return zero_fcodual(atan(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sinh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(sinh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = sinh(_x)
    d = cosh(_x)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sinh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(sinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * cosh(_x)
    return zero_fcodual(sinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cosh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(cosh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = cosh(_x)
    d = sinh(_x)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(cosh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(cosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * sinh(_x)
    return zero_fcodual(cosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tanh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(tanh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    y = tanh(primal(inner))
    d = one(P) - y^2
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(tanh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(tanh)}, x::CoDual{P})
    y = tanh(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) - y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asinh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(asinh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = asinh(_x)
    d = inv(sqrt(one(P) + _x^2))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * d, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(asinh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(asinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / sqrt(one(P) + _x^2)
    return zero_fcodual(asinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acosh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(acosh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = acosh(_x)
    d = inv(sqrt(_x^2 - one(P)))
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(acosh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(acosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(_x^2 - one(P)))
    return zero_fcodual(acosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atanh),P}
@inline function Mooncake.frule!!(::Lifted{typeof(atanh),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = atanh(_x)
    d = inv(one(P) - _x^2)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> nan_tangent_guard(lanes[n], lanes[n] * d), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(atanh),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(atanh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) - _x^2))
    return zero_fcodual(atanh(_x)), pb
end


Mooncake.@is_primitive MinimalCtx Tuple{typeof(hypot),P,P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(hypot),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    ix, iy = _unlift(x), _unlift(y)
    _x, _y = primal(ix), primal(iy)
    h = hypot(_x, _y)
    inv_h = inv(h)
    xl = tangent(ix).lanes
    yl = tangent(iy).lanes
    new_lanes = ntuple(Val(N)) do n
        (
            nan_tangent_guard(xl[n], _x * xl[n]) + nan_tangent_guard(yl[n], _y * yl[n])
        ) * inv_h
    end
    return Lifted{typeof(h),N}(h, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(hypot),P,P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    pb(dh::P) = NoRData(),
    nan_tangent_guard(dh, dh * _x / h),
    nan_tangent_guard(dh, dh * _y / h)
    return zero_fcodual(h), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(^),P,P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(^),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    ix, iy = _unlift(x), _unlift(y)
    _x, _y = primal(ix), primal(iy)
    z = _x^_y
    # `_x^(_y-1)` is computed separately from `z` to handle `_x=0`:
    # `z/_x = 0/0 = NaN`, whereas `_x^(_y-1)` gives the correct value at the
    # boundary (e.g. Inf when 0 < _y < 1, since d/dx(x^y)|_{x=0} diverges).
    dx_coeff = _y * _x^(_y - one(P))
    # `z*log(_x)` (not `tangent(y)`) is guarded: when `_x=0`, `z=0` and
    # `log(_x)=-Inf`, so `z*log(_x)*tangent(y) = 0*(-Inf)*tangent(y) = NaN`.
    dy_base = z * log(_x)
    xl = tangent(ix).lanes
    yl = tangent(iy).lanes
    new_lanes = ntuple(Val(N)) do n
        nan_tangent_guard(xl[n], dx_coeff * xl[n]) +
            nan_tangent_guard(z, dy_base * yl[n])
    end
    return Lifted{typeof(z),N}(z, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(^),P,P}}) = true
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
@inline function Mooncake.frule!!(
    ::Lifted{typeof(max),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    ix, iy = _unlift(x), _unlift(y)
    _x, _y = primal(ix), primal(iy)
    z = max(_x, _y)
    xl = tangent(ix).lanes
    yl = tangent(iy).lanes
    new_lanes = _x >= _y ? xl : yl
    return Lifted{typeof(z),N}(z, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(max),P,P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(max)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) >= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(max(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(min),P,P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(min),N}, x::Lifted{P,N}, y::Lifted{P,N}
) where {N}
    ix, iy = _unlift(x), _unlift(y)
    _x, _y = primal(ix), primal(iy)
    z = min(_x, _y)
    xl = tangent(ix).lanes
    yl = tangent(iy).lanes
    new_lanes = _x <= _y ? xl : yl
    return Lifted{typeof(z),N}(z, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(min),P,P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(min)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) <= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(min(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(abs),P}
@inline function Mooncake.frule!!(::Lifted{typeof(abs),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    _x = primal(inner)
    y = abs(_x)
    sign = _x >= zero(P) ? one(P) : -one(P)
    lanes = tangent(inner).lanes
    new_lanes = ntuple(n -> lanes[n] * sign, Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(abs),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(abs)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), _x >= zero(P) ? dy : -dy
    return zero_fcodual(abs(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(Base.eps),P}
@inline function Mooncake.frule!!(::Lifted{typeof(Base.eps),N}, x::Lifted{P,N}) where {N}
    inner = _unlift(x)
    y = eps(primal(inner))
    new_lanes = ntuple(_ -> zero(P), Val(N))
    return Lifted{typeof(y),N}(y, new_lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(Base.eps),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P})
    pb(::P) = NoRData(), zero(P)
    return zero_fcodual(eps(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(nextfloat),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(nextfloat),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = nextfloat(primal(inner))
    return Lifted{typeof(y),N}(y, tangent(inner).lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(nextfloat),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(nextfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(nextfloat(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(prevfloat),P}
@inline function Mooncake.frule!!(
    ::Lifted{typeof(prevfloat),N}, x::Lifted{P,N}
) where {N}
    inner = _unlift(x)
    y = prevfloat(primal(inner))
    return Lifted{typeof(y),N}(y, tangent(inner).lanes)
end
Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(prevfloat),P}}) = true
function Mooncake.rrule!!(::CoDual{typeof(prevfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(prevfloat(primal(x))), pb
end

end # @static if BFloat16s.BFloat16 === Core.BFloat16
#! format: on

end # module MooncakeBFloat16sExt
