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
    Lifted,
    primal,
    tangent,
    extract,
    zero_fcodual,
    MinimalCtx

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

# Forward-mode canonical V for `Core.BFloat16` — `NTuple{N, BFloat16}`,
# i.e. N parallel BFloat16 partials, one per lane. BFloat16 is not in
# `IEEEFloat` (so `NDual` doesn't cover it), and it's a single-number
# leaf type, so the structural-lift fallback doesn't apply either.
# Mirrors the TWP / Ptr V pattern (`NTuple{N, T}`).
@foldable @inline function Mooncake.dual_type(::Val{N}, ::Type{P}) where {N}
    return NTuple{N,P}
end
@foldable @inline function Mooncake.lifted_type(::Val{N}, ::Type{P}) where {N}
    return Mooncake.Lifted{P,N,NTuple{N,P}}
end
# A BFloat16 is a single-number leaf, so its width-1 V `Tuple{BFloat16}` is a leaf, not a structural
# tuple. Provide the `lift` boundary and override `_unlift_seed` to read the lane directly — the
# generic `_unlift_seed` would treat the tuple as per-field and index the fieldless primal
# (`BoundsError: access DataType at index [1]`). Mirrors the TwicePrecision NTuple-V pattern.
@inline Mooncake.lift(x::P, ẋ::P) = Mooncake.Lifted{P,1}(x, (ẋ,))
@inline Mooncake._unlift_seed(x::Mooncake.Lifted{P,1,Tuple{P}}, ::IdDict) = Mooncake.tangent(x, 1)

# Conversions

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float32},P}
function Mooncake.rrule!!(::CoDual{Type{Float32}}, x::CoDual{P})
    pb(dy::Float32) = NoRData(), P(dy)
    return zero_fcodual(Float32(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{Float64},P}
function Mooncake.rrule!!(::CoDual{Type{Float64}}, x::CoDual{P})
    pb(dy::Float64) = NoRData(), P(Float32(dy))
    return zero_fcodual(Float64(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float32}
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float32})
    pb(dy::P) = NoRData(), Float32(dy)
    return zero_fcodual(P(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{Type{P},Float64}
function Mooncake.rrule!!(::CoDual{Type{P}}, x::CoDual{Float64})
    pb(dy::P) = NoRData(), Float64(Float32(dy))
    return zero_fcodual(P(Float32(primal(x)))), pb
end

# Math rules

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sqrt),P}
function Mooncake.rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P})
    y = sqrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (2 * y))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cbrt),P}
function Mooncake.rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P})
    y = cbrt(primal(x))
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (3 * y^2))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp),P}
function Mooncake.rrule!!(::CoDual{typeof(exp)}, x::CoDual{P})
    y = exp(primal(x))
    pb(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp2),P}
function Mooncake.rrule!!(::CoDual{typeof(exp2)}, x::CoDual{P})
    y = exp2(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(2.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(exp10),P}
function Mooncake.rrule!!(::CoDual{typeof(exp10)}, x::CoDual{P})
    y = exp10(primal(x))
    pb(dy::P) = NoRData(), dy * y * P(log(10.0f0))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(expm1),P}
function Mooncake.rrule!!(::CoDual{typeof(expm1)}, x::CoDual{P})
    y = expm1(primal(x))
    pb(dy::P) = NoRData(), dy * (y + one(P))
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log),P}
function Mooncake.rrule!!(::CoDual{typeof(log)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / _x)
    return zero_fcodual(log(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log2),P}
function Mooncake.rrule!!(::CoDual{typeof(log2)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(2.0f0))))
    return zero_fcodual(log2(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log10),P}
function Mooncake.rrule!!(::CoDual{typeof(log10)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (_x * P(log(10.0f0))))
    return zero_fcodual(log10(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(log1p),P}
function Mooncake.rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) + _x))
    return zero_fcodual(log1p(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sin),P}
function Mooncake.rrule!!(::CoDual{typeof(sin)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), dy * c
    return zero_fcodual(s), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cos),P}
function Mooncake.rrule!!(::CoDual{typeof(cos)}, x::CoDual{P})
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    pb(dy::P) = NoRData(), -dy * s
    return zero_fcodual(c), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tan),P}
function Mooncake.rrule!!(::CoDual{typeof(tan)}, x::CoDual{P})
    y = tan(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) + y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asin),P}
function Mooncake.rrule!!(::CoDual{typeof(asin)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(one(P) - _x^2))
    return zero_fcodual(asin(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acos),P}
function Mooncake.rrule!!(::CoDual{typeof(acos)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, -dy / sqrt(one(P) - _x^2))
    return zero_fcodual(acos(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atan),P}
function Mooncake.rrule!!(::CoDual{typeof(atan)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / (one(P) + _x^2)
    return zero_fcodual(atan(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(sinh),P}
function Mooncake.rrule!!(::CoDual{typeof(sinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * cosh(_x)
    return zero_fcodual(sinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(cosh),P}
function Mooncake.rrule!!(::CoDual{typeof(cosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy * sinh(_x)
    return zero_fcodual(cosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(tanh),P}
function Mooncake.rrule!!(::CoDual{typeof(tanh)}, x::CoDual{P})
    y = tanh(primal(x))
    pb(dy::P) = NoRData(), dy * (one(P) - y^2)
    return zero_fcodual(y), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(asinh),P}
function Mooncake.rrule!!(::CoDual{typeof(asinh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), dy / sqrt(one(P) + _x^2)
    return zero_fcodual(asinh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(acosh),P}
function Mooncake.rrule!!(::CoDual{typeof(acosh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / sqrt(_x^2 - one(P)))
    return zero_fcodual(acosh(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(atanh),P}
function Mooncake.rrule!!(::CoDual{typeof(atanh)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), nan_tangent_guard(dy, dy / (one(P) - _x^2))
    return zero_fcodual(atanh(_x)), pb
end


Mooncake.@is_primitive MinimalCtx Tuple{typeof(hypot),P,P}
function Mooncake.rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P})
    _x, _y = primal(x), primal(y)
    h = hypot(_x, _y)
    pb(dh::P) = NoRData(),
    nan_tangent_guard(dh, dh * _x / h),
    nan_tangent_guard(dh, dh * _y / h)
    return zero_fcodual(h), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(^),P,P}
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
function Mooncake.rrule!!(::CoDual{typeof(max)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) >= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(max(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(min),P,P}
function Mooncake.rrule!!(::CoDual{typeof(min)}, x::CoDual{P}, y::CoDual{P})
    _x_wins = primal(x) <= primal(y)
    pb(dz::P) = NoRData(), _x_wins ? dz : zero(P), _x_wins ? zero(P) : dz
    return zero_fcodual(min(primal(x), primal(y))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(abs),P}
function Mooncake.rrule!!(::CoDual{typeof(abs)}, x::CoDual{P})
    _x = primal(x)
    pb(dy::P) = NoRData(), _x >= zero(P) ? dy : -dy
    return zero_fcodual(abs(_x)), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(Base.eps),P}
function Mooncake.rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P})
    pb(::P) = NoRData(), zero(P)
    return zero_fcodual(eps(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(nextfloat),P}
function Mooncake.rrule!!(::CoDual{typeof(nextfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(nextfloat(primal(x))), pb
end

Mooncake.@is_primitive MinimalCtx Tuple{typeof(prevfloat),P}
function Mooncake.rrule!!(::CoDual{typeof(prevfloat)}, x::CoDual{P})
    pb(dy::P) = NoRData(), dy
    return zero_fcodual(prevfloat(primal(x))), pb
end

# ──────────────────────────────────────────────────────────────────────────
# Forward `frule!!`s for the BFloat16 primitives (the matching `rrule!!`s are above).
#
# V for BFloat16 is `NTuple{Nw, BFloat16}` (defined at the top of this
# module). Each body applies the primitive's derivative with a per-lane
# `ntuple` over the partials tuple. For BFloat16 → Float32/Float64
# conversion rules, the output V is `NDual{F, Nw}` because the output
# primal is in `IEEEFloat`.
# ──────────────────────────────────────────────────────────────────────────

const _PNT{N} = NTuple{N,P}
using Mooncake: NDual

# Conversions
function Mooncake.frule!!(
    ::Lifted{Type{Float32},Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    y = Float32(primal(x))
    parts = tangent(x)
    dy = ntuple(k -> Float32(parts[k]), Val(Nw))
    return Lifted{Float32,Nw}(y, NDual{Float32,Nw}(y, dy))
end
function Mooncake.frule!!(
    ::Lifted{Type{Float64},Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    y = Float64(primal(x))
    parts = tangent(x)
    dy = ntuple(k -> Float64(parts[k]), Val(Nw))
    return Lifted{Float64,Nw}(y, NDual{Float64,Nw}(y, dy))
end
function Mooncake.frule!!(
    ::Lifted{Type{P},Nw}, x::Lifted{Float32,Nw,NDual{Float32,Nw}}
) where {Nw}
    y = P(primal(x))
    parts = tangent(x).partials
    dy = ntuple(k -> P(parts[k]), Val(Nw))
    return Lifted{P,Nw}(y, dy)
end
function Mooncake.frule!!(
    ::Lifted{Type{P},Nw}, x::Lifted{Float64,Nw,NDual{Float64,Nw}}
) where {Nw}
    y = P(Float32(primal(x)))
    parts = tangent(x).partials
    dy = ntuple(k -> P(Float32(parts[k])), Val(Nw))
    return Lifted{P,Nw}(y, dy)
end

# Unary scalar primitives — per-lane `deriv * partial` (with optional
# nan_tangent_guard for safety on degenerate inputs).
for (op, deriv_expr, guarded) in (
    (:sqrt, :(dx / (2 * y)), true),
    (:cbrt, :(dx / (3 * y^2)), true),
    (:exp, :(dx * y), false),
    (:exp2, :(dx * y * P(log(2.0f0))), false),
    (:exp10, :(dx * y * P(log(10.0f0))), false),
    (:expm1, :(dx * (y + one(P))), false),
    (:log, :(dx / _x), true),
    (:log2, :(dx / (_x * P(log(2.0f0)))), true),
    (:log10, :(dx / (_x * P(log(10.0f0)))), true),
    (:log1p, :(dx / (one(P) + _x)), true),
    (:tan, :(dx * (one(P) + y^2)), false),
    (:asin, :(dx / sqrt(one(P) - _x^2)), true),
    (:acos, :(-dx / sqrt(one(P) - _x^2)), true),
    (:atan, :(dx / (one(P) + _x^2)), false),
    (:sinh, :(dx * cosh(_x)), false),
    (:cosh, :(dx * sinh(_x)), false),
    (:tanh, :(dx * (one(P) - y^2)), false),
    (:asinh, :(dx / sqrt(one(P) + _x^2)), false),
    (:acosh, :(dx / sqrt(_x^2 - one(P))), true),
    (:atanh, :(dx / (one(P) - _x^2)), true),
)
    lane_expr = guarded ? :(nan_tangent_guard(dx, $deriv_expr)) : deriv_expr
    @eval function Mooncake.frule!!(
        ::Lifted{typeof($op),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
    ) where {Nw}
        _x = primal(x)
        y = $op(_x)
        parts = tangent(x)
        dy = ntuple(Val(Nw)) do k
            dx = parts[k]
            $lane_expr
        end
        return Lifted{P,Nw}(y, dy)
    end
end

# sin / cos — use separate calls because sincos(::BFloat16) is broken in 1.12.
function Mooncake.frule!!(
    ::Lifted{typeof(sin),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    parts = tangent(x)
    dy = ntuple(k -> parts[k] * c, Val(Nw))
    return Lifted{P,Nw}(s, dy)
end
function Mooncake.frule!!(
    ::Lifted{typeof(cos),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    _x = primal(x)
    s = sin(_x)
    c = cos(_x)
    parts = tangent(x)
    dy = ntuple(k -> -parts[k] * s, Val(Nw))
    return Lifted{P,Nw}(c, dy)
end

# Binary: hypot, ^
function Mooncake.frule!!(
    ::Lifted{typeof(hypot),Nw},
    x::Lifted{P,Nw,_PNT{Nw}},
    y::Lifted{P,Nw,_PNT{Nw}},
) where {Nw}
    _x = primal(x)
    _y = primal(y)
    h = hypot(_x, _y)
    x_parts = tangent(x)
    y_parts = tangent(y)
    dh = ntuple(Val(Nw)) do k
        (
            nan_tangent_guard(x_parts[k], _x * x_parts[k]) +
            nan_tangent_guard(y_parts[k], _y * y_parts[k])
        ) / h
    end
    return Lifted{P,Nw}(h, dh)
end
function Mooncake.frule!!(
    ::Lifted{typeof(^),Nw},
    x::Lifted{P,Nw,_PNT{Nw}},
    y::Lifted{P,Nw,_PNT{Nw}},
) where {Nw}
    _x = primal(x)
    _y = primal(y)
    z = _x^_y
    x_parts = tangent(x)
    y_parts = tangent(y)
    dz = ntuple(Val(Nw)) do k
        nan_tangent_guard(x_parts[k], _y * _x^(_y - one(P)) * x_parts[k]) +
            nan_tangent_guard(z, z * log(_x) * y_parts[k])
    end
    return Lifted{P,Nw}(z, dz)
end

# Binary: max, min — branch on which arg wins
function Mooncake.frule!!(
    ::Lifted{typeof(max),Nw},
    x::Lifted{P,Nw,_PNT{Nw}},
    y::Lifted{P,Nw,_PNT{Nw}},
) where {Nw}
    _x = primal(x)
    _y = primal(y)
    x_parts = tangent(x)
    y_parts = tangent(y)
    dz = ntuple(k -> _x >= _y ? x_parts[k] : y_parts[k], Val(Nw))
    return Lifted{P,Nw}(max(_x, _y), dz)
end
function Mooncake.frule!!(
    ::Lifted{typeof(min),Nw},
    x::Lifted{P,Nw,_PNT{Nw}},
    y::Lifted{P,Nw,_PNT{Nw}},
) where {Nw}
    _x = primal(x)
    _y = primal(y)
    x_parts = tangent(x)
    y_parts = tangent(y)
    dz = ntuple(k -> _x <= _y ? x_parts[k] : y_parts[k], Val(Nw))
    return Lifted{P,Nw}(min(_x, _y), dz)
end

# abs — sign-based branch on per-lane tangent
function Mooncake.frule!!(
    ::Lifted{typeof(abs),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    _x = primal(x)
    parts = tangent(x)
    dy = ntuple(k -> _x >= zero(P) ? parts[k] : -parts[k], Val(Nw))
    return Lifted{P,Nw}(abs(_x), dy)
end

# Base.eps — zero tangent
function Mooncake.frule!!(
    ::Lifted{typeof(Base.eps),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    return Lifted{P,Nw}(eps(primal(x)), ntuple(_ -> zero(P), Val(Nw)))
end

# nextfloat, prevfloat — passthrough tangent
function Mooncake.frule!!(
    ::Lifted{typeof(nextfloat),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    return Lifted{P,Nw}(nextfloat(primal(x)), tangent(x))
end
function Mooncake.frule!!(
    ::Lifted{typeof(prevfloat),Nw}, x::Lifted{P,Nw,_PNT{Nw}}
) where {Nw}
    return Lifted{P,Nw}(prevfloat(primal(x)), tangent(x))
end

end # @static if BFloat16s.BFloat16 === Core.BFloat16
#! format: on

end # module MooncakeBFloat16sExt
