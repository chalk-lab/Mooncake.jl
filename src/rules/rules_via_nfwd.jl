#
# nfwd-backed primitive rules for scalar functions.
#
# Each entry registers direct primitive `frule!!` / `rrule!!` wrappers backed by
# the nfwd engine, which avoids hand-coding pullbacks for well-supported scalar
# operations.
#
# MinimalCtx is used throughout here rather than DefaultCtx: several of these
# functions (e.g. tanpi, sincosd, sincospi) contain try/catch internally, which
# Mooncake's IR-transform-based AD cannot handle.  Registering as MinimalCtx
# primitives ensures that the nfwd rule is dispatched directly, bypassing
# any problematic IR transforms.
#
# This file intentionally uses direct primitive wrappers rather than
# `build_primitive_*`: Mooncake still has direct primitive call sites, notably
# public `rrule!!` / `frule!!` examples and rule-to-rule forwarding paths such
# as `getfield -> lgetfield` and `setfield! -> lsetfield!`.
#
# Special nfwd-backed wrappers below:
#   single-input:
#     tanpi (scalar output, DOF=1)
#     sincosd, sincospi, modf (tuple output, DOF=1)
#   multi-input:
#     atan(y, x) (scalar output, DOF=2)
#     clamp(x, lo, hi) (scalar output, DOF=3)
#

# ── nfwd-backed unary scalar rules ─────────────────────────────────────────────
for f in (
    exp,
    exp2,
    exp10,
    expm1,
    sin,
    cos,
    tan,
    sec,
    csc,
    cot,
    sind,
    cosd,
    tand,
    secd,
    cscd,
    cotd,
    sinpi,
    asin,
    acos,
    atan,
    asec,
    acsc,
    acot,
    asind,
    acosd,
    atand,
    asecd,
    acscd,
    acotd,
    sinh,
    cosh,
    tanh,
    sech,
    csch,
    coth,
    asinh,
    acosh,
    atanh,
    asech,
    acsch,
    acoth,
    sinc,
    deg2rad,
    rad2deg,
    Base.FastMath.exp_fast,
    Base.FastMath.exp2_fast,
    Base.FastMath.exp10_fast,
    Base.FastMath.sincos,
)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        function frule!!(fdual::Dual{typeof($f)}, x::Dual{P}) where {P<:IEEEFloat}
            return NfwdMooncake.Rule{Tuple{typeof($f),P},1}()(fdual, x)
        end
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake.RRule{Tuple{typeof($f),P},1}()(fcodual, x)
        end
    end
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(tanpi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(tanpi),P},1}()(f, x)
end
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(tanpi),P},1}()(f, x)
end

# ── clamp(x, lo, hi) ──────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(atan),P,P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(atan)}, y::Dual{P}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(atan),P,P},1}()(f, y, x)
end
function rrule!!(f::CoDual{typeof(atan)}, y::CoDual{P}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(atan),P,P},2}()(f, y, x)
end

# ── clamp(x, lo, hi) ──────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(clamp),P,P,P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(clamp)}, x::Dual{P}, lo::Dual{P}, hi::Dual{P}
) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(clamp),P,P,P},1}()(f, x, lo, hi)
end
function rrule!!(
    f::CoDual{typeof(clamp)}, x::CoDual{P}, lo::CoDual{P}, hi::CoDual{P}
) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(clamp),P,P,P},3}()(f, x, lo, hi)
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincosd)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(sincosd),P},1}()(f, x)
end
function rrule!!(f::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(sincosd),P},1}()(f, x)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincospi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(sincospi),P},1}()(f, x)
end
function rrule!!(f::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(sincospi),P},1}()(f, x)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(modf)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.Rule{Tuple{typeof(modf),P},1}()(f, x)
end
function rrule!!(f::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake.RRule{Tuple{typeof(modf),P},1}()(f, x)
end
