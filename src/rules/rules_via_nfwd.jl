#
# nfwd-backed primitive rules for Tier 1 scalar functions.
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

macro nfwd_unary_scalar_primitive(fn)
    f = esc(fn)
    return quote
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        function frule!!(fdual::Dual{typeof($f)}, x::Dual{P}) where {P<:IEEEFloat}
            return NfwdMooncake.Rule{Tuple{typeof($f),P},1}()(fdual, x)
        end
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake.RRule{Tuple{typeof($f),P},1}()(fcodual, x)
        end
    end
end

# ── nfwd-backed unary scalar rules ─────────────────────────────────────────────

@nfwd_unary_scalar_primitive exp
@nfwd_unary_scalar_primitive exp2
@nfwd_unary_scalar_primitive exp10
@nfwd_unary_scalar_primitive expm1
@nfwd_unary_scalar_primitive sin
@nfwd_unary_scalar_primitive cos
@nfwd_unary_scalar_primitive tan
@nfwd_unary_scalar_primitive sec
@nfwd_unary_scalar_primitive csc
@nfwd_unary_scalar_primitive cot
@nfwd_unary_scalar_primitive sind
@nfwd_unary_scalar_primitive cosd
@nfwd_unary_scalar_primitive tand
@nfwd_unary_scalar_primitive secd
@nfwd_unary_scalar_primitive cscd
@nfwd_unary_scalar_primitive cotd
@nfwd_unary_scalar_primitive sinpi
@nfwd_unary_scalar_primitive asin
@nfwd_unary_scalar_primitive acos
@nfwd_unary_scalar_primitive atan
@nfwd_unary_scalar_primitive asec
@nfwd_unary_scalar_primitive acsc
@nfwd_unary_scalar_primitive acot
@nfwd_unary_scalar_primitive asind
@nfwd_unary_scalar_primitive acosd
@nfwd_unary_scalar_primitive atand
@nfwd_unary_scalar_primitive asecd
@nfwd_unary_scalar_primitive acscd
@nfwd_unary_scalar_primitive acotd
@nfwd_unary_scalar_primitive sinh
@nfwd_unary_scalar_primitive cosh
@nfwd_unary_scalar_primitive tanh
@nfwd_unary_scalar_primitive sech
@nfwd_unary_scalar_primitive csch
@nfwd_unary_scalar_primitive coth
@nfwd_unary_scalar_primitive asinh
@nfwd_unary_scalar_primitive acosh
@nfwd_unary_scalar_primitive atanh
@nfwd_unary_scalar_primitive asech
@nfwd_unary_scalar_primitive acsch
@nfwd_unary_scalar_primitive acoth
@nfwd_unary_scalar_primitive sinc
@nfwd_unary_scalar_primitive deg2rad
@nfwd_unary_scalar_primitive rad2deg
@nfwd_unary_scalar_primitive Base.FastMath.exp_fast
@nfwd_unary_scalar_primitive Base.FastMath.exp2_fast
@nfwd_unary_scalar_primitive Base.FastMath.exp10_fast
@nfwd_unary_scalar_primitive Base.FastMath.sincos

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
