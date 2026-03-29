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
# Warning: avoid using `Rule` / `RRule` as hidden cached state for primitive rules.
# Those wrapper types own mutable workspace and are safe to reuse only when the caller
# explicitly owns the instance. Primitive rules are entered through ordinary dispatch, so
# caching a wrapper here would hide shared mutable state behind a plain rule method and
# make thread-safety hazards much less obvious.
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
    log,
    log10,
    log2,
    log1p,
    sqrt,
    cbrt,
    sin,
    cos,
    cospi,
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
    Base.FastMath.atan_fast,
    Base.FastMath.sincos,
)
    # Call the primitive nfwd entrypoints directly here rather than constructing
    # `Rule{...}()` / `RRule{...}()` on every call. These wrappers sit on hot scalar paths,
    # so avoiding per-invocation wrapper construction keeps them allocation-free. See the
    # file-level warning above for why hidden cached Rule/RRule state is also a bad fit
    # for primitive rules.
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        function frule!!(fdual::Dual{typeof($f)}, x::Dual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), fdual, x)
        end
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), fcodual, x)
        end
    end
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(tanpi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── clamp(x, lo, hi) ──────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(atan),P,P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(atan)}, y::Dual{P}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, y, x)
end
function rrule!!(f::CoDual{typeof(atan)}, y::CoDual{P}, x::CoDual{P}) where {P<:IEEEFloat}
    # Use the primitive nfwd entrypoint so the 2-argument reverse rule stays allocation-free.
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(2), f, y, x)
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.atan_fast),P,P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(Base.FastMath.atan_fast)}, y::Dual{P}, x::Dual{P}
) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, y, x)
end
function rrule!!(
    f::CoDual{typeof(Base.FastMath.atan_fast)}, y::CoDual{P}, x::CoDual{P}
) where {P<:IEEEFloat}
    # Base.FastMath.atan_fast shares the same 2-argument reverse path as atan.
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(2), f, y, x)
end

@is_primitive MinimalCtx Tuple{typeof(log),P,P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(log)}, b::Dual{P}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, b, x)
end
function rrule!!(f::CoDual{typeof(log)}, b::CoDual{P}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(2), f, b, x)
end

# ── clamp(x, lo, hi) ──────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(clamp),P,P,P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(clamp)}, x::Dual{P}, lo::Dual{P}, hi::Dual{P}
) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x, lo, hi)
end
function rrule!!(
    f::CoDual{typeof(clamp)}, x::CoDual{P}, lo::CoDual{P}, hi::CoDual{P}
) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(3), f, x, lo, hi)
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincosd)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
function rrule!!(f::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(sincospi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
function rrule!!(f::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(modf)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end
function rrule!!(f::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(hypot)}, x::Dual{P}, xs::Vararg{Dual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x, xs...)
end
function rrule!!(
    f::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(M + 1), f, x, xs...)
end
