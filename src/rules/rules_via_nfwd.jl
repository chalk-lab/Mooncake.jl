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

@inline function _nfwd_public_primitive_frule_call(fdual::Dual, x::Vararg{Any,N}) where {N}
    return Mooncake._nfwd_primitive_frule_call(
        Mooncake._nfwd_primitive_width(x...), fdual, x...
    )
end

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
    mod2pi,
    Base.eps,
    nextfloat,
    prevfloat,
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
        function frule!!(
            fdual::Dual{typeof($f)}, x::Dual{P,T}
        ) where {P<:IEEEFloat,T}
            return _nfwd_public_primitive_frule_call(fdual, x)
        end
        function frule!!(fdual::Dual{typeof($f)}, x::Nfwd.NDual{P,N}) where {P<:IEEEFloat,N}
            return _nfwd_public_primitive_frule_call(fdual, x)
        end
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return Mooncake._nfwd_primitive_rrule_call(Val(1), fcodual, x)
        end
    end
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(tanpi)}, x::Dual{P,T}
) where {P<:IEEEFloat,T}
    return _nfwd_public_primitive_frule_call(f, x)
end
function frule!!(f::Dual{typeof(tanpi)}, x::Nfwd.NDual{P,N}) where {P<:IEEEFloat,N}
    return _nfwd_public_primitive_frule_call(f, x)
end
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return Mooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        function frule!!(
            fdual::Dual{typeof($f)}, x1::Dual{P,T1}, x2::Dual{P,T2}
        ) where {P<:IEEEFloat,T1,T2}
            return _nfwd_public_primitive_frule_call(fdual, x1, x2)
        end
        function frule!!(
            fdual::Dual{typeof($f)}, x1::Nfwd.NDual{P,N}, x2::Nfwd.NDual{P,N}
        ) where {P<:IEEEFloat,N}
            return _nfwd_public_primitive_frule_call(fdual, x1, x2)
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            return Mooncake._nfwd_primitive_rrule_call(Val(2), fcodual, x1, x2)
        end
    end
end

# Integer-power fastmath rules share the same local derivative as scalar `pow_fast`,
# but only the floating-point base is differentiable.
@is_primitive MinimalCtx Tuple{
    typeof(Base.FastMath.pow_fast),P,I
} where {P<:IEEEFloat,I<:Integer}
function frule!!(
    ::Dual{typeof(Base.FastMath.pow_fast)}, x::Dual{P}, n::Dual{I}
) where {P<:IEEEFloat,I<:Integer}
    _x, dx = extract(x)
    _n = primal(n)
    y = Base.FastMath.pow_fast(_x, _n)
    return Dual(y, Nfwd._nfwd_pow_grad_x(_x, P(_n), float(y)) * dx)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.pow_fast)}, x::CoDual{P}, n::CoDual{I}
) where {P<:IEEEFloat,I<:Integer}
    _x = primal(x)
    _n = primal(n)
    y = Base.FastMath.pow_fast(_x, _n)
    function pow_fast_pb!!(dy::P)
        return NoRData(), Nfwd._nfwd_pow_grad_x(_x, P(_n), float(y)) * dy, NoRData()
    end
    return zero_fcodual(y), pow_fast_pb!!
end

for f in (clamp,)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P,P} where {P<:IEEEFloat}
        function frule!!(
            fdual::Dual{typeof($f)},
            x1::Dual{P,T1},
            x2::Dual{P,T2},
            x3::Dual{P,T3},
        ) where {P<:IEEEFloat,T1,T2,T3}
            return _nfwd_public_primitive_frule_call(fdual, x1, x2, x3)
        end
        function frule!!(
            fdual::Dual{typeof($f)},
            x1::Nfwd.NDual{P,N},
            x2::Nfwd.NDual{P,N},
            x3::Nfwd.NDual{P,N},
        ) where {P<:IEEEFloat,N}
            return _nfwd_public_primitive_frule_call(fdual, x1, x2, x3)
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
        ) where {P<:IEEEFloat}
            return Mooncake._nfwd_primitive_rrule_call(Val(3), fcodual, x1, x2, x3)
        end
    end
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(sincosd)}, x::Dual{P,T}
) where {P<:IEEEFloat,T}
    return _nfwd_public_primitive_frule_call(f, x)
end
function frule!!(f::Dual{typeof(sincosd)}, x::Nfwd.NDual{P,N}) where {P<:IEEEFloat,N}
    return _nfwd_public_primitive_frule_call(f, x)
end
function rrule!!(f::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    return Mooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(sincospi)}, x::Dual{P,T}
) where {P<:IEEEFloat,T}
    return _nfwd_public_primitive_frule_call(f, x)
end
function frule!!(f::Dual{typeof(sincospi)}, x::Nfwd.NDual{P,N}) where {P<:IEEEFloat,N}
    return _nfwd_public_primitive_frule_call(f, x)
end
function rrule!!(f::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return Mooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(modf)}, x::Dual{P,T}
) where {P<:IEEEFloat,T}
    return _nfwd_public_primitive_frule_call(f, x)
end
function frule!!(f::Dual{typeof(modf)}, x::Nfwd.NDual{P,N}) where {P<:IEEEFloat,N}
    return _nfwd_public_primitive_frule_call(f, x)
end
function rrule!!(f::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    return Mooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(hypot)}, x::Dual{P,T}, xs::Vararg{Dual{P},M}
) where {P<:IEEEFloat,T,M}
    return _nfwd_public_primitive_frule_call(f, x, xs...)
end
function frule!!(
    f::Dual{typeof(hypot)}, x::Nfwd.NDual{P,N}, xs::Vararg{Nfwd.NDual{P,N},M}
) where {P<:IEEEFloat,N,M}
    return _nfwd_public_primitive_frule_call(f, x, xs...)
end
function rrule!!(
    f::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    return Mooncake._nfwd_primitive_rrule_call(Val(M + 1), f, x, xs...)
end
