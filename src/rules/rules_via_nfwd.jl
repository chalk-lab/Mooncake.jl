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
        # `$f(::NDual)` has its own overload in Nfwd.jl that propagates
        # partials correctly; the result V's `.value` matches `f(primal(x))`,
        # preserving the canonical V invariant. Use the result type for P_out
        # so tuple-returning primitives (e.g. `sincos`) work correctly —
        # for them y::Tuple{P,P}, dy::Tuple{NDual,NDual}.
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            y = $f(primal(x))
            dy = $f(tangent(x))
            return Lifted{_typeof(y),N}(y, dy)
        end
        function rrule!!(fcodual::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), fcodual, x)
        end
    end
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(tanpi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    return Lifted{P,N}(tanpi(primal(x)), tanpi(tangent(x)))
end
function rrule!!(f::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        function frule!!(
            ::Lifted{typeof($f),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            return Lifted{P,N}($f(primal(x1), primal(x2)), $f(tangent(x1), tangent(x2)))
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(2), fcodual, x1, x2)
        end
    end
end

# Integer-power fastmath rules share the same local derivative as scalar `pow_fast`,
# but only the floating-point base is differentiable.
@is_primitive MinimalCtx Tuple{
    typeof(Base.FastMath.pow_fast),P,I
} where {P<:IEEEFloat,I<:Integer}
function frule!!(
    ::Lifted{typeof(Base.FastMath.pow_fast),N}, x::Lifted{P,N,NDual{P,N}}, n::Lifted{I,N}
) where {N,P<:IEEEFloat,I<:Integer}
    _x = primal(x)
    _n = primal(n)
    y = Base.FastMath.pow_fast(_x, _n)
    # `Nfwd._nfwd_pow_grad_x` returns a scalar `P`; scaling NDual's partials
    # by it and explicitly setting V.value to `y` preserves the invariant
    # (a naive `grad * tangent(x)` would scale `.value` to `grad * x_p`, not `y`).
    grad = Nfwd._nfwd_pow_grad_x(_x, P(_n), float(y))
    new_partials = Nfwd._pt_scale(tangent(x).partials, grad)
    return Lifted{P,N}(y, NDual{P,N}(y, new_partials))
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
            ::Lifted{typeof($f),N},
            x1::Lifted{P,N,NDual{P,N}},
            x2::Lifted{P,N,NDual{P,N}},
            x3::Lifted{P,N,NDual{P,N}},
        ) where {N,P<:IEEEFloat}
            return Lifted{P,N}(
                $f(primal(x1), primal(x2), primal(x3)),
                $f(tangent(x1), tangent(x2), tangent(x3)),
            )
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_rrule_call(Val(3), fcodual, x1, x2, x3)
        end
    end
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincosd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    pv = sincosd(primal(x))
    tv = sincosd(tangent(x))
    return Lifted{Tuple{P,P},N}(pv, tv)
end
function rrule!!(f::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    pv = sincospi(primal(x))
    tv = sincospi(tangent(x))
    return Lifted{Tuple{P,P},N}(pv, tv)
end
function rrule!!(f::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(modf),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    pv = modf(primal(x))
    tv = modf(tangent(x))
    return Lifted{Tuple{P,P},N}(pv, tv)
end
function rrule!!(f::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(hypot),N},
    x::Lifted{P,N,NDual{P,N}},
    xs::Vararg{Lifted{P,N,NDual{P,N}},M},
) where {N,P<:IEEEFloat,M}
    return Lifted{P,N}(
        hypot(primal(x), tuple_map(primal, xs)...),
        hypot(tangent(x), tuple_map(tangent, xs)...),
    )
end
function rrule!!(
    f::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(M + 1), f, x, xs...)
end
