# Historical Note:
#
# This file adds rules for all functions which DiffRules.jl defines rules for, and which
# reside in Base. Originally, this file imported rules directly from DiffRules.jl.
# Unfortunately, there were a number of issues with this:
# 1. Package extensions: DiffRules.jl was written long before package extensions were added
#   to Julia. As a result, a couple of packages are direct dependencies of DiffRules,
#   notably SpecialFunctions.jl, which we do not wish to make indirect dependencies of
#   Mooncake.jl. All in all, by removing DiffRules as a dependency, we also remove:
#   DocStringExtensions, JLLWrappers, LogExpFunctions, NaNMath, OpenSpecFun_jll,
#   OpenLibm_jll.
# 2. Interaction with Revise.jl: most modern development workflows involve using Revise.jl.
#   Unfortunately, putting `@eval` statements in a loop does not seem to play nicely with
#   it, meaning that every time you want to tweak something in the loop, you have to restart
#   your session. Such an `@eval` loop was needed for DiffRules.jl rules.
# 3. Errors in the eval loop can cause spooky action-at-a-distance errors, which are hard to
#   debug.
# 4. Some of the rules in DiffRules are not implemented in an optimal manner, and it is
#   unclear that they _could_ be implemented in an optimal manner. For example, the rules
#   for `sin` and `cos` are unable to make use of the `sincos` function (which computes both
#   `sin` and `cos` at the same time at negligible additional cost to computing either `sin`
#   or `cos` by itself), and are therefore unable to provide optimal performance.
# 5. Readability: while the @eval-loop code was concise, it was rather non-standard, and
#   quite hard to parse.
#
# There were essentially no remaining advantages to using an @eval-loop to import rules
# from DiffRules, so this file now defines the remaining scalar rules directly.

# Forward rules use `Nfwd`'s NDual overloads; reverse uses native analytic pullbacks.
@zero_derivative MinimalCtx Tuple{typeof(log),Int}

# Like `_fwd_guarded_scale`, keep inactive cotangents zero even at infinite derivatives.
@inline _rvs_guarded_scale(ȳ::T, grad::T) where {T} = iszero(ȳ) ? zero(T) : ȳ * grad

# Horner on dual coefficients preserves their partials and avoids transform overhead.
# Reverse and integer coefficients (which lift to `NoDual`) keep the derived path.
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(evalpoly),P,Tuple{Vararg{P}}
} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(evalpoly),N},
    x::Lifted{P,N,NDual{P,N}},
    p::Lifted{<:Tuple{Vararg{P}},N,<:Tuple{Vararg{NDual{P,N}}}},
) where {N,P<:IEEEFloat}
    dy = evalpoly(tangent(x), tangent(p))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end

# ---- unary scalar rules ----
# Unary rules with a single guarded coefficient; evaluate it inside the pullback.
for (f, coefficient) in (
    (exp, :(y)),
    (exp2, :(y * oftype(y, log(2)))),
    (exp10, :(y * oftype(y, log(10)))),
    (expm1, :(exp(_x))),
    (log, :(inv(_x))),
    (log2, :(inv(_x * oftype(_x, log(2))))),
    (log10, :(inv(_x * oftype(_x, log(10))))),
    (log1p, :(inv(one(_x) + _x))),
    (sqrt, :(inv(2 * y))),
    (cbrt, :(inv(3 * y^2))),
    (sec, :(y * tan(_x))),
    (csc, :(-y * cot(_x))),
    (cot, :(-(one(y) + y^2))),
    (asin, :(inv(sqrt(one(_x) - _x^2)))),
    (acos, :(-inv(sqrt(one(_x) - _x^2)))),
    (atan, :(inv(one(_x) + _x^2))),
    (asec, :(inv(abs(_x) * sqrt(_x^2 - one(_x))))),
    (acsc, :(-inv(abs(_x) * sqrt(_x^2 - one(_x))))),
    (acot, :(-inv(one(_x) + _x^2))),
    (sinh, :(cosh(_x))),
    (cosh, :(sinh(_x))),
    (sech, :(-tanh(_x) * y)),
    (csch, :(-coth(_x) * y)),
    (coth, :(-csch(_x)^2)),
    (atanh, :(inv(one(_x) - _x^2))),
    (asech, :(-inv(_x * sqrt(one(_x) - _x^2)))),
    (acsch, :(-inv(abs(_x) * sqrt(one(_x) + _x^2)))),
    (acoth, :(inv(one(_x) - _x^2))),
    (secd, :(deg2rad(y * tand(_x)))),
    (cscd, :(-deg2rad(y * cotd(_x)))),
    (cotd, :(-deg2rad(one(y) + y^2))),
    (asind, :(inv(deg2rad(sqrt(one(_x) - _x^2))))),
    (acosd, :(-inv(deg2rad(sqrt(one(_x) - _x^2))))),
    (atand, :(inv(deg2rad(one(_x) + _x^2)))),
    (asecd, :(inv(deg2rad(abs(_x) * sqrt(_x^2 - one(_x)))))),
    (acscd, :(-inv(deg2rad(abs(_x) * sqrt(_x^2 - one(_x)))))),
    (acotd, :(-inv(deg2rad(one(_x) + _x^2)))),
    (deg2rad, :(deg2rad(one(_x)))),
    (rad2deg, :(rad2deg(one(_x)))),
    (sinc, :(cosc(_x))),
    (nextfloat, :(one(_x))),
    (prevfloat, :(one(_x))),
    (Base.FastMath.exp_fast, :(y)),
    (Base.FastMath.exp2_fast, :(y * oftype(y, log(2)))),
    (Base.FastMath.exp10_fast, :(y * oftype(y, log(10)))),
    (Base.FastMath.atan_fast, :(inv(one(_x) + _x^2))),
)
    pb = Symbol(nameof(f), :_pb)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            dy = $f(tangent(x))
            y = dy.value
            return Lifted{_typeof(y),N}(y, dy)
        end
        function rrule!!(::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            _x = primal(x)
            y = $f(_x)
            $pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, $coefficient))
            return zero_fcodual(y), $pb
        end
    end
end

@is_primitive MinimalCtx Tuple{typeof(tanh),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(tanh),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = tanh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(tanh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = tanh(_x)
    function tanh_pb(ȳ::P)
        # As in NDual, avoid `1 - y^2`: it vanishes when tanh rounds to 1, before sech² does.
        u = exp(-2 * abs(_x))
        return NoRData(), _rvs_guarded_scale(ȳ, 4u / (one(P) + u)^2)
    end
    return zero_fcodual(y), tanh_pb
end

@is_primitive MinimalCtx Tuple{typeof(asinh),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(asinh),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = asinh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asinh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asinh(_x)
    c = if abs(_x) > sqrt(floatmax(P))
        inv(hypot(_x, one(_x)))
    else
        inv(sqrt(_x^2 + one(_x)))
    end
    asinh_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, c))
    return zero_fcodual(y), asinh_pb
end

@is_primitive MinimalCtx Tuple{typeof(acosh),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acosh),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acosh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acosh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acosh(_x)
    c = if abs(_x) > sqrt(floatmax(P))
        inv(abs(_x))
    else
        inv(sqrt(_x^2 - one(_x)))
    end
    acosh_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, c))
    return zero_fcodual(y), acosh_pb
end

@is_primitive MinimalCtx Tuple{typeof(mod2pi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(mod2pi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = mod2pi(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(mod2pi)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = mod2pi(_x)
    mod2pi_pb(ȳ::P) = (
        NoRData(),
        _rvs_guarded_scale(
            ȳ, ifelse(isinteger(_x / oftype(_x, 2π)), oftype(_x, NaN), one(_x))
        ),
    )
    return zero_fcodual(y), mod2pi_pb
end

# ---- fused trig (sin/cos/tan families): one shared `sincos`-type call for value + derivative ----
@is_primitive MinimalCtx Tuple{typeof(sin),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sin),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincos(v)
    y = s
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, c)))
end
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    y = s
    sin_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, c))
    return zero_fcodual(y), sin_pb
end

@is_primitive MinimalCtx Tuple{typeof(cos),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cos),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincos(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, -s)))
end
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    y = c
    cos_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, -s))
    return zero_fcodual(y), cos_pb
end

@is_primitive MinimalCtx Tuple{typeof(tan),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(tan),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = tan(tangent(x))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(tan)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    y = tan(v)
    tan_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, one(y) + y^2))
    return zero_fcodual(y), tan_pb
end

@is_primitive MinimalCtx Tuple{typeof(sind),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sind),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincosd(v)
    y = s
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, deg2rad(c))))
end
function rrule!!(::CoDual{typeof(sind)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    y = s
    sind_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, deg2rad(c)))
    return zero_fcodual(y), sind_pb
end

@is_primitive MinimalCtx Tuple{typeof(cosd),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cosd),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincosd(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, -deg2rad(s))))
end
function rrule!!(::CoDual{typeof(cosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    y = c
    cosd_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, -deg2rad(s)))
    return zero_fcodual(y), cosd_pb
end

@is_primitive MinimalCtx Tuple{typeof(tand),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(tand),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    y = tand(v)
    return Lifted{P,N}(
        y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, deg2rad(one(y) + y^2)))
    )
end
function rrule!!(::CoDual{typeof(tand)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    y = tand(v)
    tand_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, deg2rad(one(y) + y^2)))
    return zero_fcodual(y), tand_pb
end

@is_primitive MinimalCtx Tuple{typeof(sinpi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sinpi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincospi(v)
    y = s
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, oftype(v, π) * c)))
end
function rrule!!(::CoDual{typeof(sinpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincospi(v)
    y = s
    sinpi_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, oftype(v, π) * c))
    return zero_fcodual(y), sinpi_pb
end

@is_primitive MinimalCtx Tuple{typeof(cospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(cospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincospi(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, -oftype(v, π) * s)))
end
function rrule!!(::CoDual{typeof(cospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincospi(v)
    y = c
    cospi_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, -oftype(v, π) * s))
    return zero_fcodual(y), cospi_pb
end

# ---- binary scalar rules ----
@is_primitive MinimalCtx Tuple{typeof(atan),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(atan),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = atan(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(atan)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = atan(a, b)
    r2 = a^2 + b^2
    regular = isfinite(r2) && !iszero(r2)
    h = regular ? zero(P) : hypot(a, b)
    c1 = regular ? b / r2 : (b / h) / h
    c2 = regular ? -a / r2 : (-a / h) / h
    atan_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, c1), _rvs_guarded_scale(ȳ, c2))
    return zero_fcodual(y), atan_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.atan_fast),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.atan_fast),N},
    x1::Lifted{P,N,NDual{P,N}},
    x2::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.atan_fast(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.atan_fast)}, x1::CoDual{P}, x2::CoDual{P}
) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = Base.FastMath.atan_fast(a, b)
    r2 = a^2 + b^2
    atan_fast_pb(ȳ::P) = (
        NoRData(), _rvs_guarded_scale(ȳ, b / r2), _rvs_guarded_scale(ȳ, -a / r2)
    )
    return zero_fcodual(y), atan_fast_pb
end

@is_primitive MinimalCtx Tuple{typeof(log),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(log),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = log(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(log)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = log(a, b)
    lb = log(a)
    log_pb(ȳ::P) = (
        NoRData(), _rvs_guarded_scale(ȳ, -y / (a * lb)), _rvs_guarded_scale(ȳ, inv(b * lb))
    )
    return zero_fcodual(y), log_pb
end

@is_primitive MinimalCtx Tuple{typeof(mod),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(mod),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = mod(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(mod)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = mod(a, b)
    u = a / b
    nan = oftype(u, NaN)
    isint = isinteger(u)
    mod_pb(ȳ::P) = (
        NoRData(),
        _rvs_guarded_scale(ȳ, ifelse(isint, nan, one(u))),
        _rvs_guarded_scale(ȳ, ifelse(isint, nan, -floor(u))),
    )
    return zero_fcodual(y), mod_pb
end

# Avoid `rem_internal`'s bitcast guard, also reached by divrem's RoundToZero branch.
# Unlike mod, rem keeps a finite one-sided subgradient at integer ratios, as modf needs.
@is_primitive MinimalCtx Tuple{typeof(rem),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(rem),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = rem(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(rem)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = rem(a, b)
    c = trunc(a / b)
    # Only the second coefficient can be non-finite (`b == 0`); the first is exactly 1.
    rem_pb(ȳ::P) = (NoRData(), ȳ, _rvs_guarded_scale(ȳ, -c))
    return zero_fcodual(y), rem_pb
end

# As with rem, flipsign/ldexp/rem_fast need rules to avoid the bitcast guard.
@is_primitive MinimalCtx Tuple{typeof(flipsign),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(flipsign),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = flipsign(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(flipsign)}, x1::CoDual{P}, x2::CoDual{P}
) where {P<:IEEEFloat}
    # Preserve -0.0's sign; the second argument is piecewise constant away from zero.
    s = flipsign(one(P), primal(x2))
    flipsign_pb(ȳ::P) = (NoRData(), ȳ * s, zero(P))
    return zero_fcodual(flipsign(primal(x1), primal(x2))), flipsign_pb
end

@is_primitive MinimalCtx Tuple{typeof(ldexp),P,Integer} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(ldexp),N}, x::Lifted{P,N,NDual{P,N}}, n::Lifted{<:Integer}
) where {N,P<:IEEEFloat}
    dy = ldexp(tangent(x), primal(n))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(ldexp)}, x::CoDual{P}, n::CoDual{<:Integer}
) where {P<:IEEEFloat}
    _n = primal(n)
    ldexp_pb(ȳ::P) = (NoRData(), ldexp(ȳ, _n), NoRData())
    return zero_fcodual(ldexp(primal(x), _n)), ldexp_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.rem_fast),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.rem_fast),N},
    x1::Lifted{P,N,NDual{P,N}},
    x2::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.rem_fast(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.rem_fast)}, x1::CoDual{P}, x2::CoDual{P}
) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    # Same coefficients as `rem`: exactly 1, and `-trunc(a/b)`, which is `Inf` once `b` is zero.
    c = trunc(a / b)
    rem_fast_pb(ȳ::P) = (NoRData(), ȳ, _rvs_guarded_scale(ȳ, -c))
    return zero_fcodual(Base.FastMath.rem_fast(a, b)), rem_fast_pb
end

# ---- `^` : removable-singularity limits at x == 0 ----
@is_primitive MinimalCtx Tuple{typeof(^),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(^),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = tangent(x1)^tangent(x2)
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(^)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    x = primal(x1)
    p = primal(x2)
    y = x^p
    # d/dx = p·y/x for x≠0; else the exponent-dependent removable limit (0/1/Inf).
    gx = ifelse(
        !iszero(x) || p < zero(P),
        p * y / x,
        ifelse(isone(p), one(y), ifelse(iszero(p) || p > one(P), zero(y), oftype(y, Inf))),
    )
    # d/dp = y·log(x) for x≠0; else 0 (p>0) or NaN (p≤0, genuinely undefined). The log must go
    # through `complex`: bare `log(x)` is a DomainError for negative `x`, where the real part is
    # the correct coefficient.
    gp = ifelse(
        !iszero(x), y * real(log(complex(x))), ifelse(p > zero(P), zero(y), oftype(y, NaN))
    )
    power_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, gx), _rvs_guarded_scale(ȳ, gp))
    return zero_fcodual(y), power_pb
end

# ---- `max` : subgradient (1,0)/(0,1) by which argument is selected (Base's tie convention) ----
@is_primitive MinimalCtx Tuple{typeof(max),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(max),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = max(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(max)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = max(a, b)
    pick = isequal(y, a) & !isequal(y, b)
    ga = ifelse(pick, one(a), zero(a))
    gb = ifelse(pick, zero(b), one(b))
    max_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, ga), _rvs_guarded_scale(ȳ, gb))
    return zero_fcodual(y), max_pb
end

# ---- `min` : subgradient (1,0)/(0,1) by which argument is selected (Base's tie convention) ----
@is_primitive MinimalCtx Tuple{typeof(min),P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(min),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = min(tangent(x1), tangent(x2))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(min)}, x1::CoDual{P}, x2::CoDual{P}) where {P<:IEEEFloat}
    a = primal(x1)
    b = primal(x2)
    y = min(a, b)
    pick = isequal(y, a) | !isequal(y, b)
    ga = ifelse(pick, one(a), zero(a))
    gb = ifelse(pick, zero(b), one(b))
    min_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, ga), _rvs_guarded_scale(ȳ, gb))
    return zero_fcodual(y), min_pb
end

# ---- FastMath.pow_fast(x, n::Integer): gradient wrt the float base ----
@is_primitive MinimalCtx Tuple{
    typeof(Base.FastMath.pow_fast),P,I
} where {P<:IEEEFloat,I<:Integer}
function frule!!(
    ::Lifted{typeof(Base.FastMath.pow_fast),N}, x::Lifted{P,N,NDual{P,N}}, n::Lifted{I,N}
) where {N,P<:IEEEFloat,I<:Integer}
    dy = Base.FastMath.pow_fast(tangent(x), primal(n))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.pow_fast)}, x::CoDual{P}, n::CoDual{I}
) where {P<:IEEEFloat,I<:Integer}
    _x = primal(x)
    p = P(primal(n))
    y = Base.FastMath.pow_fast(_x, primal(n))
    fy = float(y)
    gx = ifelse(
        !iszero(_x) || p < zero(P),
        p * fy / _x,
        ifelse(
            isone(p), one(fy), ifelse(iszero(p) || p > one(P), zero(fy), oftype(fy, Inf))
        ),
    )
    pow_fast_pb(dy::P) = (NoRData(), _rvs_guarded_scale(dy, gx), NoRData())
    return zero_fcodual(y), pow_fast_pb
end

# ---- clamp(a, lo, hi): subgradient selects the active argument ----
@is_primitive MinimalCtx Tuple{typeof(clamp),P,P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(clamp),N},
    x1::Lifted{P,N,NDual{P,N}},
    x2::Lifted{P,N,NDual{P,N}},
    x3::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    dy = clamp(tangent(x1), tangent(x2), tangent(x3))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(clamp)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
) where {P<:IEEEFloat}
    a = primal(x1)
    lo = primal(x2)
    hi = primal(x3)
    y = clamp(a, lo, hi)
    # Upper bound first, matching Base and the `NDual` method: crossed bounds return `hi`.
    above = a >= hi
    below = (a <= lo) & !above
    ga = ifelse(below | above, zero(P), one(P))
    glo = ifelse(below, one(P), zero(P))
    ghi = ifelse(above, one(P), zero(P))
    clamp_pb(ȳ::P) = (
        NoRData(),
        _rvs_guarded_scale(ȳ, ga),
        _rvs_guarded_scale(ȳ, glo),
        _rvs_guarded_scale(ȳ, ghi),
    )
    return zero_fcodual(y), clamp_pb
end

# ---- 2-tuple-output rules (sincos family) ----
@is_primitive MinimalCtx Tuple{typeof(sincos),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincos),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincos(tangent(x))
    return Lifted{Tuple{P,P},N}(map(d -> d.value, tv), tv)
end
function rrule!!(::CoDual{typeof(sincos)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    sincos_pb(ȳ) = (NoRData(), _rvs_guarded_scale(ȳ[1], c) + _rvs_guarded_scale(ȳ[2], -s))
    return zero_fcodual((s, c)), sincos_pb
end

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincosd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincosd(tangent(x))
    return Lifted{Tuple{P,P},N}(map(d -> d.value, tv), tv)
end
function rrule!!(::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    sincosd_pb(ȳ) = (
        NoRData(),
        _rvs_guarded_scale(ȳ[1], deg2rad(c)) + _rvs_guarded_scale(ȳ[2], -deg2rad(s)),
    )
    return zero_fcodual((s, c)), sincosd_pb
end

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincospi(tangent(x))
    return Lifted{Tuple{P,P},N}(map(d -> d.value, tv), tv)
end
function rrule!!(::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincospi(v)
    sincospi_pb(ȳ) = (
        NoRData(),
        _rvs_guarded_scale(ȳ[1], oftype(v, π) * c) +
        _rvs_guarded_scale(ȳ[2], -oftype(v, π) * s),
    )
    return zero_fcodual((s, c)), sincospi_pb
end

# ---- modf(x) = (frac, int): only the fractional part is differentiable ----
@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(modf),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    tv = modf(tangent(x))
    return Lifted{Tuple{P,P},N}(map(d -> d.value, tv), tv)
end
function rrule!!(::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = modf(primal(x))
    modf_pb(ȳ) = (NoRData(), _rvs_guarded_scale(ȳ[1], one(P)))
    return zero_fcodual(y), modf_pb
end

# significand/frexp scale by a power of two constant within a binade; rules avoid bitcasts.
# frexp's integer exponent has no derivative.
@is_primitive MinimalCtx Tuple{typeof(significand),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(significand),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = significand(tangent(x))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(significand)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    # At zero and non-finite values, use the same unit-scale convention as `frexp`.
    e = (iszero(_x) || !isfinite(_x)) ? 0 : -exponent(_x)
    significand_pb(ȳ::P) = (NoRData(), ldexp(ȳ, e))
    return zero_fcodual(significand(_x)), significand_pb
end

@is_primitive MinimalCtx Tuple{typeof(frexp),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(frexp),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dv, e = frexp(tangent(x))
    return Lifted{Tuple{P,Int},N}((dv.value, e), (dv, NoDual()))
end
function rrule!!(::CoDual{typeof(frexp)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = frexp(primal(x))
    frexp_pb(ȳ) = (NoRData(), ldexp(ȳ[1], -y[2]))
    return zero_fcodual(y), frexp_pb
end

# ---- tanpi(x) = tan(π·x); derivative π·(1 + tanpi(x)²) ----
@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(tanpi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = tanpi(tangent(x))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = tanpi(primal(x))
    tanpi_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, P(π) * (one(P) + y^2)))
    return zero_fcodual(y), tanpi_pb
end

# ---- eps: piecewise-constant (zero derivative) ----
@zero_derivative MinimalCtx Tuple{typeof(Base.eps),P} where {P<:IEEEFloat}

# ---- angle_fast is constant on real inputs ⇒ zero derivative ----
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

# ---- hypot(x, xs...): d/dxᵢ = xᵢ/h, masked to 0 at xᵢ == 0 (also handles the all-zero 0/0) ----
@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(hypot),N},
    x::Lifted{P,N,NDual{P,N}},
    xs::Vararg{Lifted{P,N,NDual{P,N}},M},
) where {N,P<:IEEEFloat,M}
    dy = hypot(tangent(x), tuple_map(tangent, xs)...)
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    xvals = (primal(x), tuple_map(primal, xs)...)
    h = hypot(xvals...)
    coeffs = map(xi -> iszero(xi) ? zero(P) : xi / h, xvals)
    hypot_pb(ȳ::P) = (NoRData(), map(c -> _rvs_guarded_scale(ȳ, c), coeffs)...)
    return zero_fcodual(h), hypot_pb
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:low_level_maths})
    test_cases = vcat(
        map([Float32, Float64]) do P
            cases = [
                (sqrt, P(0.5)),
                (cbrt, P(0.4)),
                (log, P(0.1)),
                (log10, P(0.1)),
                (log2, P(0.15)),
                (log1p, P(0.95)),
                (exp, P(1.1)),
                (exp2, P(1.12)),
                (exp10, P(0.55)),
                (expm1, P(-0.3)),
                (sin, P(1.1)),
                (cos, P(1.1)),
                (tan, P(0.5)),
                (sec, P(-0.4)),
                (csc, P(0.3)),
                (cot, P(0.1)),
                (sind, P(181.1)),
                (cosd, P(-181.3)),
                (tand, P(93.5)),
                (secd, P(33.5)),
                (cscd, P(-0.5)),
                (cotd, P(5.1)),
                (sinpi, P(13.2)),
                (cospi, P(-33.2)),
                (asin, P(0.77)),
                (acos, P(0.53)),
                (atan, P(0.77)),
                (asec, P(2.55)),
                (acsc, P(1.03)),
                (acot, P(101.5)),
                (asind, P(0.23)),
                (acosd, P(0.55)),
                (atand, P(1.45)),
                (asecd, P(1.1)),
                (acscd, P(1.33)),
                (acotd, P(0.99)),
                (sinh, P(-3.56)),
                (cosh, P(3.4)),
                (tanh, P(0.25)),
                (sech, P(0.11)),
                (csch, P(-0.77)),
                (coth, P(0.22)),
                (asinh, P(1.45)),
                (acosh, P(1.56)),
                (atanh, P(-0.44)),
                (asech, P(0.75)),
                (acsch, P(0.32)),
                (acoth, P(1.05)),
                (sinc, P(0.36)),
                (sincos, P(3.0)),
                (deg2rad, P(185.4)),
                (rad2deg, P(0.45)),
                (mod2pi, P(0.1)),
                (mod, P(7.5), P(2.3)),
                (mod, P(10.2), P(3.1)),
                # Avoid rem's jumps at integer ratios; negative inputs distinguish trunc/floor.
                (rem, P(7.5), P(2.3)),
                (rem, P(-7.5), P(2.3)),
                (rem, P(7.5), P(-2.3)),
                # `flipsign`'s second argument is not finite-differenced at zero, where it jumps;
                # the `-0.0` convention is asserted by the rule, not here.
                (flipsign, P(3.0), P(-2.0)),
                (flipsign, P(3.0), P(2.0)),
                (ldexp, P(1.5), 3),
                (ldexp, P(1.5), -3),
                # Away from exact powers of two, where both jump to the next binade and a central
                # difference straddles the discontinuity.
                (significand, P(0.7)),
                (significand, P(-3.3)),
                (frexp, P(0.7)),
                (frexp, P(-3.3)),
                (Base.FastMath.rem_fast, P(7.5), P(2.3)),
                (Base.FastMath.rem_fast, P(-7.5), P(2.3)),
                (^, P(4.0), P(5.0)),
                (atan, P(4.3), P(0.23)),
                (hypot, P(4.0), P(5.0)),
                (hypot, P(4.0), P(5.0), P(6.0)),
                (log, P(2.3), P(3.76)),
                (max, P(1.5), P(0.5)),
                (max, P(0.45), P(1.1)),
                (min, P(1.5), P(0.5)),
                (min, P(0.45), P(1.1)),
                (Base.eps, P(5.0)),
                (nextfloat, P(0.25)),
                (prevfloat, P(1.1)),
            ]
            return map(case -> (false, :stability_and_allocs, nothing, case...), cases)
        end...,
        vec(
            map(Iterators.product([Float16, Float32, Float64], 1:5)) do (P, i)
                x = (P(0), -P(0), P(Inf), -P(Inf), P(NaN))[i]
                return (
                    false,
                    :none,
                    (
                        oracle=(
                            value=significand(x), deriv=(fwd=P(1), rvs=(NoRData(), P(1)))
                        ),
                        output_tangent=P(1),
                    ),
                    significand,
                    CoDual(x, P(1)),
                )
            end,
        ),
        # Forward-only primitive; seed coefficients too, at short and longer Horner folds.
        map([Float32, Float64]) do P
            return [
                (
                    false,
                    :stability_and_allocs,
                    (mode=ForwardMode,),
                    evalpoly,
                    P(1.7),
                    (P(0.3), P(-1.2)),
                ),
                (
                    false,
                    :stability_and_allocs,
                    (mode=ForwardMode,),
                    evalpoly,
                    P(0.6),
                    ntuple(i -> P(i) / 3, 8),
                ),
            ]
        end...,
        # Pin nonzero seeds where a separately materialised power of two overflows or
        # underflows. Finite differences cannot resolve these scales or binade boundaries.
        map([Float16, Float32, Float64]) do P
            x = nextfloat(zero(P))
            large = ldexp(one(P), exponent(floatmax(P)) - 4)
            cases = [
                (ldexp, x, -exponent(x)),
                (ldexp, large, exponent(x) - 1),
                (significand, x),
                (frexp, x),
            ]
            map(cases) do (f, seed, args...)
                y = f(seed, args...)
                dy = f === frexp ? y[1] : y
                fwd = f === frexp ? (dy, NoTangent()) : dy
                rvs = (NoRData(), dy, map(_ -> NoRData(), args)...)
                output_tangent = f === frexp ? (seed, NoTangent()) : seed
                opts = (
                    oracle=(value=y, deriv=(fwd=fwd, rvs=rvs)),
                    output_tangent=output_tangent,
                )
                return (false, :none, opts, f, CoDual(seed, seed), args...)
            end
        end...,
        Any[
            let
                x = Float16(1)
                bx = BigFloat(x)
                y = Float16(tan(bx))
                dy = Float16(one(bx) + tan(bx)^2)
                (
                    false,
                    :none,
                    (
                        oracle=(value=y, deriv=(fwd=dy, rvs=(NoRData(), dy))),
                        output_tangent=x,
                    ),
                    tan,
                    CoDual(x, x),
                )
            end,
            (
                false,
                :none,
                (
                    oracle=(
                        value=asinh(Float16(1000)),
                        deriv=(fwd=Float16(0.001), rvs=(NoRData(), Float16(0.001))),
                    ),
                    output_tangent=Float16(1),
                ),
                asinh,
                CoDual(Float16(1000), Float16(1)),
            ),
            (
                false,
                :none,
                (
                    oracle=(
                        value=acosh(Float16(1000)),
                        deriv=(fwd=Float16(0.001), rvs=(NoRData(), Float16(0.001))),
                    ),
                    output_tangent=Float16(1),
                ),
                acosh,
                CoDual(Float16(1000), Float16(1)),
            ),
            let
                x = 1e-200
                d = Float64(inv(BigFloat(2) * BigFloat(x)))
                (
                    false,
                    :none,
                    (
                        oracle=(deriv=d, cmp=(a, b) -> isapprox(a, b; rtol=1e-14)),
                        mode=ForwardMode,
                    ),
                    atan,
                    CoDual(x, 1.0),
                    CoDual(x, 0.0),
                )
            end,
            let
                x = 1e-200
                d = Float64(inv(BigFloat(2) * BigFloat(x)))
                (
                    false,
                    :none,
                    (
                        oracle=(
                            deriv=(NoRData(), d, -d),
                            cmp=(a, b) ->
                                isequal(a[1], b[1]) &&
                                all(isapprox(a[i], b[i]; rtol=1e-14) for i in 2:3),
                        ),
                        output_tangent=1.0,
                        mode=ReverseMode,
                    ),
                    atan,
                    CoDual(x, 0.0),
                    CoDual(x, 0.0),
                )
            end,
            (false, :stability_and_allocs, nothing, tanpi, 0.1),
            (false, :stability_and_allocs, nothing, Base.FastMath.pow_fast, 2.0, 3),
            (false, :stability_and_allocs, nothing, clamp, 0.5, 0.0, 1.0),
            # Crossed bounds select hi. Inside hi < a < lo, FD detects crediting lo even
            # if both modes make that mistake; the function is locally smooth here.
            (false, :none, nothing, clamp, 0.5, 1.0, 0.0),
            (false, :stability_and_allocs, nothing, sincosd, 30.0),
            (false, :stability_and_allocs, nothing, sincospi, 0.25),
            (false, :stability_and_allocs, nothing, modf, 1.7),
        ],
        map([
            (tanpi, Float16(0.5)),
            (tanpi, Float32(0.5)),
            (tanpi, 0.5),
            (secd, Float16(90)),
            (secd, Float32(90)),
            (secd, 90.0),
            (sec, Float16(π / 2)),
        ]) do (f, x)
            z = zero(x)
            opts = (oracle=(deriv=(fwd=z, rvs=(NoRData(), z)),), output_tangent=z)
            return (false, :none, opts, f, CoDual(x, z))
        end,
        # At hypot's singular origin, FD cannot pin the zero-derivative convention.
        # Explicit seeds pin the ray; isequal distinguishes exact zero from denormals.
        vec(
            map(Iterators.product([Float16, Float32, Float64], 1:3)) do (P, arity)
                seeds = ntuple(_ -> CoDual(P(0), P(1)), arity)
                rvs = (NoRData(), ntuple(_ -> P(0), arity)...)
                opts = (oracle=(value=P(0), deriv=(fwd=P(0), rvs=rvs)), output_tangent=P(1))
                return (false, :none, opts, hypot, seeds...)
            end,
        ),
    )
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]
