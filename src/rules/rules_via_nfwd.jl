#
# Primitive rules for scalar (and small fixed-arity) math functions.
#
# MinimalCtx is used throughout: several of these functions (e.g. tanpi, sincosd, sincospi)
# contain `try`/`catch` internally, which Mooncake's IR-transform AD cannot handle. Registering
# them as MinimalCtx primitives dispatches these rules directly, bypassing the failing transform.
#
# Forward (`frule!!`): the input `Lifted{P,N,NDual{P,N}}` already carries the N seeded directions,
# so the rule just runs the primal on `tangent(x)` (the inner `NDual`) — the `f(::NDual)` overloads
# in Nfwd.jl propagate the partials — and wraps the result. This is the one place these rules depend
# on the forward-mode `Nfwd` submodule.
#
# Reverse (`rrule!!`): DIRECT NATIVE analytic pullbacks — no `NDual` seeding, no `Nfwd`/ChainRules
# dependency. Closed-form derivative factors are dispatched through `_unary_deriv` / `_binary_deriv`
# (with `_pow_grad_x`/`_pow_grad_p` for the pow family and `_pick_first_max/min` for max/min), then
# contracted against the output cotangent via `_rev_contract`, which keeps an inactive (zero-cotangent)
# lane exactly zero even where the local derivative is ±Inf (removable-singularity NaN guard). So
# reverse mode here has zero dependence on forward mode.

# ── Forward-mode output helper ────────────────────────────────────────────────
# Output primal value from a dual result, for scalar (`NDual`) or tuple-of-`NDual` (e.g. `sincos`)
# results; used by the `frule!!`s to read the primal back out of the forward dual.
@inline _nfwd_out_value(yd::NDual) = yd.value
@inline _nfwd_out_value(yd::Tuple) = map(d -> d.value, yd)

# Reverse-mode removable-singularity guard, native (no forward-mode/NDual dependency): a zero
# incoming cotangent must yield an exact zero contribution even where the local derivative is ±Inf
# (`0 * Inf` would be `NaN`). Mirrors the forward `_pt_guarded_scale` guard, applied to the cotangent.
@inline _rev_contract(ȳ::T, grad::T) where {T} = iszero(ȳ) ? zero(T) : ȳ * grad

# Native pow gradients (pure scalar math, no NDual): d(x^p)/dx and d(x^p)/dp. The `x == 0` branches
# encode the removable-singularity limits (matching the forward path): d/dx is `p·y/x` for `x≠0`,
# else 0/1/Inf by exponent; d/dp is `y·log(x)` for `x≠0`, else 0 (p>0) or NaN (p≤0, genuinely
# undefined). `_rev_contract` then keeps an inactive (zero-cotangent) lane exactly zero.
@inline function _pow_grad_x(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x) || p < zero(P),
        p * y / x,
        ifelse(isone(p), one(y), ifelse(iszero(p) || p > one(P), zero(y), oftype(y, Inf))),
    )
end
@inline function _pow_grad_p(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x), y * real(log(complex(x))), ifelse(p > zero(P), zero(y), oftype(y, NaN))
    )
end

# max/min subgradient selection: `true` picks the first argument, matching Base's tie convention
# (second arg for max, first arg for min). Pure scalar, branchless.
@inline function _pick_first_max(a, b)
    v = max(a, b)
    am = isequal(v, a)
    bm = isequal(v, b)
    return ifelse(am & !bm, true, ifelse(bm & !am, false, false))
end
@inline function _pick_first_min(a, b)
    v = min(a, b)
    am = isequal(v, a)
    bm = isequal(v, b)
    return ifelse(am & !bm, true, ifelse(bm & !am, false, true))
end

# ===========================================================================
# nfwd-backed primitive rule registrations
# ===========================================================================

# ── native-reverse unary scalar rules (no NDual/Nfwd dependency) ───────────────
# Reverse mode uses closed-form derivative factors dispatched through `_unary_deriv` instead of
# seeding an `NDual` and running the forward primal. The factors mirror the `NDual` overloads in
# `Nfwd.jl` (which the forward frule still uses); `y === f(x)` is reused where the factor needs it.
# Removable singularities (`log`/`sqrt` have ±Inf factors at the domain edge) are handled by the
# `_rev_contract` cotangent guard, the reverse analogue of the forward `_pt_guarded_scale`.
@inline _unary_deriv(::typeof(exp), x, y) = y
@inline _unary_deriv(::typeof(exp2), x, y) = y * oftype(y, log(2))
@inline _unary_deriv(::typeof(exp10), x, y) = y * oftype(y, log(10))
@inline _unary_deriv(::typeof(expm1), x, y) = exp(x)
@inline _unary_deriv(::typeof(log), x, y) = inv(x)
@inline _unary_deriv(::typeof(log2), x, y) = inv(x * oftype(x, log(2)))
@inline _unary_deriv(::typeof(log10), x, y) = inv(x * oftype(x, log(10)))
@inline _unary_deriv(::typeof(log1p), x, y) = inv(one(x) + x)
@inline _unary_deriv(::typeof(sqrt), x, y) = inv(2 * y)
@inline _unary_deriv(::typeof(cbrt), x, y) = inv(3 * y^2)
# Trigonometric and reciprocal-trig.
@inline _unary_deriv(::typeof(sin), x, y) = cos(x)
@inline _unary_deriv(::typeof(cos), x, y) = -sin(x)
@inline _unary_deriv(::typeof(tan), x, y) = one(y) + y^2
@inline _unary_deriv(::typeof(sec), x, y) = y * tan(x)
@inline _unary_deriv(::typeof(csc), x, y) = -y * cot(x)
@inline _unary_deriv(::typeof(cot), x, y) = -(one(y) + y^2)
@inline _unary_deriv(::typeof(sinpi), x, y) = oftype(x, π) * cospi(x)
@inline _unary_deriv(::typeof(cospi), x, y) = -oftype(x, π) * sinpi(x)
# Inverse trig.
@inline _unary_deriv(::typeof(asin), x, y) = inv(sqrt(one(x) - x^2))
@inline _unary_deriv(::typeof(acos), x, y) = -inv(sqrt(one(x) - x^2))
@inline _unary_deriv(::typeof(atan), x, y) = inv(one(x) + x^2)
@inline _unary_deriv(::typeof(asec), x, y) = inv(abs(x) * sqrt(x^2 - one(x)))
@inline _unary_deriv(::typeof(acsc), x, y) = -inv(abs(x) * sqrt(x^2 - one(x)))
@inline _unary_deriv(::typeof(acot), x, y) = -inv(one(x) + x^2)
# Hyperbolic and reciprocal-hyperbolic.
@inline _unary_deriv(::typeof(sinh), x, y) = cosh(x)
@inline _unary_deriv(::typeof(cosh), x, y) = sinh(x)
@inline _unary_deriv(::typeof(tanh), x, y) = one(y) - y^2
@inline _unary_deriv(::typeof(sech), x, y) = -tanh(x) * y
@inline _unary_deriv(::typeof(csch), x, y) = -coth(x) * y
@inline _unary_deriv(::typeof(coth), x, y) = -csch(x)^2
# Inverse hyperbolic.
@inline _unary_deriv(::typeof(asinh), x, y) = inv(sqrt(x^2 + one(x)))
@inline _unary_deriv(::typeof(acosh), x, y) = inv(sqrt(x^2 - one(x)))
@inline _unary_deriv(::typeof(atanh), x, y) = inv(one(x) - x^2)
@inline _unary_deriv(::typeof(asech), x, y) = -inv(x * sqrt(one(x) - x^2))
@inline _unary_deriv(::typeof(acsch), x, y) = -inv(abs(x) * sqrt(one(x) + x^2))
@inline _unary_deriv(::typeof(acoth), x, y) = inv(one(x) - x^2)
# Degree-based trig — argument in degrees, so every factor gains the deg2rad (π/180) scale.
@inline _unary_deriv(::typeof(sind), x, y) = deg2rad(cosd(x))
@inline _unary_deriv(::typeof(cosd), x, y) = -deg2rad(sind(x))
@inline _unary_deriv(::typeof(tand), x, y) = deg2rad(one(y) + y^2)
@inline _unary_deriv(::typeof(secd), x, y) = deg2rad(y * tand(x))
@inline _unary_deriv(::typeof(cscd), x, y) = -deg2rad(y * cotd(x))
@inline _unary_deriv(::typeof(cotd), x, y) = -deg2rad(one(y) + y^2)
@inline _unary_deriv(::typeof(asind), x, y) = inv(deg2rad(sqrt(one(x) - x^2)))
@inline _unary_deriv(::typeof(acosd), x, y) = -inv(deg2rad(sqrt(one(x) - x^2)))
@inline _unary_deriv(::typeof(atand), x, y) = inv(deg2rad(one(x) + x^2))
@inline _unary_deriv(::typeof(asecd), x, y) = inv(deg2rad(abs(x) * sqrt(x^2 - one(x))))
@inline _unary_deriv(::typeof(acscd), x, y) = -inv(deg2rad(abs(x) * sqrt(x^2 - one(x))))
@inline _unary_deriv(::typeof(acotd), x, y) = -inv(deg2rad(one(x) + x^2))
# Angle-unit conversions (constant scale) and sinc.
@inline _unary_deriv(::typeof(deg2rad), x, y) = deg2rad(one(x))
@inline _unary_deriv(::typeof(rad2deg), x, y) = rad2deg(one(x))
@inline _unary_deriv(::typeof(sinc), x, y) = cosc(x)
# FastMath scalar variants — same derivative factors as their non-fast counterparts.
@inline _unary_deriv(::typeof(Base.FastMath.exp_fast), x, y) = y
@inline _unary_deriv(::typeof(Base.FastMath.exp2_fast), x, y) = y * oftype(y, log(2))
@inline _unary_deriv(::typeof(Base.FastMath.exp10_fast), x, y) = y * oftype(y, log(10))
@inline _unary_deriv(::typeof(Base.FastMath.atan_fast), x, y) = inv(one(x) + x^2)
# mod2pi has local slope 1 (away from the 2π wrap); nextfloat/prevfloat are treated as
# identity-derivative maps (they shift by one ulp).
@inline _unary_deriv(::typeof(mod2pi), x, y) = one(x)
@inline _unary_deriv(::typeof(nextfloat), x, y) = one(x)
@inline _unary_deriv(::typeof(prevfloat), x, y) = one(x)

for f in (
    exp,
    exp2,
    exp10,
    expm1,
    log,
    log2,
    log10,
    log1p,
    sqrt,
    cbrt,
    sin,
    cos,
    tan,
    sec,
    csc,
    cot,
    sinpi,
    cospi,
    asin,
    acos,
    atan,
    asec,
    acsc,
    acot,
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
    sind,
    cosd,
    tand,
    secd,
    cscd,
    cotd,
    asind,
    acosd,
    atand,
    asecd,
    acscd,
    acotd,
    deg2rad,
    rad2deg,
    sinc,
    mod2pi,
    nextfloat,
    prevfloat,
    Base.FastMath.exp_fast,
    Base.FastMath.exp2_fast,
    Base.FastMath.exp10_fast,
    Base.FastMath.atan_fast,
)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        # Forward: the `$f(::NDual)` overload in Nfwd.jl propagates partials and sets the result's
        # `.value` to `$f(primal(x))` (inner-value invariant); read the primal back from `dy`.
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            dy = $f(tangent(x))
            y = _nfwd_out_value(dy)
            return Lifted{_typeof(y),N}(y, dy)
        end
        # Reverse: native closed-form pullback — no NDual seeding.
        function rrule!!(::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            _x = primal(x)
            y = $f(_x)
            pb!!(ȳ::P) = (NoRData(), _rev_contract(ȳ, _unary_deriv($f, _x, y)))
            return zero_fcodual(y), pb!!
        end
    end
end

# ── nfwd-backed unary scalar rules ─────────────────────────────────────────────
# ── FastMath.sincos (tuple output) ────────────────────────────────────────────
@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.sincos),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.sincos),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = Base.FastMath.sincos(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
# Native reverse: sincos(x) = (sin(x), cos(x)); d(sin)/dx = cos(x), d(cos)/dx = -sin(x).
function rrule!!(::CoDual{typeof(Base.FastMath.sincos)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = Base.FastMath.sincos(primal(x))
    function fsincos_pb!!(ȳ)
        return NoRData(), _rev_contract(ȳ[1], c) + _rev_contract(ȳ[2], -s)
    end
    return zero_fcodual((s, c)), fsincos_pb!!
end

# `eps` is piecewise-constant (zero derivative). Unlike `nextfloat`/`prevfloat` it has no
# `NDual` overload, so the generic `dy = eps(tangent(x))` path above would return a bare
# `Float64`, giving a non-canonical `Lifted{Float64,N,Float64}` V. Emit a canonical
# zero-derivative `NDual` instead.
@is_primitive MinimalCtx Tuple{typeof(Base.eps),P} where {P<:IEEEFloat}
@inline function frule!!(
    ::Lifted{typeof(Base.eps),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    y = eps(primal(x))
    return Lifted{P,N}(y, NDual{P,N}(y, ntuple(_ -> zero(P), Val(N))))
end
function rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P}) where {P<:IEEEFloat}
    eps_pb!!(::P) = (NoRData(), zero(P))
    return zero_fcodual(eps(primal(x))), eps_pb!!
end

# ── tanpi ─────────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(tanpi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(tanpi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = tanpi(tangent(x))
    return Lifted{P,N}(dy.value, dy)
end
# Native reverse rule: tanpi(x) = tan(π·x); derivative = π·(1 + tanpi(x)²).
function rrule!!(::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = tanpi(primal(x))
    tanpi_pb!!(ȳ::P) = (NoRData(), _rev_contract(ȳ, P(π) * (one(P) + y^2)))
    return zero_fcodual(y), tanpi_pb!!
end

# ── native-reverse fixed-arity scalar rules ───────────────────────────────────
# Closed-form gradient pair per binary primitive, dispatched through `_binary_deriv(f, x1, x2, y)`.
@inline function _binary_deriv(::typeof(atan), a, b, y)
    r2 = a^2 + b^2
    return (b / r2, -a / r2)
end
@inline function _binary_deriv(::typeof(Base.FastMath.atan_fast), a, b, y)
    r2 = a^2 + b^2
    return (b / r2, -a / r2)
end
# log(b, a) = log(a)/log(b): d/db = -log(b,a)/(b·log(b)), d/da = 1/(a·log(b)).
@inline function _binary_deriv(::typeof(log), b, a, y)
    lb = log(b)
    return (-y / (b * lb), inv(a * lb))
end
# x^p: d/dx and d/dp share the native pow gradients (removable-singularity limits at x==0).
@inline _binary_deriv(::typeof(^), x, p, y) = (_pow_grad_x(x, p, y), _pow_grad_p(x, p, y))
# mod(x, y): d/dx = 1, d/dy = -floor(x/y); both NaN at the integer-quotient discontinuities.
@inline function _binary_deriv(::typeof(mod), x, y, out)
    u = x / y
    nan = oftype(u, NaN)
    isint = isinteger(u)
    return (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u)))
end
# max/min: subgradient (1,0) or (0,1) by which argument is selected.
@inline function _binary_deriv(::typeof(max), a, b, out)
    p = _pick_first_max(a, b)
    return (ifelse(p, one(a), zero(a)), ifelse(p, zero(b), one(b)))
end
@inline function _binary_deriv(::typeof(min), a, b, out)
    p = _pick_first_min(a, b)
    return (ifelse(p, one(a), zero(a)), ifelse(p, zero(b), one(b)))
end

for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        function frule!!(
            ::Lifted{typeof($f),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            dy = $f(tangent(x1), tangent(x2))
            return Lifted{P,N}(dy.value, dy)
        end
        # Native reverse: closed-form gradient pair, no NDual seeding.
        function rrule!!(
            ::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            a = primal(x1)
            b = primal(x2)
            y = $f(a, b)
            g1, g2 = _binary_deriv($f, a, b, y)
            pb!!(ȳ::P) = (NoRData(), _rev_contract(ȳ, g1), _rev_contract(ȳ, g2))
            return zero_fcodual(y), pb!!
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
    # The `NDual` overload sets `.value` to the primal result and scales the partials with
    # `_pt_guarded_scale`, so a zero (inactive) lane stays zero even where the gradient is
    # `±Inf` (e.g. `x == 0` with a negative exponent).
    dy = Base.FastMath.pow_fast(tangent(x), primal(n))
    return Lifted{P,N}(dy.value, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.pow_fast)}, x::CoDual{P}, n::CoDual{I}
) where {P<:IEEEFloat,I<:Integer}
    _x = primal(x)
    _n = primal(n)
    y = Base.FastMath.pow_fast(_x, _n)
    function pow_fast_pb!!(dy::P)
        return NoRData(), _rev_contract(dy, _pow_grad_x(_x, P(_n), float(y))), NoRData()
    end
    return zero_fcodual(y), pow_fast_pb!!
end

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
# Native reverse rule: clamp(a, lo, hi) = ifelse(a<=lo, lo, ifelse(a>=hi, hi, a)); the derivative is
# 1 for whichever argument is selected and 0 for the other two (subgradient at the endpoints).
function rrule!!(
    ::CoDual{typeof(clamp)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
) where {P<:IEEEFloat}
    a = primal(x1)
    lo = primal(x2)
    hi = primal(x3)
    y = clamp(a, lo, hi)
    below = a <= lo
    above = (a >= hi) & !below
    ga = ifelse(below | above, zero(P), one(P))
    glo = ifelse(below, one(P), zero(P))
    ghi = ifelse(above, one(P), zero(P))
    clamp_pb!!(ȳ::P) = (
        NoRData(), _rev_contract(ȳ, ga), _rev_contract(ȳ, glo), _rev_contract(ȳ, ghi)
    )
    return zero_fcodual(y), clamp_pb!!
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincosd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincosd(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
# Native reverse rule: sincosd(x) = (sind(x), cosd(x)); d(sind)/dx = deg2rad(cosd(x)),
# d(cosd)/dx = -deg2rad(sind(x)). The output is a 2-tuple, so the cotangent `ȳ` is a 2-tuple.
function rrule!!(::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincosd(primal(x))
    function sincosd_pb!!(ȳ)
        return NoRData(), _rev_contract(ȳ[1], deg2rad(c)) + _rev_contract(ȳ[2], -deg2rad(s))
    end
    return zero_fcodual((s, c)), sincosd_pb!!
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincospi(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
# Native reverse rule: sincospi(x) = (sinpi(x), cospi(x)); d(sinpi)/dx = π·cospi(x),
# d(cospi)/dx = -π·sinpi(x).
function rrule!!(::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincospi(primal(x))
    function sincospi_pb!!(ȳ)
        return NoRData(), _rev_contract(ȳ[1], P(π) * c) + _rev_contract(ȳ[2], -P(π) * s)
    end
    return zero_fcodual((s, c)), sincospi_pb!!
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(modf),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    tv = modf(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
# Native reverse rule: modf(x) = (frac, int) with frac = x - trunc(x); d(frac)/dx = 1,
# d(int)/dx = 0 (trunc is piecewise-constant).
function rrule!!(::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = modf(primal(x))
    modf_pb!!(ȳ) = (NoRData(), _rev_contract(ȳ[1], one(P)))
    return zero_fcodual(y), modf_pb!!
end

# ── angle_fast ──────────────────────────────────────────────────────────────────
# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

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
    # d hypot / dxᵢ = xᵢ / h, masked to 0 when xᵢ == 0 (also collapses the all-zero 0/0 = NaN case).
    coeffs = map(xi -> iszero(xi) ? zero(P) : xi / h, xvals)
    hypot_pb!!(ȳ::P) = (NoRData(), map(c -> _rev_contract(ȳ, c), coeffs)...)
    return zero_fcodual(h), hypot_pb!!
end

# Cases for the scalar primitives defined here that no other group's registry covers
# (`exp`/`log`/`sin`/.../`hypot` are in `Val{:low_level_maths}`). Driven from
# test/rules/low_level_maths.jl — the sibling scalar-math group — so they get the full battery
# without standing up a separate CI job. `tanpi` is kept away from its `0.5` singularity.
function hand_written_rule_test_cases(rng_ctor, ::Val{:rules_via_nfwd})
    (
        Any[
            (false, :stability_and_allocs, nothing, tanpi, 0.1),
            (false, :stability_and_allocs, nothing, Base.FastMath.pow_fast, 2.0, 3),
            (false, :stability_and_allocs, nothing, clamp, 0.5, 0.0, 1.0),
            (false, :stability_and_allocs, nothing, sincos, 1.0),
            (false, :stability_and_allocs, nothing, sincosd, 30.0),
            (false, :stability_and_allocs, nothing, sincospi, 0.25),
            (false, :stability_and_allocs, nothing, modf, 1.7),
        ],
        Any[],
    )
end
derived_rule_test_cases(rng_ctor, ::Val{:rules_via_nfwd}) = Any[], Any[]
