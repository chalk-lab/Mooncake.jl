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

# Many scalar smooth rules are defined further down this file: their forward `frule!!`s run the
# `f(::NDual)` overloads from the `Nfwd` submodule, while their reverse `rrule!!`s are direct native
# analytic pullbacks (no NDual/Nfwd/ChainRules dependency).
@zero_derivative MinimalCtx Tuple{typeof(log),Int}

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
                (deg2rad, P(185.4)),
                (rad2deg, P(0.45)),
                (mod2pi, P(0.1)),
                (mod, P(7.5), P(2.3)),
                (mod, P(10.2), P(3.1)),
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
    )
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]

# ============================================================================
# Forward (NDual) + native-reverse rules for scalar/fixed-arity math primitives.
# (Consolidated here from the former src/rules/rules_via_nfwd.jl.)
# ============================================================================

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
    return isequal(v, a) & !isequal(v, b)
end
@inline function _pick_first_min(a, b)
    v = min(a, b)
    return isequal(v, a) | !isequal(v, b)
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
# Trigonometric and reciprocal-trig. (sin/cos/tan moved to the fused `_value_and_deriv` pilot below,
# so that value and derivative share a single `sincos` call in BOTH modes — see task 176.)
@inline _unary_deriv(::typeof(sec), x, y) = y * tan(x)
@inline _unary_deriv(::typeof(csc), x, y) = -y * cot(x)
@inline _unary_deriv(::typeof(cot), x, y) = -(one(y) + y^2)
# sinpi/cospi moved to the fused `_value_and_deriv` block below (shared `sincospi`).
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
# sind/cosd/tand moved to the fused `_value_and_deriv` block below (shared `sincosd`).
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
@inline _unary_deriv(::typeof(Base.FastMath.atan_fast), x, y) = _unary_deriv(atan, x, y)
# mod2pi has local slope 1 away from the 2π wrap, but NaN *at* the wrap (`x` a multiple of 2π),
# where it is discontinuous — matching the forward `mod2pi(::NDual)` overload (`_nfwd_mod2pi_grad`)
# and `main`'s reverse, and consistent with the `mod`/`^` discontinuity handling below. A constant
# slope 1 here would silently disagree with forward mode at the wrap. nextfloat/prevfloat are treated
# as identity-derivative maps (they shift by one ulp).
@inline _unary_deriv(::typeof(mod2pi), x, y) = ifelse(
    isinteger(x / oftype(x, 2π)), oftype(x, NaN), one(x)
)
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
    sec,
    csc,
    cot,
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

# ── Fused value+derivative rules (task 176: trig families with a shared sincos-type primitive) ───
# `_value_and_deriv(f, x) -> (y, g)` computes the primal value `y = f(x)` and the local derivative
# factor `g = f'(x)` from a SINGLE shared evaluation, so a fused primitive (`sincos`/`sincosd`/
# `sincospi`) runs once and both the forward frule (which scales the partials by `g`) and the reverse
# rrule (which contracts the cotangent with `g`) reuse it. This replaces, for these functions, the old
# split of a forward `f(::NDual)` overload (`$f(tangent(x))`) and a separate reverse `_unary_deriv`
# factor. Notably it also fuses REVERSE mode, which previously computed `f(x)` and `f'(x)` with
# separate transcendental calls (e.g. `sin(x)` for the value and `cos(x)` for the factor).
# Only the genuine-fusion families live here; functions whose reverse factor already reuses the value
# (exp, log, sqrt, …) stay in the generic loop above.
@inline function _value_and_deriv(::typeof(sin), x)
    s, c = sincos(x)
    return s, c
end
@inline function _value_and_deriv(::typeof(cos), x)
    s, c = sincos(x)
    return c, -s
end
@inline function _value_and_deriv(::typeof(tan), x)
    s, c = sincos(x)
    t = s / c
    return t, one(t) + t^2
end
# Degree trig: `sincosd` shares the argument reduction; each derivative gains the deg2rad (π/180) scale.
@inline function _value_and_deriv(::typeof(sind), x)
    s, c = sincosd(x)
    return s, deg2rad(c)
end
@inline function _value_and_deriv(::typeof(cosd), x)
    s, c = sincosd(x)
    return c, -deg2rad(s)
end
@inline function _value_and_deriv(::typeof(tand), x)
    s, c = sincosd(x)
    t = s / c
    return t, deg2rad(one(t) + t^2)
end
# π-scaled trig: `sincospi` shares the reduction; d(sinpi)/dx = π·cospi, d(cospi)/dx = -π·sinpi.
@inline function _value_and_deriv(::typeof(sinpi), x)
    s, c = sincospi(x)
    return s, oftype(x, π) * c
end
@inline function _value_and_deriv(::typeof(cospi), x)
    s, c = sincospi(x)
    return c, -oftype(x, π) * s
end
for f in (sin, cos, tan, sind, cosd, tand, sinpi, cospi)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        # Forward: one `sincos`, then scale the partials by the shared derivative factor. Standard and
        # self-contained — no dispatch through an `f(::NDual)` overload.
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            nd = tangent(x)
            y, g = _value_and_deriv($f, nd.value)
            return Lifted{P,N}(y, NDual{P,N}(y, _pt_scale(nd.partials, g)))
        end
        # Reverse: the same fused `_value_and_deriv`, contracted against the cotangent.
        function rrule!!(::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            y, g = _value_and_deriv($f, primal(x))
            pb!!(ȳ::P) = (NoRData(), _rev_contract(ȳ, g))
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
# fastmath variant: same derivative formula as `atan` (fastmath only affects the primal).
@inline _binary_deriv(::typeof(Base.FastMath.atan_fast), a, b, y) = _binary_deriv(
    atan, a, b, y
)
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
