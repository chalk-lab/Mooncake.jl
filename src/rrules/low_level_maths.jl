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
# from DiffRules, so this file now imports them from ChainRules.jl.

@from_chainrules MinimalCtx Tuple{typeof(exp),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(exp2),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(exp10),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(expm1),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sin),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cos),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(tan),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sec),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(csc),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cot),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sind),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cosd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(tand),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(secd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cscd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cotd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sinpi),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asin),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acos),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(atan),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asec),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acsc),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acot),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asind),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acosd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(atand),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asecd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acscd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acotd),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sinh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(cosh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(tanh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sech),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(csch),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(coth),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asinh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acosh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(atanh),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(asech),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acsch),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(acoth),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(sinc),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(deg2rad),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(rad2deg),IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(^),P,P} where {P<:IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(atan),P,P} where {P<:IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(max),P,P} where {P<:IEEEFloat}
@from_chainrules MinimalCtx Tuple{typeof(min),P,P} where {P<:IEEEFloat}

@is_primitive MinimalCtx Tuple{typeof(mod2pi),IEEEFloat}
function frule!!(::Dual{typeof(mod2pi)}, x::Dual{P}) where {P<:IEEEFloat}
    t = ifelse(isinteger(primal(x) / P(2π)), P(NaN), one(P))
    return Dual(mod2pi(primal(x)), tangent(x) * t)
end
function rrule!!(::CoDual{typeof(mod2pi)}, x::CoDual{P}) where {P<:IEEEFloat}
    function mod2pi_adjoint(dy::P)
        return NoRData(), dy * ifelse(isinteger(primal(x) / P(2π)), P(NaN), one(P))
    end
    return zero_fcodual(mod2pi(primal(x))), mod2pi_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(log)}, b::Dual{P}, x::Dual{P}) where {P<:IEEEFloat}
    _b, db = extract(b)
    _x, dx = extract(x)
    y = log(_b, _x)
    log_b = log(_b)
    return Dual(y, -db * y / (log_b * _b) + dx * (inv(_x) / log_b))
end
function rrule!!(::CoDual{typeof(log)}, b::CoDual{P}, x::CoDual{P}) where {P<:IEEEFloat}
    y = log(primal(b), primal(x))
    function log_adjoint(dy::P)
        log_b = log(primal(b))
        return NoRData(),
        ifelse(iszero(primal(b)) || iszero(log_b), P(0), -dy * y / (log_b * primal(b))),
        ifelse(iszero(primal(x)) || iszero(log_b), P(0), dy / (primal(x) * log_b))
    end
    return zero_fcodual(y), log_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log),P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(log)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    y = log(_x)
    return Dual(y, dx / _x)
end
function rrule!!(::CoDual{typeof(log)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = log(primal(x))
    function log_adjoint(dy::P)
        return NoRData(), ifelse(iszero(primal(x)), P(0), dy / primal(x))
    end
    return zero_fcodual(y), log_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sqrt),IEEEFloat}
function frule!!(::Dual{typeof(sqrt)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    y = sqrt(_x)
    return Dual(y, dx / (2 * y))
end
function rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = sqrt(primal(x))
    function sqrt_adjoint(dy::P)
        return NoRData(), ifelse(iszero(primal(x)), P(0), dy / (2 * y))
    end
    return zero_fcodual(y), sqrt_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cbrt),IEEEFloat}
function frule!!(::Dual{typeof(cbrt)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    y = cbrt(_x)
    return Dual(y, dx / (3 * y^2))
end
function rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = cbrt(primal(x))
    function cbrt_adjoint(dy::P)
        return NoRData(), ifelse(iszero(y), P(0), dy / (3 * y^2))
    end
    return zero_fcodual(y), cbrt_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log10),P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(log10)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    return Dual(log10(_x), dx / (_x * log(P(10))))
end
function rrule!!(::CoDual{typeof(log10)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = log10(primal(x))
    function log10_adjoint(dy::P)
        return NoRData(), ifelse(iszero(primal(x)), P(0), dy / (primal(x) * log(P(10))))
    end
    return zero_fcodual(y), log10_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log2),P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(log2)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    return Dual(log2(_x), dx / (_x * log(P(2))))
end
function rrule!!(::CoDual{typeof(log2)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = log2(primal(x))
    function log2_adjoint(dy::P)
        return NoRData(), ifelse(iszero(primal(x)), P(0), dy / (primal(x) * log(P(2))))
    end
    return zero_fcodual(y), log2_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log1p),P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(log1p)}, x::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    return Dual(log1p(_x), dx / (1 + _x))
end
function rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = log1p(primal(x))
    function log1p_adjoint(dy::P)
        return NoRData(), ifelse(iszero(1 + primal(x)), P(0), dy / (1 + primal(x)))
    end
    return zero_fcodual(y), log1p_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cospi),IEEEFloat}
function frule!!(::Dual{typeof(cospi)}, x::Dual{P}) where {P<:IEEEFloat}
    s, c = sincospi(primal(x))
    return Dual(c, -tangent(x) * P(π) * s)
end
function rrule!!(::CoDual{typeof(cospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincospi(primal(x))
    cospi_adjoint(dy::P) = NoRData(), -dy * P(π) * s
    return zero_fcodual(c), cospi_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(hypot)}, x::Dual{P}, xs::Vararg{Dual{P}}
) where {P<:IEEEFloat}
    h = hypot(primal(x), map(primal, xs)...)
    dh = sum(primal(a) * tangent(a) for a in (x, xs...)) / h
    return Dual(h, dh)
end
function rrule!!(
    ::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P}}
) where {P<:IEEEFloat}
    h = hypot(primal(x), map(primal, xs)...)
    function hypot_pb!!(dh::P)
        grads = map(a -> ifelse(iszero(h), P(0), dh * (primal(a) / h)), (x, xs...))
        return NoRData(), grads...
    end
    return zero_fcodual(h), hypot_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.eps),<:IEEEFloat}
function frule!!(::Dual{typeof(Base.eps)}, x::Dual{<:IEEEFloat})
    return Dual(eps(primal(x)), zero(primal(x)))
end
function rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = Base.eps(primal(x))
    eps_pb!!(dy::P) = NoRData(), zero(y)
    return zero_fcodual(y), eps_pb!!
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
                (deg2rad, P(185.4)),
                (rad2deg, P(0.45)),
                (mod2pi, P(0.1)),
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
            ]
            return map(case -> (false, :stability_and_allocs, nothing, case...), cases)
        end...,
    )
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]
