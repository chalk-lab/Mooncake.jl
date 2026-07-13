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

# Forward (NDual) + native-reverse analytic rules for scalar/fixed-arity math primitives.

# Reverse-mode removable-singularity guard, native (no forward-mode/NDual dependency): a zero
# incoming cotangent must yield an exact zero contribution even where the local derivative is ±Inf
# (`0 * Inf` would be `NaN`). Mirrors the forward `_pt_guarded_scale` guard, applied to the cotangent.
@inline _rev_contract(ȳ::T, grad::T) where {T} = iszero(ȳ) ? zero(T) : ȳ * grad

# ---- unary scalar rules ----
@is_primitive MinimalCtx Tuple{typeof(exp),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(exp),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = exp(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(exp)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = exp(_x)
    exp_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y))
    return zero_fcodual(y), exp_pb
end

@is_primitive MinimalCtx Tuple{typeof(exp2),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(exp2),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = exp2(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(exp2)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = exp2(_x)
    exp2_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y * oftype(y, log(2))))
    return zero_fcodual(y), exp2_pb
end

@is_primitive MinimalCtx Tuple{typeof(exp10),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(exp10),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = exp10(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(exp10)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = exp10(_x)
    exp10_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y * oftype(y, log(10))))
    return zero_fcodual(y), exp10_pb
end

@is_primitive MinimalCtx Tuple{typeof(expm1),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(expm1),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = expm1(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(expm1)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = expm1(_x)
    expm1_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, exp(_x)))
    return zero_fcodual(y), expm1_pb
end

@is_primitive MinimalCtx Tuple{typeof(log),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(log),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = log(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(log)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = log(_x)
    log_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(_x)))
    return zero_fcodual(y), log_pb
end

@is_primitive MinimalCtx Tuple{typeof(log2),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(log2),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = log2(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(log2)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = log2(_x)
    log2_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(_x * oftype(_x, log(2)))))
    return zero_fcodual(y), log2_pb
end

@is_primitive MinimalCtx Tuple{typeof(log10),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(log10),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = log10(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(log10)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = log10(_x)
    log10_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(_x * oftype(_x, log(10)))))
    return zero_fcodual(y), log10_pb
end

@is_primitive MinimalCtx Tuple{typeof(log1p),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(log1p),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = log1p(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = log1p(_x)
    log1p_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(one(_x) + _x)))
    return zero_fcodual(y), log1p_pb
end

@is_primitive MinimalCtx Tuple{typeof(sqrt),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sqrt),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = sqrt(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = sqrt(_x)
    sqrt_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(2 * y)))
    return zero_fcodual(y), sqrt_pb
end

@is_primitive MinimalCtx Tuple{typeof(cbrt),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cbrt),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = cbrt(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cbrt(_x)
    cbrt_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(3 * y^2)))
    return zero_fcodual(y), cbrt_pb
end

@is_primitive MinimalCtx Tuple{typeof(sec),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sec),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = sec(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(sec)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = sec(_x)
    sec_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y * tan(_x)))
    return zero_fcodual(y), sec_pb
end

@is_primitive MinimalCtx Tuple{typeof(csc),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(csc),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = csc(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(csc)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = csc(_x)
    csc_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -y * cot(_x)))
    return zero_fcodual(y), csc_pb
end

@is_primitive MinimalCtx Tuple{typeof(cot),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cot),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = cot(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(cot)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cot(_x)
    cot_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -(one(y) + y^2)))
    return zero_fcodual(y), cot_pb
end

@is_primitive MinimalCtx Tuple{typeof(asin),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(asin),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = asin(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asin)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asin(_x)
    asin_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(sqrt(one(_x) - _x^2))))
    return zero_fcodual(y), asin_pb
end

@is_primitive MinimalCtx Tuple{typeof(acos),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(acos),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = acos(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acos)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acos(_x)
    acos_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(sqrt(one(_x) - _x^2))))
    return zero_fcodual(y), acos_pb
end

@is_primitive MinimalCtx Tuple{typeof(atan),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(atan),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = atan(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(atan)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = atan(_x)
    atan_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(one(_x) + _x^2)))
    return zero_fcodual(y), atan_pb
end

@is_primitive MinimalCtx Tuple{typeof(asec),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(asec),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = asec(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asec)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asec(_x)
    asec_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(abs(_x) * sqrt(_x^2 - one(_x)))))
    return zero_fcodual(y), asec_pb
end

@is_primitive MinimalCtx Tuple{typeof(acsc),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(acsc),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = acsc(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acsc)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acsc(_x)
    acsc_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(abs(_x) * sqrt(_x^2 - one(_x)))))
    return zero_fcodual(y), acsc_pb
end

@is_primitive MinimalCtx Tuple{typeof(acot),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(acot),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = acot(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acot)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acot(_x)
    acot_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(one(_x) + _x^2)))
    return zero_fcodual(y), acot_pb
end

@is_primitive MinimalCtx Tuple{typeof(sinh),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sinh),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = sinh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(sinh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = sinh(_x)
    sinh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, cosh(_x)))
    return zero_fcodual(y), sinh_pb
end

@is_primitive MinimalCtx Tuple{typeof(cosh),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cosh),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = cosh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(cosh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cosh(_x)
    cosh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, sinh(_x)))
    return zero_fcodual(y), cosh_pb
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
    tanh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, one(y) - y^2))
    return zero_fcodual(y), tanh_pb
end

@is_primitive MinimalCtx Tuple{typeof(sech),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sech),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = sech(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(sech)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = sech(_x)
    sech_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -tanh(_x) * y))
    return zero_fcodual(y), sech_pb
end

@is_primitive MinimalCtx Tuple{typeof(csch),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(csch),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = csch(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(csch)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = csch(_x)
    csch_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -coth(_x) * y))
    return zero_fcodual(y), csch_pb
end

@is_primitive MinimalCtx Tuple{typeof(coth),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(coth),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = coth(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(coth)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = coth(_x)
    coth_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -csch(_x)^2))
    return zero_fcodual(y), coth_pb
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
    asinh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(sqrt(_x^2 + one(_x)))))
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
    acosh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(sqrt(_x^2 - one(_x)))))
    return zero_fcodual(y), acosh_pb
end

@is_primitive MinimalCtx Tuple{typeof(atanh),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(atanh),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = atanh(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(atanh)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = atanh(_x)
    atanh_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(one(_x) - _x^2)))
    return zero_fcodual(y), atanh_pb
end

@is_primitive MinimalCtx Tuple{typeof(asech),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(asech),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = asech(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asech)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asech(_x)
    asech_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(_x * sqrt(one(_x) - _x^2))))
    return zero_fcodual(y), asech_pb
end

@is_primitive MinimalCtx Tuple{typeof(acsch),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acsch),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acsch(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acsch)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acsch(_x)
    acsch_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(abs(_x) * sqrt(one(_x) + _x^2))))
    return zero_fcodual(y), acsch_pb
end

@is_primitive MinimalCtx Tuple{typeof(acoth),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acoth),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acoth(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acoth)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acoth(_x)
    acoth_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(one(_x) - _x^2)))
    return zero_fcodual(y), acoth_pb
end

@is_primitive MinimalCtx Tuple{typeof(secd),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(secd),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = secd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(secd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = secd(_x)
    secd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, deg2rad(y * tand(_x))))
    return zero_fcodual(y), secd_pb
end

@is_primitive MinimalCtx Tuple{typeof(cscd),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cscd),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = cscd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(cscd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cscd(_x)
    cscd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -deg2rad(y * cotd(_x))))
    return zero_fcodual(y), cscd_pb
end

@is_primitive MinimalCtx Tuple{typeof(cotd),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cotd),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = cotd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(cotd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cotd(_x)
    cotd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -deg2rad(one(y) + y^2)))
    return zero_fcodual(y), cotd_pb
end

@is_primitive MinimalCtx Tuple{typeof(asind),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(asind),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = asind(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asind)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asind(_x)
    asind_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(deg2rad(sqrt(one(_x) - _x^2)))))
    return zero_fcodual(y), asind_pb
end

@is_primitive MinimalCtx Tuple{typeof(acosd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acosd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acosd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acosd(_x)
    acosd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(deg2rad(sqrt(one(_x) - _x^2)))))
    return zero_fcodual(y), acosd_pb
end

@is_primitive MinimalCtx Tuple{typeof(atand),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(atand),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = atand(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(atand)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = atand(_x)
    atand_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(deg2rad(one(_x) + _x^2))))
    return zero_fcodual(y), atand_pb
end

@is_primitive MinimalCtx Tuple{typeof(asecd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(asecd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = asecd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(asecd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = asecd(_x)
    asecd_pb(ȳ::P) = (
        NoRData(), _rev_contract(ȳ, inv(deg2rad(abs(_x) * sqrt(_x^2 - one(_x)))))
    )
    return zero_fcodual(y), asecd_pb
end

@is_primitive MinimalCtx Tuple{typeof(acscd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acscd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acscd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acscd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acscd(_x)
    acscd_pb(ȳ::P) = (
        NoRData(), _rev_contract(ȳ, -inv(deg2rad(abs(_x) * sqrt(_x^2 - one(_x)))))
    )
    return zero_fcodual(y), acscd_pb
end

@is_primitive MinimalCtx Tuple{typeof(acotd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(acotd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = acotd(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(acotd)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = acotd(_x)
    acotd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -inv(deg2rad(one(_x) + _x^2))))
    return zero_fcodual(y), acotd_pb
end

@is_primitive MinimalCtx Tuple{typeof(deg2rad),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(deg2rad),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = deg2rad(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(deg2rad)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = deg2rad(_x)
    deg2rad_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, deg2rad(one(_x))))
    return zero_fcodual(y), deg2rad_pb
end

@is_primitive MinimalCtx Tuple{typeof(rad2deg),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(rad2deg),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = rad2deg(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(rad2deg)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = rad2deg(_x)
    rad2deg_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, rad2deg(one(_x))))
    return zero_fcodual(y), rad2deg_pb
end

@is_primitive MinimalCtx Tuple{typeof(sinc),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sinc),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = sinc(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(sinc)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = sinc(_x)
    sinc_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, cosc(_x)))
    return zero_fcodual(y), sinc_pb
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
        _rev_contract(ȳ, ifelse(isinteger(_x / oftype(_x, 2π)), oftype(_x, NaN), one(_x))),
    )
    return zero_fcodual(y), mod2pi_pb
end

@is_primitive MinimalCtx Tuple{typeof(nextfloat),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(nextfloat),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = nextfloat(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(nextfloat)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = nextfloat(_x)
    nextfloat_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, one(_x)))
    return zero_fcodual(y), nextfloat_pb
end

@is_primitive MinimalCtx Tuple{typeof(prevfloat),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(prevfloat),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = prevfloat(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(::CoDual{typeof(prevfloat)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = prevfloat(_x)
    prevfloat_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, one(_x)))
    return zero_fcodual(y), prevfloat_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp_fast),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.exp_fast),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.exp_fast(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    _x = primal(x)
    y = Base.FastMath.exp_fast(_x)
    exp_fast_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y))
    return zero_fcodual(y), exp_fast_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp2_fast),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.exp2_fast),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.exp2_fast(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp2_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    _x = primal(x)
    y = Base.FastMath.exp2_fast(_x)
    exp2_fast_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y * oftype(y, log(2))))
    return zero_fcodual(y), exp2_fast_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp10_fast),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.exp10_fast),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.exp10_fast(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp10_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    _x = primal(x)
    y = Base.FastMath.exp10_fast(_x)
    exp10_fast_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, y * oftype(y, log(10))))
    return zero_fcodual(y), exp10_fast_pb
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.atan_fast),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.atan_fast),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = Base.FastMath.atan_fast(tangent(x))
    y = dy.value
    return Lifted{_typeof(y),N}(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.atan_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    _x = primal(x)
    y = Base.FastMath.atan_fast(_x)
    atan_fast_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, inv(one(_x) + _x^2)))
    return zero_fcodual(y), atan_fast_pb
end

# ---- fused trig (sin/cos/tan families): one shared `sincos`-type call for value + derivative ----
@is_primitive MinimalCtx Tuple{typeof(sin),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sin),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincos(v)
    y = s
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, c)))
end
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    y = s
    sin_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, c))
    return zero_fcodual(y), sin_pb
end

@is_primitive MinimalCtx Tuple{typeof(cos),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cos),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincos(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, -s)))
end
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    y = c
    cos_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -s))
    return zero_fcodual(y), cos_pb
end

@is_primitive MinimalCtx Tuple{typeof(tan),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(tan),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincos(v)
    t = s / c
    y = t
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, one(t) + t^2)))
end
function rrule!!(::CoDual{typeof(tan)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincos(v)
    t = s / c
    y = t
    tan_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, one(t) + t^2))
    return zero_fcodual(y), tan_pb
end

@is_primitive MinimalCtx Tuple{typeof(sind),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(sind),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincosd(v)
    y = s
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, deg2rad(c))))
end
function rrule!!(::CoDual{typeof(sind)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    y = s
    sind_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, deg2rad(c)))
    return zero_fcodual(y), sind_pb
end

@is_primitive MinimalCtx Tuple{typeof(cosd),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cosd),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincosd(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, -deg2rad(s))))
end
function rrule!!(::CoDual{typeof(cosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    y = c
    cosd_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -deg2rad(s)))
    return zero_fcodual(y), cosd_pb
end

@is_primitive MinimalCtx Tuple{typeof(tand),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(tand),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincosd(v)
    t = s / c
    y = t
    return Lifted{P,N}(
        y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, deg2rad(one(t) + t^2)))
    )
end
function rrule!!(::CoDual{typeof(tand)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincosd(v)
    t = s / c
    y = t
    tand_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, deg2rad(one(t) + t^2)))
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
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, oftype(v, π) * c)))
end
function rrule!!(::CoDual{typeof(sinpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincospi(v)
    y = s
    sinpi_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, oftype(v, π) * c))
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
    return Lifted{P,N}(y, NDual{P,N}(y, _pt_guarded_scale(nd.partials, -oftype(v, π) * s)))
end
function rrule!!(::CoDual{typeof(cospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = sincospi(v)
    y = c
    cospi_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, -oftype(v, π) * s))
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
    atan_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, b / r2), _rev_contract(ȳ, -a / r2))
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
    atan_fast_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, b / r2), _rev_contract(ȳ, -a / r2))
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
        NoRData(), _rev_contract(ȳ, -y / (a * lb)), _rev_contract(ȳ, inv(b * lb))
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
        _rev_contract(ȳ, ifelse(isint, nan, one(u))),
        _rev_contract(ȳ, ifelse(isint, nan, -floor(u))),
    )
    return zero_fcodual(y), mod_pb
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
    # d/dp = y·log(x) for x≠0; else 0 (p>0) or NaN (p≤0, genuinely undefined).
    gp = ifelse(
        !iszero(x), y * real(log(complex(x))), ifelse(p > zero(P), zero(y), oftype(y, NaN))
    )
    power_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, gx), _rev_contract(ȳ, gp))
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
    max_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, ga), _rev_contract(ȳ, gb))
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
    min_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, ga), _rev_contract(ȳ, gb))
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
    pow_fast_pb(dy::P) = (NoRData(), _rev_contract(dy, gx), NoRData())
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
    below = a <= lo
    above = (a >= hi) & !below
    ga = ifelse(below | above, zero(P), one(P))
    glo = ifelse(below, one(P), zero(P))
    ghi = ifelse(above, one(P), zero(P))
    clamp_pb(ȳ::P) = (
        NoRData(), _rev_contract(ȳ, ga), _rev_contract(ȳ, glo), _rev_contract(ȳ, ghi)
    )
    return zero_fcodual(y), clamp_pb
end

# ---- 2-tuple-output rules (sincos family) ----
@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.sincos),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(Base.FastMath.sincos),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = Base.FastMath.sincos(tangent(x))
    return Lifted{Tuple{P,P},N}(map(d -> d.value, tv), tv)
end
function rrule!!(::CoDual{typeof(Base.FastMath.sincos)}, x::CoDual{P}) where {P<:IEEEFloat}
    v = primal(x)
    s, c = Base.FastMath.sincos(v)
    sincos_pb(ȳ) = (NoRData(), _rev_contract(ȳ[1], c) + _rev_contract(ȳ[2], -s))
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
        NoRData(), _rev_contract(ȳ[1], deg2rad(c)) + _rev_contract(ȳ[2], -deg2rad(s))
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
        _rev_contract(ȳ[1], oftype(v, π) * c) + _rev_contract(ȳ[2], -oftype(v, π) * s),
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
    modf_pb(ȳ) = (NoRData(), _rev_contract(ȳ[1], one(P)))
    return zero_fcodual(y), modf_pb
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
    tanpi_pb(ȳ::P) = (NoRData(), _rev_contract(ȳ, P(π) * (one(P) + y^2)))
    return zero_fcodual(y), tanpi_pb
end

# ---- eps: piecewise-constant (zero derivative); emit a canonical zero-derivative dual ----
@is_primitive MinimalCtx Tuple{typeof(Base.eps),P} where {P<:IEEEFloat}
@inline function frule!!(
    ::Lifted{typeof(Base.eps),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    return zero_lifted(Val(N), eps(primal(x)))
end
function rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P}) where {P<:IEEEFloat}
    eps_pb(::P) = (NoRData(), zero(P))
    return zero_fcodual(eps(primal(x))), eps_pb
end

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
    hypot_pb(ȳ::P) = (NoRData(), map(c -> _rev_contract(ȳ, c), coeffs)...)
    return zero_fcodual(h), hypot_pb
end

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
