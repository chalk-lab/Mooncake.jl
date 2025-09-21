# Credit for many of these rules belongs with the authors, and various contributors to,
# DiffRules.jl -- https://github.com/JuliaDiff/DiffRules.jl .
# Mooncake originally incorporated the implementation found there directly. However,
# there were a number of issues with this:
# 1. Package extensions: it was written long before package extensions were added to Julia.
#   As a result, a couple of packages are direct dependencies of DiffRules, notably
#   SpecialFunctions.jl, which we do not wish to make indirect dependencies of Mooncake.jl.
#   All in all, by removing DiffRules as a dependency, we also remove: DocStringExtensions,
#   JLLWrappers, LogExpFunctions, NaNMath, OpenSpecFun_jll, OpenLibm_jll.
# 2. Interaction with Revise.jl: most modern development workflows involve using Revise.jl.
#   Unfortunately, putting `@eval` statements in a loop does not seem to play nicely with
#   it, meaning that every time you want to tweak something in the loop, you have to restart
#   your session.
# 3. Errors in the eval loop can cause spooky action-at-a-distance errors, which are hard to
#   debug.
# 4. Some of the rules in DiffRules are not implemented in an optimal manner, and it is
#   unclear that they _could_ be implemented in an optimal manner. For example, the rules
#   for `sin` and `cos` are unable to make use of the `sincos` function (which computes both
#   `sin` and `cos` at the same time at negligible additional cost to computing either `sin`
#   or `cos` by itself), and are therefore unable to provide optimal performance.
#
# The advantage of using DiffRules.jl was that it removed the need to write out the rules
# below. While this saved some effort in the early stages of development in Mooncake.jl,
# the downsides of continuing to import rules from DiffRules now outweigh the upsides, hence
# they are written out below.

@is_primitive MinimalCtx Tuple{typeof(sqrt),IEEEFloat}
function frule!!(::Dual{typeof(sqrt)}, x::Dual{<:IEEEFloat})
    y = sqrt(primal(x))
    return Dual(y, tangent(x) / 2y)
end
function rrule!!(::CoDual{typeof(sqrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = sqrt(primal(x))
    sqrt_adjoint(dy::P) = NoRData(), dy / (2 * y)
    return zero_fcodual(y), sqrt_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cbrt),IEEEFloat}
function frule!!(::Dual{typeof(cbrt)}, x::Dual{<:IEEEFloat})
    y = cbrt(primal(x))
    return Dual(y, tangent(x) / (3 * y^2))
end
function rrule!!(::CoDual{typeof(cbrt)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = cbrt(primal(x))
    cbrt_adjoint(dy::P) = NoRData(), dy / (3 * y^2)
    return zero_fcodual(y), cbrt_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log),IEEEFloat}
function frule!!(::Dual{typeof(log)}, x::Dual{<:IEEEFloat})
    return Dual(log(primal(x)), tangent(x) / primal(x))
end
function rrule!!(::CoDual{typeof(log)}, x::CoDual{P}) where {P<:IEEEFloat}
    log_adjoint(dy::P) = NoRData(), dy / primal(x)
    return zero_fcodual(log(primal(x))), log_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log10),IEEEFloat}
function frule!!(::Dual{typeof(log10)}, x::Dual{P}) where {P<:IEEEFloat}
    return Dual(log10(primal(x)), tangent(x) / (primal(x) * log(P(10))))
end
function rrule!!(::CoDual{typeof(log10)}, x::CoDual{P}) where {P<:IEEEFloat}
    log10_adjoint(dy::P) = NoRData(), dy / (primal(x) * log(P(10)))
    return zero_fcodual(log10(primal(x))), log10_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log2),IEEEFloat}
function frule!!(::Dual{typeof(log2)}, x::Dual{P}) where {P<:IEEEFloat}
    return Dual(log2(primal(x)), tangent(x) / (primal(x) * log(P(2))))
end
function rrule!!(::CoDual{typeof(log2)}, x::CoDual{P}) where {P<:IEEEFloat}
    log2_adjoint(dy::P) = NoRData(), dy / (primal(x) * log(P(2)))
    return zero_fcodual(log2(primal(x))), log2_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(log1p),IEEEFloat}
function frule!!(::Dual{typeof(log1p)}, x::Dual{<:IEEEFloat})
    return Dual(log1p(primal(x)), tangent(x) / (primal(x) + 1))
end
function rrule!!(::CoDual{typeof(log1p)}, x::CoDual{P}) where {P<:IEEEFloat}
    log1p_adjoint(dy::P) = NoRData(), dy / (primal(x) + 1)
    return zero_fcodual(log1p(primal(x))), log1p_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(exp),IEEEFloat}
function frule!!(::Dual{typeof(exp)}, x::Dual{P}) where {P<:IEEEFloat}
    y = exp(primal(x))
    return Dual(y, y * tangent(x))
end
function rrule!!(::CoDual{typeof(exp)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = exp(primal(x))
    exp_adjoint(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), exp_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(exp2),IEEEFloat}
function frule!!(::Dual{typeof(exp2)}, x::Dual{P}) where {P<:IEEEFloat}
    y = exp2(primal(x))
    return Dual(y, y * tangent(x) * P(log(2)))
end
function rrule!!(::CoDual{typeof(exp2)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = exp2(primal(x))
    exp2_adjoint(dy::P) = NoRData(), dy * y * P(log(2))
    return zero_fcodual(y), exp2_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(exp10),IEEEFloat}
function frule!!(::Dual{typeof(exp10)}, x::Dual{P}) where {P<:IEEEFloat}
    y = exp10(primal(x))
    return Dual(y, y * tangent(x) * P(log(10)))
end
function rrule!!(::CoDual{typeof(exp10)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = exp10(primal(x))
    exp10_adjoint(dy::P) = NoRData(), dy * y * P(log(10))
    return zero_fcodual(y), exp10_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(expm1),IEEEFloat}
function frule!!(::Dual{typeof(expm1)}, x::Dual{P}) where {P<:IEEEFloat}
    y = expm1(primal(x))
    return Dual(y, (y + P(1)) * tangent(x))
end
function rrule!!(::CoDual{typeof(expm1)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = expm1(primal(x))
    expm1_adjoint(dy::P) = NoRData(), dy * (y + P(1))
    return zero_fcodual(y), expm1_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sin),IEEEFloat}
function frule!!(::Dual{typeof(sin)}, x::Dual{<:IEEEFloat})
    s, c = sincos(primal(x))
    return Dual(s, c * tangent(x))
end
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    sin_pullback!!(dy::P) = NoRData(), dy * c
    return zero_fcodual(s), sin_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(cos),IEEEFloat}
function frule!!(::Dual{typeof(cos)}, x::Dual{<:IEEEFloat})
    s, c = sincos(primal(x))
    return Dual(c, -s * tangent(x))
end
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    cos_pullback!!(dy::P) = NoRData(), -dy * s
    return zero_fcodual(c), cos_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(tan),IEEEFloat}
function frule!!(::Dual{typeof(tan)}, x::Dual{<:IEEEFloat})
    y = tan(primal(x))
    return Dual(y, (1 + y^2) * tangent(x))
end
function rrule!!(::CoDual{typeof(tan)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = tan(primal(x))
    tan_adjoint(dy::P) = NoRData(), dy * (1 + y^2)
    return zero_fcodual(y), tan_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sec),IEEEFloat}
function frule!!(::Dual{typeof(sec)}, x::Dual{<:IEEEFloat})
    y = sec(primal(x))
    return Dual(y, tangent(x) * y * tan(primal(x)))
end
function rrule!!(::CoDual{typeof(sec)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = sec(primal(x))
    sec_adjoint(dy::P) = NoRData(), dy * y * tan(primal(x))
    return zero_fcodual(y), sec_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(csc),IEEEFloat}
function frule!!(::Dual{typeof(csc)}, x::Dual{<:IEEEFloat})
    y = csc(primal(x))
    return Dual(y, -tangent(x) * y * cot(primal(x)))
end
function rrule!!(::CoDual{typeof(csc)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = csc(primal(x))
    csc_adjoint(dy::P) = NoRData(), -dy * y * cot(primal(x))
    return zero_fcodual(y), csc_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cot),IEEEFloat}
function frule!!(::Dual{typeof(cot)}, x::Dual{<:IEEEFloat})
    y = cot(primal(x))
    return Dual(y, -tangent(x) * (1 + y^2))
end
function rrule!!(::CoDual{typeof(cot)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = cot(primal(x))
    cot_adjoint(dy::P) = NoRData(), -dy * (1 + y^2)
    return zero_fcodual(y), cot_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sind),IEEEFloat}
function frule!!(::Dual{typeof(sind)}, x::Dual{<:IEEEFloat})
    s, c = sincosd(primal(x))
    return Dual(s, tangent(x) * deg2rad(c))
end
function rrule!!(::CoDual{typeof(sind)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincosd(primal(x))
    sind_adjoint(dy::P) = NoRData(), dy * deg2rad(c)
    return zero_fcodual(s), sind_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cosd),IEEEFloat}
function frule!!(::Dual{typeof(cosd)}, x::Dual{<:IEEEFloat})
    s, c = sincosd(primal(x))
    return Dual(c, -tangent(x) * deg2rad(s))
end
function rrule!!(::CoDual{typeof(cosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincosd(primal(x))
    cosd_adjoint(dy::P) = NoRData(), -dy * deg2rad(s)
    return zero_fcodual(c), cosd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(tand),IEEEFloat}
function frule!!(::Dual{typeof(tand)}, x::Dual{<:IEEEFloat})
    y = tand(primal(x))
    return Dual(y, tangent(x) * deg2rad(1 + y^2))
end
function rrule!!(::CoDual{typeof(tand)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = tand(primal(x))
    tand_adjoint(dy::P) = NoRData(), dy * deg2rad(1 + y^2)
    return zero_fcodual(y), tand_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(secd),IEEEFloat}
function frule!!(::Dual{typeof(secd)}, x::Dual{<:IEEEFloat})
    y = secd(primal(x))
    return Dual(y, tangent(x) * deg2rad(y) * tand(primal(x)))
end
function rrule!!(::CoDual{typeof(secd)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = secd(primal(x))
    secd_adjoint(dy::P) = NoRData(), dy * deg2rad(y) * tand(primal(x))
    return zero_fcodual(y), secd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cscd),IEEEFloat}
function frule!!(::Dual{typeof(cscd)}, x::Dual{<:IEEEFloat})
    y = cscd(primal(x))
    return Dual(y, -tangent(x) * deg2rad(y) * cotd(primal(x)))
end
function rrule!!(::CoDual{typeof(cscd)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = cscd(primal(x))
    cscd_adjoint(dy::P) = NoRData(), -dy * deg2rad(y) * cotd(primal(x))
    return zero_fcodual(y), cscd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cotd),IEEEFloat}
function frule!!(::Dual{typeof(cotd)}, x::Dual{<:IEEEFloat})
    y = cotd(primal(x))
    return Dual(y, -tangent(x) * deg2rad(1 + y^2))
end
function rrule!!(::CoDual{typeof(cotd)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = cotd(primal(x))
    cotd_adjoint(dy::P) = NoRData(), -dy * deg2rad(1 + y^2)
    return zero_fcodual(y), cotd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sinpi),IEEEFloat}
function frule!!(::Dual{typeof(sinpi)}, x::Dual{P}) where {P<:IEEEFloat}
    s, c = sincospi(primal(x))
    return Dual(s, tangent(x) * P(π) * c)
end
function rrule!!(::CoDual{typeof(sinpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    s, c = sincospi(primal(x))
    sinpi_adjoint(dy::P) = NoRData(), dy * P(π) * c
    return zero_fcodual(s), sinpi_adjoint
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

@is_primitive MinimalCtx Tuple{typeof(asin),IEEEFloat}
function frule!!(::Dual{typeof(asin)}, x::Dual{<:IEEEFloat})
    return Dual(asin(primal(x)), tangent(x) * inv(sqrt(1 - primal(x)^2)))
end
function rrule!!(::CoDual{typeof(asin)}, x::CoDual{P}) where {P<:IEEEFloat}
    asin_adjoint(dy::P) = NoRData(), dy * inv(sqrt(1 - primal(x)^2))
    return zero_fcodual(asin(primal(x))), asin_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acos),IEEEFloat}
function frule!!(::Dual{typeof(acos)}, x::Dual{<:IEEEFloat})
    return Dual(acos(primal(x)), -tangent(x) * inv(sqrt(1 - primal(x)^2)))
end
function rrule!!(::CoDual{typeof(acos)}, x::CoDual{P}) where {P<:IEEEFloat}
    acos_adjoint(dy::P) = NoRData(), -dy * inv(sqrt(1 - primal(x)^2))
    return zero_fcodual(acos(primal(x))), acos_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(atan),IEEEFloat}
function frule!!(::Dual{typeof(atan)}, x::Dual{<:IEEEFloat})
    return Dual(atan(primal(x)), tangent(x) / (1 + primal(x)^2))
end
function rrule!!(::CoDual{typeof(atan)}, x::CoDual{P}) where {P<:IEEEFloat}
    atan_adjoint(dy::P) = NoRData(), dy / (1 + primal(x)^2)
    return zero_fcodual(atan(primal(x))), atan_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(asec),IEEEFloat}
function frule!!(::Dual{typeof(asec)}, x::Dual{<:IEEEFloat})
    return Dual(asec(primal(x)), tangent(x) / (abs(primal(x)) * sqrt(primal(x)^2 - 1)))
end
function rrule!!(::CoDual{typeof(asec)}, x::CoDual{P}) where {P<:IEEEFloat}
    asec_adjoint(dy::P) = NoRData(), dy / (abs(primal(x)) * sqrt(primal(x)^2 - 1))
    return zero_fcodual(asec(primal(x))), asec_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acsc),IEEEFloat}
function frule!!(::Dual{typeof(acsc)}, x::Dual{<:IEEEFloat})
    return Dual(acsc(primal(x)), -tangent(x) / (abs(primal(x)) * sqrt(primal(x)^2 - 1)))
end
function rrule!!(::CoDual{typeof(acsc)}, x::CoDual{P}) where {P<:IEEEFloat}
    acsc_adjoint(dy::P) = NoRData(), -dy / (abs(primal(x)) * sqrt(primal(x)^2 - 1))
    return zero_fcodual(acsc(primal(x))), acsc_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acot),IEEEFloat}
function frule!!(::Dual{typeof(acot)}, x::Dual{<:IEEEFloat})
    return Dual(acot(primal(x)), -tangent(x) / (1 + primal(x)^2))
end
function rrule!!(::CoDual{typeof(acot)}, x::CoDual{P}) where {P<:IEEEFloat}
    acot_adjoint(dy::P) = NoRData(), -dy / (1 + primal(x)^2)
    return zero_fcodual(acot(primal(x))), acot_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(asind),IEEEFloat}
function frule!!(::Dual{typeof(asind)}, x::Dual{<:IEEEFloat})
    return Dual(asind(primal(x)), tangent(x) / deg2rad(sqrt(1 - primal(x)^2)))
end
function rrule!!(::CoDual{typeof(asind)}, x::CoDual{P}) where {P<:IEEEFloat}
    asind_adjoint(dy::P) = NoRData(), dy / deg2rad(sqrt(1 - primal(x)^2))
    return zero_fcodual(asind(primal(x))), asind_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acosd),IEEEFloat}
function frule!!(::Dual{typeof(acosd)}, x::Dual{<:IEEEFloat})
    return Dual(acosd(primal(x)), -tangent(x) / deg2rad(sqrt(1 - primal(x)^2)))
end
function rrule!!(::CoDual{typeof(acosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    acosd_adjoint(dy::P) = NoRData(), -dy / deg2rad(sqrt(1 - primal(x)^2))
    return zero_fcodual(acosd(primal(x))), acosd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(atand),IEEEFloat}
function frule!!(::Dual{typeof(atand)}, x::Dual{<:IEEEFloat})
    return Dual(atand(primal(x)), tangent(x) / deg2rad(1 + primal(x)^2))
end
function rrule!!(::CoDual{typeof(atand)}, x::CoDual{P}) where {P<:IEEEFloat}
    atand_adjoint(dy::P) = NoRData(), dy / deg2rad(1 + primal(x)^2)
    return zero_fcodual(atand(primal(x))), atand_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(asecd),IEEEFloat}
function frule!!(::Dual{typeof(asecd)}, x::Dual{<:IEEEFloat})
    _x, dx = extract(x)
    return Dual(asecd(_x), dx / deg2rad(abs(_x) * sqrt(_x^2 - 1)))
end
function rrule!!(::CoDual{typeof(asecd)}, x::CoDual{P}) where {P<:IEEEFloat}
    asecd_adjoint(dy::P) = NoRData(), dy / deg2rad(abs(primal(x)) * sqrt(primal(x)^2 - 1))
    return zero_fcodual(asecd(primal(x))), asecd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acscd),IEEEFloat}
function frule!!(::Dual{typeof(acscd)}, x::Dual{<:IEEEFloat})
    _x, dx = extract(x)
    return Dual(acscd(_x), -dx / deg2rad(abs(_x) * sqrt(_x^2 - 1)))
end
function rrule!!(::CoDual{typeof(acscd)}, x::CoDual{P}) where {P<:IEEEFloat}
    acscd_adjoint(dy::P) = NoRData(), -dy / deg2rad(abs(primal(x)) * sqrt(primal(x)^2 - 1))
    return zero_fcodual(acscd(primal(x))), acscd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acotd),IEEEFloat}
function frule!!(::Dual{typeof(acotd)}, x::Dual{<:IEEEFloat})
    return Dual(acotd(primal(x)), -tangent(x) / deg2rad(1 + primal(x)^2))
end
function rrule!!(::CoDual{typeof(acotd)}, x::CoDual{P}) where {P<:IEEEFloat}
    acotd_adjoint(dy::P) = NoRData(), -dy / deg2rad(1 + primal(x)^2)
    return zero_fcodual(acotd(primal(x))), acotd_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sinh),IEEEFloat}
function frule!!(::Dual{typeof(sinh)}, x::Dual{<:IEEEFloat})
    return Dual(sinh(primal(x)), tangent(x) * cosh(primal(x)))
end
function rrule!!(::CoDual{typeof(sinh)}, x::CoDual{P}) where {P<:IEEEFloat}
    asinh_adjoint(dy::P) = NoRData(), dy * cosh(primal(x))
    return zero_fcodual(sinh(primal(x))), asinh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(cosh),IEEEFloat}
function frule!!(::Dual{typeof(cosh)}, x::Dual{<:IEEEFloat})
    return Dual(cosh(primal(x)), tangent(x) * sinh(primal(x)))
end
function rrule!!(::CoDual{typeof(cosh)}, x::CoDual{P}) where {P<:IEEEFloat}
    cosh_adjoint(dy::P) = NoRData(), dy * sinh(primal(x))
    return zero_fcodual(cosh(primal(x))), cosh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(tanh),IEEEFloat}
function frule!!(::Dual{typeof(tanh)}, x::Dual{<:IEEEFloat})
    y = tanh(primal(x))
    return Dual(y, tangent(x) * (1 - y^2))
end
function rrule!!(::CoDual{typeof(tanh)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = tanh(primal(x))
    tanh_adjoint(dy::P) = NoRData(), dy * (1 - y^2)
    return zero_fcodual(y), tanh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sech),IEEEFloat}
function frule!!(::Dual{typeof(sech)}, x::Dual{<:IEEEFloat})
    y = sech(primal(x))
    return Dual(y, -tangent(x) * tanh(primal(x) * y))
end
function rrule!!(::CoDual{typeof(sech)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = sech(primal(x))
    sech_adjoint(dy::P) = NoRData(), -dy * tanh(primal(x) * y)
    return zero_fcodual(y), sech_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(csch),IEEEFloat}
function frule!!(::Dual{typeof(csch)}, x::Dual{<:IEEEFloat})
    y = csch(primal(x))
    return Dual(y, -tangent(x) * coth(primal(x)) * y)
end
function rrule!!(::CoDual{typeof(csch)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = csch(primal(x))
    csch_adjoint(dy::P) = NoRData(), -dy * coth(primal(x)) * y
    return zero_fcodual(y), csch_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(coth),IEEEFloat}
function frule!!(::Dual{typeof(coth)}, x::Dual{<:IEEEFloat})
    return Dual(coth(primal(x)), -tangent(x) * csch(primal(x))^2)
end
function rrule!!(::CoDual{typeof(coth)}, x::CoDual{P}) where {P<:IEEEFloat}
    coth_adjoint(dy::P) = NoRData(), -dy * csch(primal(x))^2
    return zero_fcodual(coth(primal(x))), coth_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(asinh),IEEEFloat}
function frule!!(::Dual{typeof(asinh)}, x::Dual{<:IEEEFloat})
    return Dual(asinh(primal(x)), tangent(x) / sqrt(primal(x)^2 + 1))
end
function rrule!!(::CoDual{typeof(asinh)}, x::CoDual{P}) where {P<:IEEEFloat}
    asinh_adjoint(dy::P) = NoRData(), dy / sqrt(primal(x)^2 + 1)
    return zero_fcodual(asinh(primal(x))), asinh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acosh),IEEEFloat}
function frule!!(::Dual{typeof(acosh)}, x::Dual{<:IEEEFloat})
    return Dual(acosh(primal(x)), tangent(x) / sqrt(primal(x)^2 - 1))
end
function rrule!!(::CoDual{typeof(acosh)}, x::CoDual{P}) where {P<:IEEEFloat}
    acosh_adjoint(dy::P) = NoRData(), dy / sqrt(primal(x)^2 - 1)
    return zero_fcodual(acosh(primal(x))), acosh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(atanh),IEEEFloat}
function frule!!(::Dual{typeof(atanh)}, x::Dual{<:IEEEFloat})
    return Dual(atanh(primal(x)), tangent(x) / (1 - primal(x)^2))
end
function rrule!!(::CoDual{typeof(atanh)}, x::CoDual{P}) where {P<:IEEEFloat}
    atanh_adjoint(dy::P) = NoRData(), dy / (1 - primal(x)^2)
    return zero_fcodual(atanh(primal(x))), atanh_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(asech),IEEEFloat}
function frule!!(::Dual{typeof(asech)}, x::Dual{<:IEEEFloat})
    return Dual(asech(primal(x)), -tangent(x) / (primal(x) * sqrt(1 - primal(x)^2)))
end
function rrule!!(::CoDual{typeof(asech)}, x::CoDual{P}) where {P<:IEEEFloat}
    asech_adjoint(dy::P) = NoRData(), -dy / (primal(x) * sqrt(1 - primal(x)^2))
    return zero_fcodual(asech(primal(x))), asech_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acsch),IEEEFloat}
function frule!!(::Dual{typeof(acsch)}, x::Dual{<:IEEEFloat})
    return Dual(acsch(primal(x)), -tangent(x) / (abs(primal(x)) * sqrt(1 + primal(x)^2)))
end
function rrule!!(::CoDual{typeof(acsch)}, x::CoDual{P}) where {P<:IEEEFloat}
    acsch_adjoint(dy::P) = NoRData(), -dy / (abs(primal(x)) * sqrt(1 + primal(x)^2))
    return zero_fcodual(acsch(primal(x))), acsch_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(acoth),IEEEFloat}
function frule!!(::Dual{typeof(acoth)}, x::Dual{<:IEEEFloat})
    return Dual(acoth(primal(x)), tangent(x) / (1 - primal(x)^2))
end
function rrule!!(::CoDual{typeof(acoth)}, x::CoDual{P}) where {P<:IEEEFloat}
    acoth_adjoint(dy::P) = NoRData(), dy / (1 - primal(x)^2)
    return zero_fcodual(acoth(primal(x))), acoth_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(sinc),IEEEFloat}
function frule!!(::Dual{typeof(sinc)}, x::Dual{<:IEEEFloat})
    return Dual(sinc(primal(x)), tangent(x) * cosc(primal(x)))
end
function rrule!!(::CoDual{typeof(sinc)}, x::CoDual{P}) where {P<:IEEEFloat}
    sinc_adjoint(dy::P) = NoRData(), dy * cosc(primal(x))
    return zero_fcodual(sinc(primal(x))), sinc_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(deg2rad),IEEEFloat}
function frule!!(::Dual{typeof(deg2rad)}, x::Dual{P}) where {P<:IEEEFloat}
    return Dual(deg2rad(primal(x)), tangent(x) * deg2rad(one(P)))
end
function rrule!!(::CoDual{typeof(deg2rad)}, x::CoDual{P}) where {P<:IEEEFloat}
    deg2rad_adjoint(dy::P) = NoRData(), dy * deg2rad(one(P))
    return zero_fcodual(deg2rad(primal(x))), deg2rad_adjoint
end

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

@is_primitive MinimalCtx Tuple{typeof(rad2deg),IEEEFloat}
function frule!!(::Dual{typeof(rad2deg)}, x::Dual{P}) where {P<:IEEEFloat}
    return Dual(rad2deg(primal(x)), tangent(x) * rad2deg(one(P)))
end
function rrule!!(::CoDual{typeof(rad2deg)}, x::CoDual{P}) where {P<:IEEEFloat}
    rad2deg_adjoint(dy::P) = NoRData(), dy * rad2deg(one(P))
    return zero_fcodual(rad2deg(primal(x))), rad2deg_adjoint
end

@from_chainrules MinimalCtx Tuple{typeof(^),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(^)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    t = (ChainRules.NoTangent(), tangent(x), tangent(y))
    z, dz = ChainRules.frule(t, ^, primal(x), primal(y))
    return Dual(z, dz)
end

@is_primitive MinimalCtx Tuple{typeof(atan),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(atan)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    _x, dx = extract(x)
    _y, dy = extract(y)
    return Dual(atan(_x, _y), (_y * dx - _x * dy) / (_x^2 + _y^2))
end
function rrule!!(::CoDual{typeof(atan)}, x::CoDual{P}, y::CoDual{P}) where {P<:IEEEFloat}
    function atan_adjoint(dz::P)
        tmp = primal(x)^2 + primal(y)^2
        return NoRData(), dz * primal(y) / tmp, -dz * primal(x) / tmp
    end
    return zero_fcodual(atan(primal(x), primal(y))), atan_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(hypot),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(hypot)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    h = hypot(primal(x), primal(y))
    dh = (primal(x) * tangent(x) + primal(y) * tangent(y)) / h
    return Dual(h, dh)
end
function rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P}) where {P<:IEEEFloat}
    h = hypot(primal(x), primal(y))
    hypot_pb!!(dh::P) = NoRData(), dh * (primal(x) / h), dh * (primal(y) / h)
    return zero_fcodual(h), hypot_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(hypot),P,P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(hypot)}, x::Dual{P}, y::Dual{P}, xs::Vararg{Dual{P},N}
) where {P<:IEEEFloat,N}
    h = hypot(primal(x), primal(y), map(primal, xs)...)
    dh = sum(primal(a) * tangent(a) for a in (x, y, xs...)) / h
    return Dual(h, dh)
end
function rrule!!(
    ::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P}, xs::Vararg{CoDual{P},N}
) where {P<:IEEEFloat,N}
    h = hypot(primal(x), primal(y), map(primal, xs)...)
    function hypot_pb!!(dh::P)
        grads = map(a -> dh * (primal(a) / h), (x, y, xs...))
        return NoRData(), grads...
    end
    return zero_fcodual(h), hypot_pb!!
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
        return NoRData(), -dy * y / (log_b * primal(b)), dy / (primal(x) * log_b)
    end
    return zero_fcodual(y), log_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(max),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(max)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    dz = ifelse(primal(x) > primal(y), tangent(x), tangent(y))
    return Dual(max(primal(x), primal(y)), dz)
end
function rrule!!(::CoDual{typeof(max)}, x::CoDual{P}, y::CoDual{P}) where {P<:IEEEFloat}
    function max_adjoint(dz::P)
        t = primal(x) > primal(y)
        return NoRData(), ifelse(t, dz, zero(P)), ifelse(t, zero(P), dz)
    end
    return zero_fcodual(max(primal(x), primal(y))), max_adjoint
end

@is_primitive MinimalCtx Tuple{typeof(min),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(min)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    dz = ifelse(primal(x) > primal(y), tangent(y), tangent(x))
    return Dual(min(primal(x), primal(y)), dz)
end
function rrule!!(::CoDual{typeof(min)}, x::CoDual{P}, y::CoDual{P}) where {P<:IEEEFloat}
    function min_adjoint(dz::P)
        t = primal(x) > primal(y)
        return NoRData(), ifelse(t, zero(P), dz), ifelse(t, dz, zero(P))
    end
    return zero_fcodual(min(primal(x), primal(y))), min_adjoint
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

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths})
    test_cases = vcat(
        map([Float32,Float64]) do P
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

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]
