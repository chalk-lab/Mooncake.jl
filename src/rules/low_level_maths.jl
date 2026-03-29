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

# Many scalar smooth rules now route through `nfwd` in `rules_via_nfwd.jl`.
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
