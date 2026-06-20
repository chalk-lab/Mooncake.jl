#
# nfwd-backed primitive rules for scalar functions.
#
# Each entry registers a primitive `frule!!` / `rrule!!` for a scalar (or small fixed-arity)
# math function by lifting its inputs into `NDual`s and running the overloaded primal directly:
# the `f(::NDual)` methods in Nfwd.jl propagate partials, so the rules stay close to the
# function's own definition and need no hand-coded derivative.
#
# MinimalCtx is used throughout: several of these functions (e.g. tanpi, sincosd, sincospi)
# contain `try`/`catch` internally, which Mooncake's IR-transform AD cannot handle. Registering
# them as MinimalCtx primitives dispatches the nfwd rule directly, bypassing the failing transform.
#
# Forward (`frule!!`): the input `Lifted{P,N,NDual{P,N}}` already carries the N seeded directions,
# so the rule just runs the primal on `tangent(x)` (the inner `NDual`) and wraps the result.
#
# Reverse (`rrule!!`): seed one `NDual` per input scalar on its own identity lane, run the primal
# once to obtain the (tiny, fixed-arity) Jacobian in the result's `.partials`, and contract it with
# the output cotangent in an allocation-free pullback (the result `NDual`(s) are `isbits` and
# captured by value). This mirrors the forward rule — no separate reverse engine.
#
# `Nfwd` (NDual + dual arithmetic) is a separate submodule; Nfwd names not in the top-level
# `using .Nfwd:` import list are referenced with a `Nfwd.` qualifier (e.g. `Nfwd._nfwd_pow_grad_x`).

# ── Reverse-mode helpers ──────────────────────────────────────────────────────
# Seed M scalar inputs as width-M `NDual`s, input i carrying the i-th identity direction, so a
# single primal evaluation produces the full M-column Jacobian in the output's `.partials`.
# `Tuple{P,Vararg{P,Mm1}}` (≥1 element), not `NTuple{M,P}`: the latter leaves `P` unbound at M==0
# (`Tuple{}`), which Aqua's `test_unbound_args` flags. Every caller passes ≥1 input, so requiring a
# first element is exact; the width is `Mm1 + 1`.
#
# `@generated` to unroll into an explicit tuple of `NDual` constructions with literal identity
# partials. A plain nested `ntuple` would close over the width `M` as a runtime `Int`, so the
# inner `Val(M)` became a dynamic `Val(::Int)` — JET flagged "failed to optimize due to recursion"
# and the uninferred result poisoned the downstream primal call (e.g. `max(::NDual,::NDual)::Any`)
# into runtime dispatch and allocation. The generator body only builds the construction `Expr`
# (every `one`/`zero`/`NDual` call runs at expansion's *result*, not in the generator), so it has
# no world-age sub-dispatch trap.
@generated function _nfwd_seed_inputs(
    primals::Tuple{P,Vararg{P,Mm1}}
) where {P<:IEEEFloat,Mm1}
    M = Mm1 + 1
    duals = map(1:M) do i
        partials = Expr(:tuple, (i == j ? :(one(P)) : :(zero(P)) for j in 1:M)...)
        return :(NDual{P,$M}(primals[$i], $partials))
    end
    return Expr(:tuple, duals...)
end

# Output primal value, for scalar (`NDual`) or tuple-of-`NDual` (e.g. `sincos`) results.
@inline _nfwd_out_value(yd::NDual) = yd.value
@inline _nfwd_out_value(yd::Tuple) = map(d -> d.value, yd)

# Contract the output cotangent(s) with the output partials → `NTuple{M}` per-input gradient.
# Scalar output: `yd::NDual{P,M}`, `ȳ::Number`. Tuple output: `yd::Tuple{NDual{P,M}...}`, `ȳ::Tuple`
# (each component's cotangent contracts its own partials; contributions sum per input lane).
# `_nfwd_zero_mask(ȳ, p)` zeroes the partial when the cotangent is zero, so a removable-singularity
# `Inf` partial (e.g. d(log)/dx at 0) does not poison an unused output to `Inf*0 = NaN` (issue #807).
@inline _contract(ȳ::Number, p) = ȳ * Nfwd._nfwd_zero_mask(ȳ, p)
@inline _nfwd_input_grads(yd::NDual, ȳ::Number) = map(p -> _contract(ȳ, p), yd.partials)
@inline function _nfwd_input_grads(yd::Tuple, ȳ::Tuple)
    return reduce(
        (a, b) -> map(+, a, b), map((d, c) -> map(p -> _contract(c, p), d.partials), yd, ȳ)
    )
end

# ===========================================================================
# nfwd-backed primitive rule registrations
# ===========================================================================

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
    nextfloat,
    prevfloat,
    Base.FastMath.exp_fast,
    Base.FastMath.exp2_fast,
    Base.FastMath.exp10_fast,
    Base.FastMath.atan_fast,
    Base.FastMath.sincos,
)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        # `$f(::NDual)` has its own overload in Nfwd.jl that propagates partials and sets
        # the result's `.value` to `f(primal(x))` (inner-value invariant), so read the
        # primal back from `dy` instead of recomputing it. `_nfwd_out_value`/`_typeof`
        # handle tuple-returning primitives (e.g. `sincos`): dy::Tuple{NDual,NDual}.
        function frule!!(
            ::Lifted{typeof($f),N}, x::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            dy = $f(tangent(x))
            y = _nfwd_out_value(dy)
            return Lifted{_typeof(y),N}(y, dy)
        end
        function rrule!!(::CoDual{typeof($f)}, x::CoDual{P}) where {P<:IEEEFloat}
            yd = $f(NDual{P,1}(primal(x), (one(P),)))
            nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
            return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
        end
    end
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
function rrule!!(::CoDual{typeof(tanpi)}, x::CoDual{P}) where {P<:IEEEFloat}
    yd = tanpi(NDual{P,1}(primal(x), (one(P),)))
    nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
    return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
for f in (atan, Base.FastMath.atan_fast, log, ^, mod, max, min)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        function frule!!(
            ::Lifted{typeof($f),N}, x1::Lifted{P,N,NDual{P,N}}, x2::Lifted{P,N,NDual{P,N}}
        ) where {N,P<:IEEEFloat}
            dy = $f(tangent(x1), tangent(x2))
            return Lifted{P,N}(dy.value, dy)
        end
        function rrule!!(
            ::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            yd = $f(_nfwd_seed_inputs((primal(x1), primal(x2)))...)
            nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
            return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
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
            dy = $f(tangent(x1), tangent(x2), tangent(x3))
            return Lifted{P,N}(dy.value, dy)
        end
        function rrule!!(
            ::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
        ) where {P<:IEEEFloat}
            yd = $f(_nfwd_seed_inputs((primal(x1), primal(x2), primal(x3)))...)
            nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
            return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
        end
    end
end

# ── sincosd ───────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincosd),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincosd),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincosd(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
function rrule!!(::CoDual{typeof(sincosd)}, x::CoDual{P}) where {P<:IEEEFloat}
    yd = sincosd(NDual{P,1}(primal(x), (one(P),)))
    nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
    return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
end

# ── sincospi ──────────────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(sincospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(sincospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    tv = sincospi(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
function rrule!!(::CoDual{typeof(sincospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    yd = sincospi(NDual{P,1}(primal(x), (one(P),)))
    nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
    return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
end

# ── modf ──────────────────────────────────────────────────────────────────────
# modf(x) = (frac, int) where frac = x - trunc(x); d(frac)/dx = 1, d(int)/dx = 0.

@is_primitive MinimalCtx Tuple{typeof(modf),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(modf),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    tv = modf(tangent(x))
    return Lifted{Tuple{P,P},N}(_nfwd_out_value(tv), tv)
end
function rrule!!(::CoDual{typeof(modf)}, x::CoDual{P}) where {P<:IEEEFloat}
    yd = modf(NDual{P,1}(primal(x), (one(P),)))
    nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
    return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
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
    yd = hypot(_nfwd_seed_inputs((primal(x), tuple_map(primal, xs)...))...)
    nfwd_pb!!(ȳ) = (NoRData(), _nfwd_input_grads(yd, ȳ)...)
    return zero_fcodual(_nfwd_out_value(yd)), nfwd_pb!!
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
