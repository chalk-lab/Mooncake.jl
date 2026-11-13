#
# Primitive rules for scalar functions: the `frule!!`s run the nfwd engine on `NDual`
# numbers, and the `rrule!!`s are analytic pullbacks.
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
# Each pullback scales the cotangent through `nan_tangent_guard`, so a zero cotangent
# stays zero at a nonfinite derivative, and its forward rule keeps the cotangent's
# perturbation alive under forward-over-reverse.
# Beside an infinite argument, `atan`/`atan_fast` and `hypot` can use zero limits.
# Adding each result to `+0.0` gives a zero derivative the sign that accumulating into
# a zero gradient gives.
#

# ── nfwd-backed unary scalar rules ─────────────────────────────────────────────
for (f, coefficient) in (
    (exp, :y),
    (exp2, :(y * oftype(y, log(2)))),
    (exp10, :(y * oftype(y, log(10)))),
    (expm1, :(exp(x))),
    (log, :(inv(x))),
    (log10, :(inv(x * oftype(x, log(10))))),
    (log2, :(inv(x * oftype(x, log(2))))),
    (log1p, :(inv(one(x) + x))),
    (sqrt, :(inv(2 * y))),
    (cbrt, :(inv(3 * y^2))),
    (sin, :(cos(x))),
    (cos, :(-sin(x))),
    (cospi, :(-oftype(x, π) * sinpi(x))),
    (tan, :(inv(cos(x))^2)),
    (sec, :(y * tan(x))),
    (csc, :(-y * cot(x))),
    (cot, :(-(one(y) + y^2))),
    (sind, :(deg2rad(cosd(x)))),
    (cosd, :(-deg2rad(sind(x)))),
    (tand, :(deg2rad(one(y) + y^2))),
    (secd, :(deg2rad(y * tand(x)))),
    (cscd, :(-deg2rad(y * cotd(x)))),
    (cotd, :(-deg2rad(one(y) + y^2))),
    (sinpi, :(oftype(x, π) * cospi(x))),
    (asin, :(inv(sqrt(one(x) - x^2)))),
    (acos, :(-inv(sqrt(one(x) - x^2)))),
    (atan, :(inv(one(x) + x^2))),
    (asec, :(inv(abs(x) * sqrt(x^2 - one(x))))),
    (acsc, :(-inv(abs(x) * sqrt(x^2 - one(x))))),
    (acot, :(-inv(one(x) + x^2))),
    (asind, :(inv(deg2rad(sqrt(one(x) - x^2))))),
    (acosd, :(-inv(deg2rad(sqrt(one(x) - x^2))))),
    (atand, :(inv(deg2rad(one(x) + x^2)))),
    (asecd, :(inv(deg2rad(abs(x) * sqrt(x^2 - one(x)))))),
    (acscd, :(-inv(deg2rad(abs(x) * sqrt(x^2 - one(x)))))),
    (acotd, :(-inv(deg2rad(one(x) + x^2)))),
    (sinh, :(cosh(x))),
    (cosh, :(sinh(x))),
    # Not `1 - y^2`, which vanishes once `tanh` rounds to one, before `sech(x)^2` does.
    (tanh, :(
        let u = exp(-2 * abs(x))
            4u / (one(u) + u)^2
        end
    )),
    (sech, :(-tanh(x) * y)),
    (csch, :(-coth(x) * y)),
    (coth, :(-csch(x)^2)),
    (asinh, :(inv(sqrt(x^2 + one(x))))),
    (acosh, :(inv(sqrt(x^2 - one(x))))),
    (atanh, :(inv(one(x) - x^2))),
    (asech, :(-inv(x * sqrt(one(x) - x^2)))),
    (acsch, :(-inv(abs(x) * sqrt(one(x) + x^2)))),
    (acoth, :(inv(one(x) - x^2))),
    (sinc, :(cosc(x))),
    (deg2rad, :(deg2rad(one(x)))),
    (rad2deg, :(rad2deg(one(x)))),
    (mod2pi, :(Nfwd._nfwd_mod2pi_grad(x))),
    (Base.eps, :(zero(x))),
    (nextfloat, :(one(x))),
    (prevfloat, :(one(x))),
    (Base.FastMath.exp_fast, :(exp(x))),
    (Base.FastMath.exp2_fast, :(exp2(x) * oftype(x, log(2)))),
    (Base.FastMath.exp10_fast, :(exp10(x) * oftype(x, log(10)))),
    (Base.FastMath.atan_fast, :(inv(one(x) + x^2))),
    (Base.FastMath.sincos, :((y[2], -y[1]))),
    (tanpi, :(oftype(x, π) * (one(y) + y^2))),
    (sincosd, :((deg2rad(y[2]), -deg2rad(y[1])))),
    (sincospi, :((oftype(x, π) * y[2], -oftype(x, π) * y[1]))),
    # At ±Inf, `modf` returns `x` itself as the integral part.
    (
        modf,
        :(ifelse(
            isinf(x), (zero(x), one(x)), (ifelse(isnan(x), oftype(x, NaN), one(x)), zero(x))
        )),
    ),
)
    # Call the primitive nfwd entrypoints directly here rather than constructing
    # `Rule{...}()` / `RRule{...}()` on every call. These wrappers sit on hot scalar paths,
    # so avoiding per-invocation wrapper construction keeps them allocation-free. See the
    # file-level warning above for why hidden cached Rule/RRule state is also a bad fit
    # for primitive rules.
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P} where {P<:IEEEFloat}
        function frule!!(fdual::Dual{typeof($f)}, x::Dual{P}) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), fdual, x)
        end
        function rrule!!(::CoDual{typeof($f)}, x_codual::CoDual{P}) where {P<:IEEEFloat}
            x = primal(x_codual)
            y = $f(x)
            c = $coefficient
            # `map` over numbers applies the function once, so scalar outputs share this.
            pullback(ȳ) = (
                NoRData(),
                zero(P) + sum(map((ȳi, ci) -> nan_tangent_guard(ȳi, ȳi * ci), ȳ, c)),
            )
            return zero_fcodual(y), pullback
        end
    end
end

# ── nfwd-backed fixed-arity scalar rules ──────────────────────────────────────
for (f, coefficients) in (
    (atan, :((inv(a^2 + b^2) * b, inv(a^2 + b^2) * -a))),
    (Base.FastMath.atan_fast, :((inv(a^2 + b^2) * b, inv(a^2 + b^2) * -a))),
    (log, :((-y / (a * log(a)), inv(b * log(a))))),
    (^, :((Nfwd._nfwd_pow_grad_x(a, b, y), Nfwd._nfwd_pow_grad_p(a, b, y)))),
    # As in the `NDual` partial, a nonfinite `b` coefficient makes the `a` derivative NaN.
    (mod, :(
        let (ca, cb) = Nfwd._nfwd_mod_grad_coeffs(a, b)
            (ca + zero(ca) * cb, cb)
        end
    )),
    (max, :(ifelse(Nfwd._ndual_pick_max(a, b), (one(a), zero(a)), (zero(a), one(a))))),
    (min, :(ifelse(Nfwd._ndual_pick_min(a, b), (one(a), zero(a)), (zero(a), one(a))))),
)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P} where {P<:IEEEFloat}
        function frule!!(
            fdual::Dual{typeof($f)}, x1::Dual{P}, x2::Dual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), fdual, x1, x2)
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}
        ) where {P<:IEEEFloat}
            a, b = primal(x1), primal(x2)
            y = $f(a, b)
            c = $coefficients
            pullback(ȳ::P) = (
                NoRData(), map(ci -> zero(P) + nan_tangent_guard(ȳ, ȳ * ci), c)...
            )
            return zero_fcodual(y), pullback
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
    c = Nfwd._nfwd_pow_grad_x(_x, P(_n), float(y))
    pow_fast_pb!!(dy::P) = (NoRData(), nan_tangent_guard(dy, dy * c), NoRData())
    return zero_fcodual(y), pow_fast_pb!!
end

for f in (clamp,)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),P,P,P} where {P<:IEEEFloat}
        function frule!!(
            fdual::Dual{typeof($f)}, x1::Dual{P}, x2::Dual{P}, x3::Dual{P}
        ) where {P<:IEEEFloat}
            return NfwdMooncake._nfwd_primitive_frule_call(Val(1), fdual, x1, x2, x3)
        end
        function rrule!!(
            fcodual::CoDual{typeof($f)}, x1::CoDual{P}, x2::CoDual{P}, x3::CoDual{P}
        ) where {P<:IEEEFloat}
            a, lo, hi = primal(x1), primal(x2), primal(x3)
            # As in the `NDual` method, the lower bound is tested first.
            at_lo = a <= lo
            at_hi = !at_lo & (a >= hi)
            c = (
                ifelse(at_lo | at_hi, zero(P), one(P)),
                ifelse(at_lo, one(P), zero(P)),
                ifelse(at_hi, one(P), zero(P)),
            )
            clamp_pb(ȳ::P) = (
                NoRData(), map(ci -> zero(P) + nan_tangent_guard(ȳ, ȳ * ci), c)...
            )
            return zero_fcodual(clamp(a, lo, hi)), clamp_pb
        end
    end
end

# angle_fast is constant on real inputs, so dispatch directly to the zero-derivative path.
@zero_derivative MinimalCtx Tuple{typeof(Base.FastMath.angle_fast),P} where {P<:IEEEFloat}

# ── hypot(x, xs...) ───────────────────────────────────────────────────────────

@is_primitive MinimalCtx Tuple{typeof(hypot),P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    f::Dual{typeof(hypot)}, x::Dual{P}, xs::Vararg{Dual{P},M}
) where {P<:IEEEFloat,M}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x, xs...)
end
function rrule!!(
    f::CoDual{typeof(hypot)}, x::CoDual{P}, xs::Vararg{CoDual{P},M}
) where {P<:IEEEFloat,M}
    v = (primal(x), map(primal, xs)...)
    h = hypot(v...)
    # Unary `hypot` is `abs`, so `sign` gives ±1 at ±Inf, where `v / h` is NaN. Otherwise
    # fold pairwise as the `NDual` method does, `hypot(hypot(a, b), c, ...)`, chaining each
    # step's coefficients, which are zero at the origin, where `v / h` is `0 / 0`.
    # An entry not yet folded in holds that argument's partial: zero, or NaN once `hp`
    # overflows, as in the `NDual` method; an infinite argument contributes no such NaN.
    c =
        let c = if M == 0
                map(sign, v)
            else
                ntuple(j -> ifelse(j == 1, one(P), zero(P)), Val(M + 1))
            end,
            hp = v[1],
            finite = all(isfinite, v)

            for i in 2:(M + 1)
                hi = hypot(hp, v[i])
                k, ki = ifelse(iszero(hi), (zero(P), zero(P)), (hp / hi, v[i] / hi))
                c = map(cj -> k * cj, c)
                c = Base.setindex(c, ki + ifelse(finite, c[i], zero(P)), i)
                hp = hi
            end
            c
        end
    hypot_pb(ȳ::P) = (NoRData(), map(ci -> zero(P) + nan_tangent_guard(ȳ, ȳ * ci), c)...)
    return zero_fcodual(h), hypot_pb
end
