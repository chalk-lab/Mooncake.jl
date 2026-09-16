module MooncakeSpecialFunctionsExt

using SpecialFunctions, Mooncake
using Base: IEEEFloat
import ChainRulesCore as CRC
import Mooncake:
    DefaultCtx,
    @from_chainrules,
    @from_rrule,
    @zero_derivative,
    @is_primitive,
    Dual,
    CoDual,
    rrule!!,
    zero_fcodual,
    NoRData,
    nan_tangent_guard,
    frule!!,
    Tangent,
    primal,
    notimplemented_tangent_guard,
    ForwardMode,
    extract

@from_chainrules DefaultCtx Tuple{typeof(airyai),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaix),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaiprime),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaiprimex),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airybi),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airybiprime),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(besselj0),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(besselj1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(bessely0),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(bessely1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(dawson),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(digamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erf),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erf),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfc),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logerfc),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfcinv),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfcx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logerfcx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfi),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(erfinv),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(gamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(invdigamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(trigamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(polygamma),Integer,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(beta),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logbeta),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logabsgamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(loggamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expintx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(expinti),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(sinint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cosint),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(ellipk),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(ellipe),IEEEFloat}

@zero_derivative DefaultCtx Tuple{typeof(logfactorial),Integer}

"""
## Handling `ChainRulesCore.NotImplemented` Tangents in Imported Rules

Mooncake uses a *masking* trick to handle
`ChainRulesCore.NotImplemented` partial derivatives.

**NOTE:**
A missing partial derivative is irrelevant if it is multiplied by the zero
element of the corresponding tangent or cotangent space.

The result is therefore either correct or explicitly marked as unknown
(`NaN`).

### Forward mode (pushforward)

For `f(x, y)` with unimplemented `∂f/∂x`:

    ḟ = (∂f/∂x)·ẋ + (∂f/∂y)·ẏ

- If `ẋ == 0`, the contribution from `x` vanishes:
  
      ḟ = (∂f/∂y)·ẏ

- If `ẋ != 0`, the missing derivative is required ⇒ `ḟ = NaN`.

### Reverse mode (pullback)

Given upstream cotangent `f̄`:

    x̄ = (∂f/∂x)'·f̄
    ȳ = (∂f/∂y)'·f̄

- If `f̄ == 0`, then `x̄ = 0` even if `∂f/∂x` is not implemented.
- If `f̄ != 0`, then `x̄ = NaN`, while `ȳ` is computed normally.

### Notes

- “Zero” refers to the additive identity of the tangent/cotangent space.
- This trick relies on `NaN` and therefore applies only to
  floating-point tangent spaces and their compositions
  (e.g. arrays of floats).
  This restriction exists because `NaN`, `NaN32`, and `NaN16`
  live exclusively in `AbstractFloat` spaces.
- `NotImplemented` indicates a missing AD rule, not a non-differentiable
  function. Any resulting `NaN`s are handled using the same masking
  principle: they affect the result only when multiplied by a nonzero
  tangent or cotangent, and are otherwise safely ignored.

### Outcome

Correct derivatives when possible, and explicit `NaN` only when
an unimplemented partial is mathematically required.
"""
#
# Standard Bessel & Hankel functions
#
@from_rrule DefaultCtx Tuple{typeof(besseli),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselj),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselk),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(bessely),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2),IEEEFloat,Union{IEEEFloat,<:Complex}}

# Scaled bessel-i,j,k,y & hankelh1, hankelh2 rrules
@from_rrule DefaultCtx Tuple{typeof(besselix),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besseljx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselkx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselyx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1x),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2x),IEEEFloat,Union{IEEEFloat,<:Complex}}

# Gamma and exponential integral rrules
@from_rrule DefaultCtx Tuple{
    typeof(gamma),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(loggamma),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(expint),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}
@from_rrule DefaultCtx Tuple{
    typeof(expintx),Union{IEEEFloat,<:Complex},Union{IEEEFloat,<:Complex}
}

# Ensure the frule return type matches the primal type.
function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:IEEEFloat}
    return Dual(y, primal_eltype(dy_val))
end
function real_or_complex_valued(y::Complex{L}, primal_eltype, dy_val) where {L<:IEEEFloat}
    return Dual(y, Complex(primal_eltype(real(dy_val)), primal_eltype(imag(dy_val))))
end

function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:Complex}
    return Dual(
        y,
        Mooncake.Tangent((re=primal_eltype(real(dy_val)), im=primal_eltype(imag(dy_val)))),
    )
end

# Both partials are supported for finite positive shapes and positive x. At a=0 or
# x=0, use one-sided first derivatives; at x=Inf, both partials are zero. Infinite
# shapes are unsupported. The series and continued fraction have a 100,000-iteration
# limit, which very large shapes near x=a can reach. The separate gamma(a,x) and
# loggamma(a,x) shape derivatives remain unimplemented.
# Stan also changes numerical regimes for the shape derivative:
# https://github.com/stan-dev/math/blob/develop/stan/math/prim/fun/grad_reg_lower_inc_gamma.hpp
function gamma_inc_partials(a::T, x::T, y) where {T<:IEEEFloat}
    (isnan(a) || isnan(x)) && return (T(NaN), T(NaN))
    isfinite(a) || throw(DomainError(a, "gamma_inc derivatives require finite a"))
    isinf(x) && return (zero(T), zero(T))
    iszero(a) && return (-expint(x), zero(T))
    iszero(x) && return zero(T), x^(a - 1) * exp(-loggamma(a))
    tol = 4 * eps(T)
    # Differentiate the lower series (DLMF 8.7.1) without subtracting two full sums.
    if x < a + 1
        t, s = one(T), one(T)
        dt, ds = zero(T), zero(T)
        for n in 1:100_000
            r = x / (a + n)
            dt = r * (dt - t / (a + n))
            t *= r
            s += t
            ds += dt
            if abs(t) <= tol * abs(s) && abs(dt) <= tol * abs(ds)
                g = log(x) - digamma(a + 1) + ds / s
                A = y[1] * g
                if x < 1
                    D = exp((a - 1) * log(x) + log(a) - loggamma(a + 1) - x)
                    # Keep the x factor outside exp so underflow preserves its derivative.
                    y[1] < floatmin(T) && (A = D * ((x / a) * (s * g)))
                elseif y[1] < floatmin(T)
                    # Recover representable derivatives when the primal ratio underflows.
                    l = a * (log(x) - loggamma(a + 1) / a) - x
                    A = -exp(l + log(s) + log(-g))
                    D = exp(l + log(a) - log(x))
                else
                    D = (y[1] / x) * (a / s)
                end
                return A, D
            end
        end
    else
        # Differentiate the upper continued fraction (DLMF 8.9.2).
        b = x + 1 - a
        c, f = b, b
        dc, df = -one(T), -one(T)
        # ld stores d'/d; d' itself can underflow in the far upper tail.
        d, ld = zero(T), zero(T)
        for n in 1:100_000
            an = n * (a - n)
            b += 2
            ld = 1 - n * d - an * d * ld
            d = inv(b + an * d)
            ld *= d
            dc = -1 + n / c - (an / c) * (dc / c)
            c = b + an / c
            delta = c * d
            ddelta = delta * (dc / c + ld)
            df = df * delta + f * ddelta
            f *= delta
            # Integer shapes terminate the primal fraction before its derivative.
            if abs(delta - 1) <= tol && abs(ddelta) <= tol * abs(df / f)
                # Avoid the digamma pole at tiny positive shapes.
                g = log(x) - digamma(a + 1) - df / f
                A = -(y[2] * g + y[2] / a)
                D = y[2] * (f / x)
                if y[2] < floatmin(T)
                    l = a * (log(x) - loggamma(a + 1) / a) - x
                    A = -exp(l - log(f) + log1p(a * g))
                    D = exp(l + log(a) - log(x))
                end
                return A, D
            end
        end
    end
    error("gamma_inc derivative did not converge")
end

# Keep working precision through seed multiplication to avoid premature overflow.
function gamma_inc_partials(a::T, x::T, y) where {T<:Union{Float16,Float32}}
    aw, xw = widen(a), widen(x)
    return gamma_inc_partials(aw, xw, gamma_inc(aw, xw))
end

# NIST Digital Library of Mathematical Functions, incomplete beta expansions:
# https://dlmf.nist.gov/8.17.E7 (series), https://dlmf.nist.gov/8.17.E22
# and https://dlmf.nist.gov/8.17.E23 (continued fraction and coefficients).
# Shapes must be finite and positive; endpoints use one-sided first derivatives.
# The fourth argument remains 1-x under differentiation; nonconvergence throws.
function beta_inc_partials(a::T, b::T, x::T, y::T, pq) where {T<:IEEEFloat}
    any(isnan, (a, b, x, y)) && return (T(NaN), T(NaN), T(NaN))
    (isfinite(a) && isfinite(b) && a > 0 && b > 0) ||
        throw(DomainError((a, b), "beta_inc derivatives require finite positive shapes"))
    if iszero(x)
        D = if a == 1
            b
        elseif a < 1
            T(Inf)
        else
            zero(T)
        end
        return zero(T), zero(T), D
    elseif iszero(y)
        D = if b == 1
            a
        elseif b < 1
            T(Inf)
        else
            zero(T)
        end
        return zero(T), zero(T), D
    end
    # Choose the faster continued fraction (DLMF 8.17(v)).
    reflect = x > (a+1)/(a+b+2)
    if reflect
        a, b, x, y = b, a, y, x
    end
    lx, ly = x < y ? (log(x), log1p(-x)) : (log1p(-y), log(y))
    lb = logbeta(a, b)
    D = exp((a-1)*lx + (b-1)*ly - lb)
    tol = 8eps(T)
    converged = false
    local ga, gb, lh
    # Factor out a in the power series so small shape partials do not cancel.
    if a <= 1 && x <= 0.5 && b*x <= 1
        u, ub = one(T), zero(T)
        s, sa, sb = zero(T), zero(T), zero(T)
        for n in 1:100_000
            ub = ((n-b)*ub-u)*x/n
            u *= (n-b)*x/n
            t, ta, tb = u/(a+n), (n/(a+n))*(u/(a+n)), ub/(a+n)
            s += t
            sa += ta
            sb += tb
            if abs(t) <= tol*abs(s) && abs(ta) <= tol*abs(sa) && abs(tb) <= tol*abs(sb)
                h = 1+a*s
                ga = lx + digamma(a+b+1)-digamma(a+1)+sa/h
                # A midpoint expansion avoids cancellation in the digamma difference.
                db = if a < sqrt(sqrt(eps(T)))*(b+1)
                    mid = b+1+a/2
                    a*(trigamma(mid)+a^2*polygamma(3, mid)/24)
                else
                    digamma(a+b+1)-digamma(b+1)
                end
                gb = db+a*sb/h
                lh = log1p(a*s) - b*ly
                converged = true
                break
            end
        end
    else
        c = one(T)
        d = inv(1 - (a+b)*x/(a+1))
        h = d
        ca, cb = zero(T), zero(T)
        da, db = d*x*(1-b)/(a+1)^2, d*x/(a+1)
        ha, hb = da, db
        for n in 2:100_000
            m = n ÷ 2
            if iseven(n)
                r = (m/(a+2m))*((b-m)/(a+2m-1))*x
                ra = -r*(inv(a+2m-1)+inv(a+2m))
                rb = (m/(a+2m))*x/(a+2m-1)
            else
                r = -((a+m)/(a+2m))*((a+b+m)/(a+2m+1))*x
                ra = r*(inv(a+m)+inv(a+b+m)-inv(a+2m)-inv(a+2m+1))
                rb = -((a+m)/(a+2m))*x/(a+2m+1)
            end
            dn = inv(1+r*d)
            da, db = -dn*d*(ra+r*da), -dn*d*(rb+r*db)
            d = dn
            cn = 1+r/c
            ca, cb = (ra-r*ca)/(c*cn), (rb-r*cb)/(c*cn)
            c = cn
            delta = c*d
            dha, dhb = ca+da, cb+db
            ha += dha
            hb += dhb
            h *= delta
            # Integer shapes can terminate the fraction before its derivatives.
            if abs(delta-1) <= tol && abs(dha) <= tol*abs(ha) && abs(dhb) <= tol*abs(hb)
                ga = lx + digamma(a+b+1) - digamma(a+1) + ha
                gb = ly + digamma(a+b+1) - digamma(b+1) + hb
                lh = log(h)
                converged = true
                break
            end
        end
    end
    converged || error("beta_inc derivatives did not converge")
    p = reflect ? pq[2] : pq[1]
    # Shift digamma away from zero and combine reciprocal terms before scaling.
    A, B = p*ga - p/(a+b), p*gb + (p*(a/(a+b)))/b
    if p < floatmin(T)
        lp = a*lx+b*ly-lb+lh-log(a)
        A = copysign(exp(lp+log(abs(ga))), ga) - exp(lp-log(a+b))
        B = copysign(exp(lp+log(abs(gb))), gb) + exp(lp+log(a/(a+b))-log(b))
    end
    return reflect ? (-B, -A, D) : (A, B, D)
end

function beta_inc_partials(a::T, b::T, x::T, pq) where {T<:IEEEFloat}
    return beta_inc_partials(a, b, x, 1-x, pq)
end

function beta_inc_partials(a::T, b::T, x::T, pq) where {T<:Union{Float16,Float32}}
    values = widen.((a, b, x))
    return beta_inc_partials(values..., beta_inc(values...))
end

function beta_inc_partials(a::T, b::T, x::T, y::T, pq) where {T<:Union{Float16,Float32}}
    values = widen.((a, b, x, y))
    return beta_inc_partials(values..., beta_inc(values...))
end

for f in (beta_inc, SpecialFunctions._beta_inc)
    @eval @is_primitive DefaultCtx Tuple{typeof($f),IEEEFloat,IEEEFloat,IEEEFloat}
    @eval @is_primitive DefaultCtx Tuple{typeof($f),IEEEFloat,IEEEFloat,IEEEFloat,IEEEFloat}
end

function frule!!(
    _f::Dual{<:Union{typeof(beta_inc),typeof(SpecialFunctions._beta_inc)}},
    _a::Dual{T},
    _b::Dual{S},
    _x::Dual{U},
    _y::Vararg{Dual{<:IEEEFloat},N},
) where {T<:IEEEFloat,S<:IEEEFloat,U<:IEEEFloat,N}
    a, adot = extract(_a)
    b, bdot = extract(_b)
    x, xdot = extract(_x)
    y, ydot = isempty(_y) ? (1-x, -xdot) : extract(only(_y))
    pq = primal(_f)(a, b, x, map(primal, _y)...)
    A, B, D = beta_inc_partials(promote(a, b, x, map(primal, _y)...)..., pq)
    # The fourth argument is 1-x; use the smaller coordinate for its tangent.
    seed = x <= y ? xdot : -ydot
    dp = typeof(pq[1])(
        nan_tangent_guard(adot, A*adot) +
        nan_tangent_guard(bdot, B*bdot) +
        nan_tangent_guard(seed, D*seed),
    )
    return Dual(pq, (dp, -dp))
end

function rrule!!(
    _f::CoDual{<:Union{typeof(beta_inc),typeof(SpecialFunctions._beta_inc)}},
    _a::CoDual{T},
    _b::CoDual{S},
    _x::CoDual{U},
    _y::Vararg{CoDual{<:IEEEFloat},N},
) where {T<:IEEEFloat,S<:IEEEFloat,U<:IEEEFloat,N}
    a, b, x = primal(_a), primal(_b), primal(_x)
    y = isempty(_y) ? 1-x : primal(only(_y))
    pq = primal(_f)(a, b, x, map(primal, _y)...)
    A, B, D = beta_inc_partials(promote(a, b, x, map(primal, _y)...)..., pq)
    function beta_inc_pb!!(dpq)
        d = typeof(A)(dpq[1]) - typeof(A)(dpq[2])
        da = T(nan_tangent_guard(d, A*d))
        db = S(nan_tangent_guard(d, B*d))
        dx = nan_tangent_guard(d, D*d)
        isempty(_y) && return NoRData(), da, db, U(dx)
        return NoRData(),
        da, db, x <= y ? U(dx) : zero(U),
        typeof(y)(x <= y ? zero(dx) : -dx)
    end
    return zero_fcodual(pq), beta_inc_pb!!
end

@is_primitive DefaultCtx Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

function frule!!(
    ::Dual{typeof(gamma_inc)}, _a::Dual{T}, _x::Dual{S}, _ind::Dual{I}
) where {T<:IEEEFloat,S<:IEEEFloat,I<:Integer}
    a, adot = extract(_a)
    x, xdot = extract(_x)
    ind = primal(_ind)
    y = gamma_inc(a, x, ind)
    A, D = gamma_inc_partials(promote(a, x)..., iszero(ind) ? y : gamma_inc(a, x))
    dx = isfinite(D) ? D * xdot : nan_tangent_guard(xdot, D * xdot)
    dp = typeof(y[1])(A * adot + dx)
    return Dual(y, (dp, -dp))
end

function rrule!!(
    ::CoDual{typeof(gamma_inc)}, _a::CoDual{T}, _x::CoDual{S}, _ind::CoDual{I}
) where {T<:IEEEFloat,S<:IEEEFloat,I<:Integer}
    a, x, ind = primal(_a), primal(_x), primal(_ind)
    y = gamma_inc(a, x, ind)
    A, D = gamma_inc_partials(promote(a, x)..., iszero(ind) ? y : gamma_inc(a, x))
    function gamma_inc_pb!!(dy)
        d = typeof(A)(dy[1]) - typeof(A)(dy[2])
        dx = isfinite(D) ? D * d : nan_tangent_guard(d, D * d)
        return NoRData(), T(A * d), S(dx), NoRData()
    end
    return zero_fcodual(y), gamma_inc_pb!!
end

# 2-arg Gamma and exponential integrals (first-argument gradient is `NotImplemented`)
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gamma),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Dual{typeof(gamma)}, _a::Dual{T}, _x::Dual{P}
) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a, da = extract(_a)
    x, dx = extract(_x)

    y = gamma(a, x) # primal is always complex for complex inputs.
    primal_eltype = eltype(y isa Complex ? y.re : y)

    ∂a = Mooncake.notimplemented_tangent_guard(da)  # ∂f/∂a - NotImplemented Gradient
    ∂x = -exp((a - 1) * log(x) - x)    # ∂f/∂x

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val) # ensure dy and primal y are same types.
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(loggamma),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Dual{typeof(loggamma)}, _a::Dual{T}, _x::Dual{P}
) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a, da = extract(_a)
    x, dx = extract(_x)

    y = loggamma(a, x) # primal is always complex for complex inputs.
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂a - NotImplemented Gradient
    ∂a = Mooncake.notimplemented_tangent_guard(da)
    # ∂f/∂x - Derivative of log(Γ(a,x)) is originally -(x^(a-1) * e^-x) / Γ(a,x)
    ∂x = -exp((a - 1) * log(x) - x - loggamma(a, x))

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(expint),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Dual{typeof(expint)}, _a::Dual{T}, _x::Dual{P}
) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a, da = extract(_a)
    x, dx = extract(_x)

    y = expint(a, x) # primal is always complex for complex inputs.
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂a - NotImplemented Gradient
    ∂a = Mooncake.notimplemented_tangent_guard(da)
    # ∂f/∂x - Derivative of E_n(x) = -E_{n-1}(x)
    ∂x = -expint(a - 1, x)

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(expintx),
    Union{IEEEFloat,Complex{<:IEEEFloat}},
    Union{IEEEFloat,Complex{<:IEEEFloat}},
}
function frule!!(
    ::Dual{typeof(expintx)}, _a::Dual{T}, _x::Dual{P}
) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    a, da = extract(_a)
    x, dx = extract(_x)

    y = expintx(a, x) # expintx(a, x) = exp(x) * expint(a, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂a - NotImplemented Gradient
    ∂a = Mooncake.notimplemented_tangent_guard(da)
    # ∂f/∂x -  Derivative of e^x * E_a(x) is originally e^x * E_a(x) - e^x * E_{a-1}(x)
    ∂x = y - expintx(a - 1, x)

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

# 2-arg standard Bessel and Hankel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselj),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besselj)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besselj(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = (besselj(v - 1, x) - besselj(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    # All Bessel functions return complex values only for complex inputs.
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(bessely),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(bessely)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = bessely(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = (bessely(v - 1, x) - bessely(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besseli),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besseli)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besseli(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = (besseli(v - 1, x) + besseli(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselk),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besselk)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besselk(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = -(besselk(v - 1, x) + besselk(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh1),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(hankelh1)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = hankelh1(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = (hankelh1(v - 1, x) - hankelh1(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh2),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(hankelh2)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = hankelh2(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = (hankelh2(v - 1, x) - hankelh2(v + 1, x)) / 2

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

#
# Non Holomorphic functions
#

# 2-arg scaled Bessel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselix),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besselix)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besselix(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)     # to ensure final Dual Tangent type is valid

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x_1 = (besselix(v - 1, x) + besselix(v + 1, x)) / 2
    ∂x_2 = -sign(real(x)) * y

    # Non Holomorphic scaling
    dy_val = ∂v + ∂x_1 * dx + ∂x_2 * real(dx)

    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselkx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besselkx)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besselkx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x = -(besselkx(v - 1, x) + besselkx(v + 1, x)) / 2 + y

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besseljx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besseljx)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besseljx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # Recurrence relations for derivatives w.r.t. x.
    ∂x_1 = (besseljx(v - 1, x) - besseljx(v + 1, x)) / 2
    ∂x_2 = ∂x_2 = -sign(imag(x)) * y

    # ∂f/∂x - Non Holomorphic scaling
    dy_val = (∂v + ∂x_1 * dx + ∂x_2 * imag(dx))
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(besselyx),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(besselyx)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = besselyx(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations for derivatives w.r.t. x.
    ∂x_1 = (besselyx(v - 1, x) - besselyx(v + 1, x)) / 2
    ∂x_2 = ∂x_2 = -sign(imag(x)) * y

    # Non Holomorphic scaling
    dy_val = ∂v + ∂x_1 * dx + ∂x_2 * imag(dx)
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

# Scaled Hankel functions
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh1x),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(hankelh1x)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = hankelh1x(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations
    ∂x = (hankelh1x(v - 1, x) - hankelh1x(v + 1, x)) / 2 - im * y

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(hankelh2x),IEEEFloat,Union{IEEEFloat,Complex{<:IEEEFloat}}
}
function frule!!(
    ::Dual{typeof(hankelh2x)}, _v::Dual{T}, _x::Dual{P}
) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex{<:IEEEFloat}}}
    v, dv = extract(_v)
    x, dx = extract(_x)

    y = hankelh2x(v, x)
    primal_eltype = eltype(y isa Complex ? y.re : y)

    # ∂f/∂v - NotImplemented Gradient
    ∂v = Mooncake.notimplemented_tangent_guard(dv)
    # ∂f/∂x - Recurrence relations
    ∂x = (hankelh2x(v - 1, x) - hankelh2x(v + 1, x)) / 2 + im * y

    dy_val = ∂v + ∂x * dx
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

# ── NDual overloads for SpecialFunctions ──────────────────────────────────────
#
# These let NDual-typed inputs (used by nfwd / Hessian) propagate through
# the special-function calls that appear inside distribution logpdfs (Beta,
# Gamma, Chi, Dirichlet, …).
#
# Each method evaluates the primal at the Float64 value, then propagates the
# N partial slots using the known scalar derivative:
#
#   d/dx loggamma(x)     = digamma(x)
#   d/dx digamma(x)      = trigamma(x)
#   d/dx trigamma(x)     = polygamma(2, x)
#   d/dx polygamma(n, x) = polygamma(n+1, x)
#   d/dx logbeta(x, y)   = digamma(x) - digamma(x+y)
#   d/dy logbeta(x, y)   = digamma(y) - digamma(x+y)
#   d/dx beta(x, y)      = beta(x, y) * (digamma(x) - digamma(x+y))
#   d/dx gamma(x)        = gamma(x) * digamma(x)
#   d/dx erf(x)          = 2/√π · exp(-x²)
#   d/dx erfc(x)         = -2/√π · exp(-x²)
#   d/dx erfinv(x)       = √π/2 · exp(erfinv(x)²)
#   d/dx besselk(ν, x)   = -(besselk(ν-1,x) + besselk(ν+1,x)) / 2

using Mooncake.Nfwd: NDual

@inline function SpecialFunctions.loggamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.digamma(v)
    return NDual{T,N}(SpecialFunctions.loggamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.logabsgamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    lv, sv = SpecialFunctions.logabsgamma(v)
    dv = SpecialFunctions.digamma(v)
    return (NDual{T,N}(lv, ntuple(k -> dv * x.partials[k], Val(N))), sv)
end

@inline function SpecialFunctions.digamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.trigamma(v)
    return NDual{T,N}(SpecialFunctions.digamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.trigamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.polygamma(2, v)
    return NDual{T,N}(SpecialFunctions.trigamma(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.polygamma(
    n::Integer, x::NDual{T,N}
) where {T<:IEEEFloat,N}
    v = x.value
    dv = SpecialFunctions.polygamma(n + 1, v)
    return NDual{T,N}(
        SpecialFunctions.polygamma(n, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.gamma(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    gv = SpecialFunctions.gamma(v)
    dv = gv * SpecialFunctions.digamma(v)
    return NDual{T,N}(gv, ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.logbeta(
    x::NDual{T,N}, y::NDual{T,N}
) where {T<:IEEEFloat,N}
    xv, yv = x.value, y.value
    ψx = SpecialFunctions.digamma(xv)
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv),
        ntuple(k -> (ψx - ψxy) * x.partials[k] + (ψy - ψxy) * y.partials[k], Val(N)),
    )
end

@inline function SpecialFunctions.logbeta(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N}
    xv, yv = x.value, T(y)
    ψx = SpecialFunctions.digamma(xv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv), ntuple(k -> (ψx - ψxy) * x.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.logbeta(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N}
    xv, yv = T(x), y.value
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        SpecialFunctions.logbeta(xv, yv), ntuple(k -> (ψy - ψxy) * y.partials[k], Val(N))
    )
end

@inline function SpecialFunctions.beta(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    xv, yv = x.value, y.value
    bv = SpecialFunctions.beta(xv, yv)
    ψx = SpecialFunctions.digamma(xv)
    ψy = SpecialFunctions.digamma(yv)
    ψxy = SpecialFunctions.digamma(xv + yv)
    return NDual{T,N}(
        bv,
        ntuple(k -> bv * ((ψx - ψxy) * x.partials[k] + (ψy - ψxy) * y.partials[k]), Val(N)),
    )
end

@inline function SpecialFunctions.erf(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = T(2 / sqrt(π)) * exp(-v^2)
    return NDual{T,N}(SpecialFunctions.erf(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.erfc(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    dv = -T(2 / sqrt(π)) * exp(-v^2)
    return NDual{T,N}(SpecialFunctions.erfc(v), ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline function SpecialFunctions.erfinv(x::NDual{T,N}) where {T<:IEEEFloat,N}
    v = x.value
    inv_v = SpecialFunctions.erfinv(v)
    dv = T(sqrt(π) / 2) * exp(inv_v^2)
    return NDual{T,N}(inv_v, ntuple(k -> dv * x.partials[k], Val(N)))
end

@inline _ndual_partials_are_zero(partials::NTuple) = all(iszero, partials)

@noinline function _throw_ndual_notimplemented(name::Symbol, argname::Symbol)
    throw(
        ArgumentError(
            "SpecialFunctions.$name does not support nfwd differentiation with respect to " *
            "`$argname`; pass a constant parameter or a promoted NDual with zero partials.",
        ),
    )
end

# Helper: extract the scalar (Float64/Float32) value from ν, which may be an NDual when the
# Julia promote machinery wraps an order into the same type as x. Active order tangents are
# not supported here and must fail loudly rather than being silently dropped.
function _bessel_nu(ν::NDual)
    _ndual_partials_are_zero(ν.partials) || _throw_ndual_notimplemented(:bessel, :ν)
    return ν.value
end
_bessel_nu(ν::Real) = ν

# d/dx besselk(ν, x) = -(besselk(ν-1, x) + besselk(ν+1, x)) / 2
@inline function SpecialFunctions.besselk(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = -(SpecialFunctions.besselk(νv - 1, v) + SpecialFunctions.besselk(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besselk(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# d/dx besseli(ν, x) = (besseli(ν-1, x) + besseli(ν+1, x)) / 2
# Without NDual overloads the generic `bessel*(nu::Real, x::Real) = bessel*(nu, float(x))`
# path recurses infinitely because `float(x::NDual) = x`.
@inline function SpecialFunctions.besseli(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.besseli(νv - 1, v) + SpecialFunctions.besseli(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besseli(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# besselix(ν, x) = besseli(ν, x) * exp(-|x|)  (exponentially scaled).
# VonMises stores besselix(0, κ) in the constructor, making this the hot path.
# d/dx besselix(ν,x) = (besselix(ν-1,x)+besselix(ν+1,x))/2 - sign(x) * besselix(ν,x)
@inline function SpecialFunctions.besselix(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    yv = SpecialFunctions.besselix(νv, v)
    dv =
        (SpecialFunctions.besselix(νv - 1, v) + SpecialFunctions.besselix(νv + 1, v)) / 2 -
        sign(v) * yv
    return NDual{T,N}(yv, ntuple(k -> dv * x.partials[k], Val(N)))
end

# besselkx(ν, x) = besselk(ν, x) * exp(x)  (exponentially scaled).
# d/dx besselkx(ν,x) = besselkx(ν,x) - (besselkx(ν-1,x)+besselkx(ν+1,x))/2  (x>0)
@inline function SpecialFunctions.besselkx(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    yv = SpecialFunctions.besselkx(νv, v)
    dv =
        yv -
        (SpecialFunctions.besselkx(νv - 1, v) + SpecialFunctions.besselkx(νv + 1, v)) / 2
    return NDual{T,N}(yv, ntuple(k -> dv * x.partials[k], Val(N)))
end

# d/dx bessely(ν, x) = (bessely(ν-1, x) - bessely(ν+1, x)) / 2
@inline function SpecialFunctions.bessely(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.bessely(νv - 1, v) - SpecialFunctions.bessely(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.bessely(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

# d/dx besselj(ν, x) = (besselj(ν-1, x) - besselj(ν+1, x)) / 2
@inline function SpecialFunctions.besselj(ν::Real, x::NDual{T,N}) where {T<:IEEEFloat,N}
    νv, v = _bessel_nu(ν), x.value
    dv = (SpecialFunctions.besselj(νv - 1, v) - SpecialFunctions.besselj(νv + 1, v)) / 2
    return NDual{T,N}(
        SpecialFunctions.besselj(νv, v), ntuple(k -> dv * x.partials[k], Val(N))
    )
end

function SpecialFunctions._beta_inc(
    a::NDual{T,N}, b::NDual{T,N}, x::NDual{T,N}, _y::Vararg{NDual{T,N},M}
) where {T<:IEEEFloat,N,M}
    y = isempty(_y) ? 1-x : only(_y)
    av, bv, xv, yv = a.value, b.value, x.value, y.value
    values = (av, bv, xv, map(t -> t.value, _y)...)
    pq = beta_inc(values...)
    A, B, D = beta_inc_partials(values..., pq)
    dp = ntuple(Val(N)) do k
        adot, bdot = a.partials[k], b.partials[k]
        seed = xv <= yv ? x.partials[k] : -y.partials[k]
        T(
            nan_tangent_guard(adot, A*adot) +
            nan_tangent_guard(bdot, B*bdot) +
            nan_tangent_guard(seed, D*seed),
        )
    end
    return NDual{T,N}(pq[1], dp), NDual{T,N}(pq[2], map(-, dp))
end

end
