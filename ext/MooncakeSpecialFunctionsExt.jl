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
    frule!!,
    Tangent,
    primal,
    notimplemented_tangent_guard,
    ForwardMode,
    numberify,
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
    Mooncake handling for `ChainRuleCore.NotImplemented` Tangents in imported rules.

In the AD graph, derivative/gradient calculations follow the Forward Mode / Reverse Mode AD directions.
Assume inputs `x`, `y` to a node for function `f`, with `df/dx` being NotImplemented.
Now, 

1. Forward Mode AD calculates df using dx, dy.

Therefore, for Forward Mode rules : dx = 0 (dead) OR dx != 0 (active) -> We make this the conditional criteria.
if dx == 0
    # x does not contribute to df, so just set derivative as 0. 
    ∂f/∂x = 0
else 
    # x is expected to contribute to df.
    ∂f/∂x = NaN
    @info "Please use Finite Differences. NotImplemented Derivative for f wrt to a - setting ∂f/∂a to NaN."
end

df = (∂f/∂x)*dx + (∂f/∂y)*dy

if user/program inputs dx != 0 :
    ∂f = NaN     # @info used + upstream derivatives are simply `NaN` (we don't know the derivative anyways)

if user/program inputs dx == 0 :
    ∂f = (∂f/∂y)*dy    # upstream derivatives are correct as they don't depend on the "NotImplemented" derivative.

**Bottomline** : we always get correct derivatives or a `NaN` and information if we try to calculate at a singular/NotImplemented Point.


2. Reverse Mode AD calculates dx and dy using df (pullback).
For Reverse Mode rules : just set final dx = `NaN` directly and use with dy.

dx = (∂f/∂x)' * df     # Just set directly to `NaN`.
dy = (∂f/∂y)' * df     # calculate gradient as is.
@info "Currently NotImplemented gradients for f wrt to x - setting ∂f/∂x to NaN. Please use Finite Differences."

pullback for f simply returns `NaN`, dy

**Bottomline** : All downstream branches get correct gradients to work, In case they actively use dx they are set to `NaN` + Information given (unknown gradient).


**Note** - This will only work for Float point values & it's compositions (Complex etc) due to NaN, NaN32 & NaN16 living only AbstractFloat's Space.

**Final Note** - NaN issues is also solved in a similar manner.

"""
# SpecialFunctions.jl Rules for functions with ChainRuleCore.`NotImplemented` gradients.
# Standard Bessel & Hankel rrules
@from_rrule DefaultCtx Tuple{typeof(besseli),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselj),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselk),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(bessely),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2),IEEEFloat,Union{IEEEFloat,<:Complex}}

# scaled bessel-i,j,k,y & hankelh1, hankelh2 rrules
@from_rrule DefaultCtx Tuple{typeof(besselix),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besseljx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselkx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(besselyx),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh1x),IEEEFloat,Union{IEEEFloat,<:Complex}}
@from_rrule DefaultCtx Tuple{typeof(hankelh2x),IEEEFloat,Union{IEEEFloat,<:Complex}}

# Gamma & Exponential Integrals rrules
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

@from_rrule DefaultCtx Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

# handle frule return type according to primal type. 
function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:IEEEFloat}
    return Dual(y, primal_eltype(dy_val))
end

function real_or_complex_valued(y::L, primal_eltype, dy_val) where {L<:Complex}
    return Dual(
        y,
        Mooncake.Tangent((re=primal_eltype(real(dy_val)), im=primal_eltype(imag(dy_val)))),
    )
end

# 3 arg gamma_inc (1st arg gradient Not Implemented)
@is_primitive DefaultCtx ForwardMode Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

function frule!!(
    ::Dual{typeof(gamma_inc)}, _a::Dual{T}, _x::Dual{P}, _IND::Dual{I}
) where {T<:IEEEFloat,P<:IEEEFloat,I<:Integer}
    a, da = extract(_a)
    x, dx = extract(_x)
    IND = primal(_IND)

    y = gamma_inc(a, x, IND) # primal is always Real for gamma_inc
    primal_eltype = eltype(y) # to ensure final Dual Tangent is valid type

    ∂a = Mooncake.notimplemented_tangent_guard(da, :gamma_inc)     # ∂p/∂a - NotImplemented Gradient
    z = exp((a - 1) * log(x) - x - loggamma(a))    # ∂p/∂x

    # dot_p = ∂p/∂a * da + ∂p/∂x * dx
    # dot_q = ∂p/∂a * da + (-∂p/∂x) * dx
    return Dual(y, (primal_eltype(∂a + (dx * z)), primal_eltype(∂a + (dx * -z))))
end

# 2 arg Gamma and Exponential Integrals (1st arg gradients Not Implemented)
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

    ∂a = Mooncake.notimplemented_tangent_guard(da, :gamma)  # ∂f/∂a - NotImplemented Gradient
    ∂x = -exp((a - 1) * log(x) - x)    # ∂f/∂x

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * numberify(dx)
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
    ∂a = Mooncake.notimplemented_tangent_guard(da, :loggamma)
    # ∂f/∂x - Derivative of log(Γ(a,x)) is originally -(x^(a-1) * e^-x) / Γ(a,x)
    ∂x = -exp((a - 1) * log(x) - x - loggamma(a, x))

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * numberify(dx)
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
    ∂a = Mooncake.notimplemented_tangent_guard(da, :expint)
    # ∂f/∂x - Derivative of E_n(x) = -E_{n-1}(x)
    ∂x = -expint(a - 1, x)

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * numberify(dx)
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
    ∂a = Mooncake.notimplemented_tangent_guard(da, :expintx)
    # ∂f/∂x -  Derivative of e^x * E_a(x) is originally e^x * E_a(x) - e^x * E_{a-1}(x)
    ∂x = y - expintx(a - 1, x)

    # Ignore tangent(a) - NotImplemented Gradient
    dy_val = ∂a + ∂x * numberify(dx)
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

# 2 arg Standard Bessel and Hankel Functions
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besselj)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = (besselj(v - 1, x) - besselj(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
    # All Bessel return Complex Numbers only for Complex inputs
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :bessely)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = (bessely(v - 1, x) - bessely(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besseli)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = (besseli(v - 1, x) + besseli(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besselk)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = -(besselk(v - 1, x) + besselk(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :hankelh1)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = (hankelh1(v - 1, x) - hankelh1(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :hankelh2)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = (hankelh2(v - 1, x) - hankelh2(v + 1, x)) / 2

    dy_val = ∂v + ∂x * numberify(dx)
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

## Non - Holomorphic functions
# 2-arg Scaled Bessel Functions
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besselix)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x_1 = (besselix(v - 1, x) + besselix(v + 1, x)) / 2
    ∂x_2 = -sign(real(x)) * y

    # Non Holomorphic scaling
    dy_val = (
        ∂v + ∂x_1 * numberify(dx) + ∂x_2 * (P <: Complex ? dx.fields.re : numberify(dx))
    )

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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besselkx)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x = -(besselkx(v - 1, x) + besselkx(v + 1, x)) / 2 + y

    dy_val = ∂v + ∂x * numberify(dx)
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besseljx)
    # Recurrence relations for derivatives wrt x.
    ∂x_1 = (besseljx(v - 1, x) - besseljx(v + 1, x)) / 2
    ∂x_2 = ∂x_2 = -sign(imag(x)) * y

    # ∂f/∂x - Non Holomorphic scaling
    dy_val = (
        ∂v + ∂x_1 * numberify(dx) + (P <: Complex ? ∂x_2 * dx.fields.im : primal_eltype(0))
    )
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :besselyx)
    # ∂f/∂x - Recurrence relations for derivatives wrt x.
    ∂x_1 = (besselyx(v - 1, x) - besselyx(v + 1, x)) / 2
    ∂x_2 = ∂x_2 = -sign(imag(x)) * y

    # Non Holomorphic scaling
    dy_val = (
        ∂v + ∂x_1 * numberify(dx) + (P <: Complex ? ∂x_2 * dx.fields.im : primal_eltype(0))
    )
    return real_or_complex_valued(y, primal_eltype, dy_val)
end

# Scaled Hankel Functions
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :hankelh1x)
    # ∂f/∂x - Recurrence relations
    ∂x = (hankelh1x(v - 1, x) - hankelh1x(v + 1, x)) / 2 - im * y

    dy_val = ∂v + ∂x * numberify(dx)
    return Dual(
        y,
        Mooncake.Tangent((re=primal_eltype(real(dy_val)), im=primal_eltype(imag(dy_val)))),
    )
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
    ∂v = Mooncake.notimplemented_tangent_guard(dv, :hankelh2x)
    # ∂f/∂x - Recurrence relations
    ∂x = (hankelh2x(v - 1, x) - hankelh2x(v + 1, x)) / 2 + im * y

    dy_val = ∂v + ∂x * numberify(dx)
    return Dual(
        y,
        Mooncake.Tangent((re=primal_eltype(real(dy_val)), im=primal_eltype(imag(dy_val)))),
    )
end

end
