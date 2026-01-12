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
    CoDual,
    Dual,
    NoRData,
    NoFData,
    rrule!!,
    frule!!,
    primal,
    tangent,
    Tangent,
    zero_fcodual,
    mooncake_tangent,
    increment_and_get_rdata!,
    zero_tangent,
    ReverseMode,
    ForwardMode,
    NoTangent,
    to_cr_tangent,
    increment!!

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
    Handling ChainRuleCore.`NotImplemented` Tangents with @info logging.

In the CFG, derivative/gradient calculations follow the Forward Mode (FM) / Reverse Mode(RM) AD directions.
Assume inputs `x`, `y` to a node for function `f`, with `df/dx` being Intractable.
Now, 

1. FM AD calculates df using dx, dy.

Therefore, for FM AD rules : dx = 0 (dead) OR dx != 0 (active) -> We make this the conditional criteria.
if dx == 0
    # x does not contribute to df, so just set derivative as 0. 
    ∂f/∂x = 0
else 
    # x is expected to contribute to df.
    ∂f/∂x = NaN
    @info "Please use Finite Differences. Intractable Derivative for f wrt to a - setting ∂f/∂a to NaN."
end

df = (∂f/∂x)*dx + (∂f/∂y)*dy

if user/program inputs dx != 0 :
    ∂f = NaN     # @info used + upstream derivatives are simply `NaN` (we dint know the derivative anyways)

if user/program inputs dx == 0 :
    ∂f = (∂f/∂y)*dy    # upstream derivatives are correct as they dint depend on the "Intractable" derivative.

**Bottomline** : we always get correct derivatives or a `NaN` and information if we try to calculate at a singular/Intractable Point.


2. RM AD calculates dx and dy using df (pullback).
For RM AD : just set final dx = NaN directly and use with dy.

dx = (∂f/∂x)' * df     # Just set directly to `NaN`.
dy = (∂f/∂y)' * df     # calculate gradient as is.
@info "Currently Intractable gradients for f wrt to x - setting ∂f/∂x to NaN. Please use Finite Differences."

pullback for f simply returns `NaN`, dy

**Bottomline** : All downstream branches get correct gradients to work, In case they actively use dx they are set to `NaN` + Information given (unknown gradient).


**Note** - This will only work for Float point values & it's compositions (Complex etc) due to NaN, NaN32 & NaN16 living only AbstractFloat's Space.

**Final Note** - NaN issues is also solved in a similar manner.

"""

# NaN fillers for Floating Points Numbers
function Mooncake.increment_and_get_rdata!(
    f, r::T, t::CRC.NotImplemented
) where {T<:IEEEFloat}
    return T(NaN)
end

function Mooncake.mooncake_tangent(
    p::T, cr_tangent::CRC.NotImplemented
) where {T<:IEEEFloat}
    return T(NaN)
end

# NaN fillers for Complex Numbers
function Mooncake.increment_and_get_rdata!(
    f, r::T, t::CRC.NotImplemented
) where {P<:IEEEFloat,T<:Complex{P}}
    return mooncake_tangent(r, T(P(NaN), P(NaN)))
end

function Mooncake.mooncake_tangent(
    p::T, cr_tangent::CRC.NotImplemented
) where {P<:IEEEFloat,T<:Complex{P}}
    return mooncake_tangent(p, T(P(NaN), P(NaN)))
end

# for Non Holomorphic ChainRules -  Complex Number handling
function Mooncake.mooncake_tangent(p::T, cr_tangent::T) where {P<:IEEEFloat,T<:Complex{P}}
    return Mooncake.Tangent((re=real(cr_tangent), im=imag(cr_tangent)))
end

function Mooncake.to_cr_tangent(c::Tangent{@NamedTuple{re::T,im::T}}) where {T<:IEEEFloat}
    return Complex(c.fields.re, c.fields.im)
end

function Mooncake.increment_and_get_rdata!(
    f::NoFData, r::Mooncake.RData{@NamedTuple{re::T,im::T}}, t::Complex{T}
) where {T<:IEEEFloat}
    return Mooncake.RData((re=real(t) + r.data.re, im=imag(t) + r.data.im))
end

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

# 3 arg gamma_inc
@from_rrule DefaultCtx Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

# 3 arg gamma_inc (1st arg gradient Intractable)
@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gamma_inc),T,P,I
} where {T<:IEEEFloat,P<:IEEEFloat,I<:Integer}

function frule!!(
    ::Dual{typeof(gamma_inc)}, a::Dual{T}, x::Dual{P}, IND::Dual{I}
) where {T<:IEEEFloat,P<:IEEEFloat,I<:Integer}
    ap, xp, indp = primal(a), primal(x), primal(IND)
    da, dx = tangent(a), tangent(x)

    # ∂p/∂a - Intractable Gradient
    ∂a = if da != zero_tangent(da)
        @info "Please use Finite Differences. Intractable Derivative for gamma_inc wrt to a - setting ∂f/∂a to NaN."
        T(NaN)
    else
        T(0)
    end

    # ∂p/∂x - use log-space for z for numerical stability
    z = exp((ap - T(1)) * log(xp) - xp - loggamma(ap))

    # dot_p = ∂p/∂a * da + ∂p/∂x * dx
    # dot_q = ∂p/∂a * da + ∂p/∂x * dx
    return Dual(gamma_inc(ap, xp, indp), (∂a + (dx * z), ∂a + (dx * (-z))))
end

# 2 arg Gamma and Exponential Integrals (1st arg gradient Intractable)
for func in [:gamma, :loggamma, :expint, :expintx]
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{
            typeof($func),T,P
        } where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:IEEEFloat}

        function frule!!(
            ::Dual{typeof($func)}, a::Dual{T}, x::Dual{P}
        ) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:IEEEFloat}
            ap, xp = primal(a), primal(x)
            da, dx = tangent(a), tangent(x)

            y = $func(ap, xp)
            primal_type = typeof(y) # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂a - Intractable Gradient
            ∂a = if da != zero_tangent(da)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to a - setting ∂f/∂a to NaN."
                T(NaN)
            else
                T(0)
            end

            # ∂f/∂x - log-space for z to maintain numerical stability
            ∂x = if f_sym == :gamma
                -exp((ap - T(1)) * log(xp) - xp)
            elseif f_sym == :loggamma
                # Derivative of log(Γ(a,x)) is originally -(x^(a-1) * e^-x) / Γ(a,x)
                -exp((ap - T(1)) * log(xp) - xp - loggamma(ap, xp))
            elseif f_sym == :expint
                # Derivative of E_n(x) = -E_{n-1}(x)
                -expint(ap - T(1), xp)
            else # :expintx
                # expintx(a, x) = exp(x) * expint(a, x)
                # Derivative of e^x * E_a(x) is originally e^x * E_a(x) - e^x * E_{a-1}(x) 
                y - expintx(ap - T(1), xp)
            end

            # tangent(a) - Intractable Gradient
            # ensure dy and primal y are same types.

            dy_val = ∂a + ∂x * dx
            return Dual(y, ∂a + (dx * primal_type(∂x)))
        end

        @is_primitive DefaultCtx ForwardMode Tuple{
            typeof($func),T,P
        } where {T<:Union{IEEEFloat,Complex},P<:Complex}

        function frule!!(
            ::Dual{typeof($func)}, a::Dual{T}, x::Dual{P}
        ) where {L<:IEEEFloat,T<:Union{L,Complex{L}},P<:Complex}
            ap, xp = primal(a), primal(x)
            da, dx = tangent(a), tangent(x)

            y = $func(ap, xp)
            primal_type = typeof(y) # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂a - Intractable Gradient
            ∂a = if da != zero_tangent(da)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to a - setting ∂f/∂a to NaN."
                Complex(L(NaN), L(NaN))
            else
                Complex(L(0), L(0))
            end

            # ∂f/∂x - log-space for z to maintain numerical stability
            ∂x = if f_sym == :gamma
                -exp((ap - L(1)) * log(xp) - xp)
            elseif f_sym == :loggamma
                # Derivative of log(Γ(a,x)) is originally -(x^(a-1) * e^-x) / Γ(a,x)
                -exp((ap - L(1)) * log(xp) - xp - loggamma(ap, xp))
            elseif f_sym == :expint
                # Derivative of E_n(x) = -E_{n-1}(x)
                -expint(ap - L(1), xp)
            else # :expintx
                # expintx(a, x) = exp(x) * expint(a, x)
                # Derivative of e^x * E_a(x) is originally e^x * E_a(x) - e^x * E_{a-1}(x) 
                y - expintx(ap - L(1), xp)
            end

            # tangent(a) - Intractable Gradient
            # ensure dy and primal y are same types.
            dy_val = ∂a + ∂x * (dx.fields.re + im * dx.fields.im)
            return Dual(y, Mooncake.Tangent((re=real(dy_val), im=imag(dy_val))))
        end
    end
end

# 2 arg Standard Bessel and Hankel Functions
for func in [:besselj, :bessely, :besseli, :besselk, :hankelh1, :hankelh2]
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{typeof($func),IEEEFloat,IEEEFloat}

        function frule!!(
            ::Dual{typeof($func)}, v::Dual{T}, x::Dual{P}
        ) where {T<:IEEEFloat,P<:IEEEFloat}
            vp, xp = primal(v), primal(x)
            dv, dx = tangent(v), tangent(x)

            y = $func(vp, xp)
            primal_type = typeof(y)     # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂v - Intractable Gradient
            ∂v = if dv != zero_tangent(dv)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to x - setting ∂f/∂x to NaN."
                T(NaN)
            else
                T(0)
            end

            # Recurrence relations for derivatives wrt x.
            if f_sym == :besselk
                ∂x = -(besselk(vp - T(1), xp) + besselk(vp + T(1), xp)) / primal_type(2)
            elseif $(QuoteNode(func)) == :besseli
                ∂x = (besseli(vp - T(1), xp) + besseli(vp + T(1), xp)) / primal_type(2)
            else
                ∂x = ($func(vp - T(1), xp) - $func(vp + T(1), xp)) / primal_type(2)
            end

            return Dual(y, ∂v + (dx * ∂x))
        end

        # write frule for Complex X to avoid Method Ambiguities
        @is_primitive DefaultCtx ForwardMode Tuple{typeof($func),IEEEFloat,Complex}

        function frule!!(
            ::Dual{typeof($func)}, v::Dual{T}, x::Dual{P}
        ) where {T<:IEEEFloat,P<:Complex}
            vp, xp = primal(v), primal(x)
            dv, dx = tangent(v), tangent(x)

            y = $func(vp, xp)
            primal_type = typeof(y)     # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂v - Intractable Gradient
            ∂v = if dv != zero_tangent(dv)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to x - setting ∂f/∂x to NaN."
                Complex(T(NaN), T(NaN))
            else
                Complex(T(0), T(0))
            end

            # Recurrence relations for derivatives wrt x.
            if f_sym == :besselk
                ∂x = -(besselk(vp - T(1), xp) + besselk(vp + T(1), xp)) / primal_type(2)
            elseif $(QuoteNode(func)) == :besseli
                ∂x = (besseli(vp - T(1), xp) + besseli(vp + T(1), xp)) / primal_type(2)
            else
                ∂x = ($func(vp - T(1), xp) - $func(vp + T(1), xp)) / primal_type(2)
            end

            # x complex
            dy_val = ∂v + ∂x * (dx.fields.re + im * dx.fields.im)

            return Dual(y, Mooncake.Tangent((re=real(dy_val), im=imag(dy_val))))
        end
    end
end

## Holomorphic Stuff
# 2-arg Scaled Bessel Functions
for func in [:besselix, :besselkx, :besseljx, :besselyx]
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{
            typeof($func),T,P
        } where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex}}

        function frule!!(
            ::Dual{typeof($func)}, v::Dual{T}, x::Dual{P}
        ) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex}}
            vp, xp = primal(v), primal(x)
            dv, dx = tangent(v), tangent(x)
            y = $func(vp, xp)
            primal_type = typeof(y)      # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂v - Intractable Gradient
            ∂v = if dv != zero_tangent(dv)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to x - setting ∂f/∂x to NaN."
                P <: Complex ? Complex(T(NaN), T(NaN)) : T(NaN)
            else
                P <: Complex ? Complex(T(0), T(0)) : T(0)
            end

            # Recurrence relations for derivatives wrt x.
            if f_sym == :besselix
                ∂x =
                    (besselix(vp - T(1), xp) + besselix(vp + T(1), xp)) / primal_type(2) -
                    sign(real(xp)) * y
            elseif f_sym == :besselkx
                ∂x =
                    -(besselkx(vp - T(1), xp) + besselkx(vp + T(1), xp)) / primal_type(2) +
                    y
            else
                # besseljx and besselyx scaling contribution is 0 for Real x
                ∂x = ($func(vp - T(1), xp) - $func(vp + T(1), xp)) / primal_type(2)
            end

            # x can be complex
            dy_val = if P <: Complex
                ∂v + ∂x * (dx.fields.re + im * dx.fields.im)
            else
                ∂v + ∂x * dx
            end

            return Dual(y, Mooncake.Tangent((re=real(dy_val), im=imag(dy_val))))
        end
    end
end

# Scaled Hankel Functions
for func in [:hankelh1x, :hankelh2x]
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{
            typeof($func),T,P
        } where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex}}

        function frule!!(
            ::Dual{typeof($func)}, v::Dual{T}, x::Dual{P}
        ) where {T<:IEEEFloat,P<:Union{IEEEFloat,Complex}}
            vp, xp = primal(v), primal(x)
            dv, dx = tangent(v), tangent(x)

            y = $func(vp, xp)
            primal_type = typeof(y)     # to ensure final Dual Tangent type is valid
            f_sym = $(QuoteNode(func))

            # ∂f/∂v - Intractable Gradient
            ∂v = if dv != zero_tangent(dv)
                @info "Please use Finite Differences. Intractable Derivative for $f_sym wrt to x - setting ∂f/∂x to NaN."
                P <: Complex ? Complex(T(NaN), T(NaN)) : T(NaN)
            else
                P <: Complex ? Complex(T(0), T(0)) : T(0)
            end

            # Recurrence relations for derivatives wrt x
            if f_sym == :hankelh1x
                ∂x =
                    (hankelh1x(vp - T(1), xp) - hankelh1x(vp + T(1), xp)) / primal_type(2) -
                    im * y
            else
                ∂x =
                    (hankelh2x(vp - T(1), xp) - hankelh2x(vp + T(1), xp)) / primal_type(2) +
                    im * y
            end

            # raw complex tangents - x can be complex
            dy_val = if P <: Complex
                ∂v + ∂x * (dx.fields.re + im * dx.fields.im)
            else
                ∂v + ∂x * dx
            end

            return Dual(y, Mooncake.Tangent((re=real(dy_val), im=imag(dy_val))))
        end
    end
end

end
