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
    zero_fcodual,
    mooncake_tangent,
    increment_and_get_rdata!,
    zero_tangent,
    ReverseMode,
    ForwardMode,
    NoTangent

@from_chainrules DefaultCtx Tuple{typeof(airyai),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaix),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(airyaiprime),IEEEFloat}
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

# For using ChainRulesCore.NotImplemented within Mooncake Reverse Mode.
increment_and_get_rdata!(f::NoFData, r, t::CRC.NotImplemented) = r # zero gradient contribution for variable with cr_tangent = NotImplemented
mooncake_tangent(p, t::CRC.NotImplemented) = zero_tangent(p) # t - zero rdata for corresponding primal, so it's Mooncake Tangent gives zero to all branches.

# Mooncake.rrule!!s for SpecialFunctions whose ChainRules.rrule uses ChainRuleCore.NotImplemented gradients.
# Standard Bessel & Hankel
@from_rrule DefaultCtx Tuple{typeof(besseli),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(besselj),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(besselk),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(bessely),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(hankelh1),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(hankelh2),Number,Number}

# scaled bessel-i,j,k,y & hankelh1, hankelh2
@from_rrule DefaultCtx Tuple{typeof(besselix),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(besseljx),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(besselkx),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(besselyx),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(hankelh1x),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(hankelh2x),Number,Number}

# Scaled Airy Prime
@from_chainrules DefaultCtx Tuple{typeof(airyaiprimex),IEEEFloat}

# Gamma & Exponential Integrals
@from_rrule DefaultCtx Tuple{typeof(gamma),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(loggamma),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(expint),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(expintx),Number,Number}
@from_rrule DefaultCtx Tuple{typeof(gamma_inc),Number,Number,Integer}

"""
 In primitive ChainRules.frules, we see that in functions that use `NotImplemented` gradients - A special case of "NotImplemented poisoning" (As Base.:+/- overloaded uniquely for type)

```
julia> dself = ChainRuleCore.NoTangent()

julia> ChainRulesCore.frule((dself, 1,1,0), gamma_inc, x,x,0)
((0.6574288331765153, 0.34257116682348465), Tangent{Tuple{Float64, Float64}}(NotImplemented(SpecialFunctionsChainRulesCoreExt, #= Module =#, derivatives of the incomplete Gamma functions with 
respect to parameter `a` are not
implemented currently:
https://github.com/JuliaMath/SpecialFunctions.jl/issues/317
), NotImplemented(SpecialFunctionsChainRulesCoreExt, #= Module =#, derivatives of the incomplete Gamma functions with respect to parameter `a` are not
implemented currently:
https://github.com/JuliaMath/SpecialFunctions.jl/issues/317
)))

```

This spreads to all externally calling functions that internally call functions with `NotImplemented` internally.
example : ChainRulesCore.NotImplemented(Main,LineNumberNode(1),"test") + 12.0 = NotImplemented(Main, #= line 1 =#, test)

Therefore, we must write only *Mooncake* frules. (ChainRules frule is implicit zeroing the total derivative - "0 poisoning")
**With this combination of explicit 0 Mooncake.frule!!'s + `NotImplemented` ChainRules.rrule's we have coherent Math correct + AD safe results.**

"""
# 3 arg gamma_inc (1st arg gradient Intractable)
@is_primitive DefaultCtx ForwardMode Tuple{typeof(gamma_inc),IEEEFloat,IEEEFloat,Integer}

function frule!!(
    ::Dual{typeof(gamma_inc)}, a::Dual{T}, x::Dual{P}, IND::Dual{I,NoTangent}
) where {T<:IEEEFloat,P<:IEEEFloat,I<:Integer}
    ap, xp, indp = primal(a), primal(x), primal(IND)
    # use log-space for z to maintain numerical stability
    z = exp((ap - 1) * log(xp) - xp - loggamma(ap))

    # dot_p = dp/da * da + dp/dx * dx
    # dot_q = dq/da * da + dq/dx * dx
    # tangent(a) = 0 - Intractable Gradient
    return Dual(gamma_inc(ap, xp, indp), (tangent(x) * z, tangent(x) * (-z)))
end

# 2 arg Gamma and Exponential Integrals (1st arg gradient Intractable)
for func in [:gamma, :loggamma, :expint, :expintx]
    @eval begin
        @is_primitive DefaultCtx ForwardMode Tuple{typeof($func),IEEEFloat,IEEEFloat}

        function frule!!(
            ::Dual{typeof($func)}, a::Dual{T}, x::Dual{P}
        ) where {T<:IEEEFloat,P<:IEEEFloat}
            ap, xp = primal(a), primal(x)
            y = $func(ap, xp)

            # use log-space for z to maintain numerical stability
            ∂x = if $(QuoteNode(func)) == :gamma
                -exp((ap - 1) * log(xp) - xp)
            elseif $(QuoteNode(func)) == :loggamma
                # Derivative of log(Γ(a,x)) is originally -(x^(a-1) * e^-x) / Γ(a,x)
                -exp((ap - 1) * log(xp) - xp - loggamma(ap, xp))
            elseif $(QuoteNode(func)) == :expint
                # Derivative of E_n(x) = -E_{n-1}(x)
                -expint(ap - 1, xp)
            else # :expintx
                # expintx(a, x) = exp(x) * expint(a, x)
                # Derivative of  e^x * E_a(x) is originally e^x * E_a(x) - e^x * E_{a-1}(x) 
                y - expintx(ap - 1, xp)
            end

            # tangent(a) = 0 - Intractable Gradient
            return Dual(y, tangent(x) * ∂x)
        end
    end
end

end
