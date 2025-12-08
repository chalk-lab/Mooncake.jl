module MooncakeLogExpFunctionsExt

using LinearAlgebra: dot
using LogExpFunctions
using Base: IEEEFloat
import Mooncake:
    DefaultCtx,
    @from_chainrules,
    frule!!,
    rrule!!,
    Dual,
    CoDual,
    primal,
    tangent,
    @is_primitive,
    zero_fcodual,
    NoRData,
    extract

@from_chainrules DefaultCtx Tuple{typeof(xlogx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xlogy),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xlog1py),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xexpx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(xexpy),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logistic),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logit),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logcosh),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logabssinh),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1psq),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1pexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1mexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log2mexp),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logexpm1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(log1pmx),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logmxp1),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logaddexp),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(logsubexp),IEEEFloat,IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cloglog),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(cexpexp),IEEEFloat}

# logsumexp and logsumexp! need a custom rule to avoid incorrect derivatives due to
# branching in the primal implementation. (In principle, the forward-mode rule for logsumexp
# could be imported from ChainRulesCore, but that leads to extra allocations, so we
# reimplement them.)
@is_primitive DefaultCtx Tuple{typeof(logsumexp),AbstractArray{<:IEEEFloat}}
@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),AbstractArray{<:IEEEFloat}
}
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(logsumexp)},
    x::Dual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    y = logsumexp(primal(x); primal(kwargs)...)
    dy = sum(tangent(x) .* (exp.(primal(x) .- y)); primal(kwargs)...)
    return Dual(y, dy)
end
function frule!!(
    ::Dual{typeof(logsumexp)}, x::Dual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    y = logsumexp(primal(x))
    dy = zero(P)
    xp, dx = extract(x)
    # same as dy = dot(dx, exp.(xp .- y)) but unrolled to avoid allocations
    for i in eachindex(dx)
        @inbounds dy += dx[i] * exp(xp[i] - y)
    end
    return Dual(y, dy)
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual{<:NamedTuple{(:dims,),<:Tuple{Colon}}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    y = logsumexp(primal(x); primal(kwargs)...)
    dx = tangent(x)
    function logsumexp_pb!!(dy::P)
        dx .+= dy * exp.(primal(x) .- y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    # all other dim arguments - i.e. ints or tuples thereof
    kwargs::CoDual{<:NamedTuple{(:dims,),<:Tuple{Any}}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    y = logsumexp(primal(x); primal(kwargs)...)
    dy = zero(y)
    dx = tangent(x)
    function logsumexp_pb!!(::NoRData)
        dx .+= dy .* exp.(primal(x) .- y)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), logsumexp_pb!!
end
function rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:AbstractArray{P}}
) where {P<:IEEEFloat}
    y = logsumexp(primal(x))
    dx = tangent(x)
    function logsumexp_pb!!(dy::P)
        dx .+= dy .* exp.(primal(x) .- y)
        return NoRData(), NoRData()
    end
    return zero_fcodual(y), logsumexp_pb!!
end

@is_primitive DefaultCtx Tuple{typeof(logsumexp!),AbstractArray{<:P},AbstractArray{<:P}} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(logsumexp!)},
    out::Dual{<:AbstractArray{P}},
    x::Dual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    logsumexp!(primal(out), primal(x))
    sum!(tangent(out), tangent(x) .* exp.(primal(x) .- primal(out)))
    return out
end
function rrule!!(
    ::CoDual{typeof(logsumexp!)},
    out::CoDual{<:AbstractArray{P}},
    x::CoDual{<:AbstractArray{P}},
) where {P<:IEEEFloat}
    old_out = copy(primal(out))
    logsumexp!(primal(out), primal(x))
    y, dy = extract(out)
    dx = tangent(x)
    function logsumexp!_pb!!(::NoRData)
        dx .+= dy .* exp.(primal(x) .- y)
        copyto!(y, old_out)
        return NoRData(), NoRData(), NoRData()
    end
    return out, logsumexp!_pb!!
end

end
