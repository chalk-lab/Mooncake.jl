module MooncakeFluxExt

using Mooncake, Flux
using Base: IEEEFloat
import Mooncake: DefaultCtx, rrule!!, @is_primitive, CoDual, zero_fcodual, NoRData

@is_primitive DefaultCtx Tuple{
    typeof(Flux.Losses.mse),Array{P},Array{P}
} where {P<:IEEEFloat}

# This is a performance-specific rule motivated by https://github.com/chalk-lab/Mooncake.jl/issues/466
function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)}, X::CoDual{<:Array{P}}, Y::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function flux_mse_pullback(dloss::P)
        # adjoints got by VJP reverse pass equations.

        tmp = dloss * P(2 / length(X.x))
        @inbounds for n in eachindex(X.x)
            d = X.x[n] - Y.x[n]
            X.dx[n] += tmp * d
            Y.dx[n] -= tmp * d
        end
        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(Flux.Losses.mse(X.x, Y.x)), flux_mse_pullback
end

end
