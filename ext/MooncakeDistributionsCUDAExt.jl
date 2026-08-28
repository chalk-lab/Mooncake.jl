module MooncakeDistributionsCUDAExt

using Base: IEEEFloat
using CUDA: RNG
using CUDA.CUDACore: CuArray
using Distributions: ArrayLikeVariate, Sampleable
using Mooncake: Mooncake
using Random: rand!

import Mooncake:
    @is_primitive,
    CoDual,
    Dual,
    MinimalCtx,
    NoRData,
    frule!!,
    primal,
    rrule!!,
    tangent,
    zero_rdata

const ArrayLikeSampleable = Sampleable{<:ArrayLikeVariate}
const CuFloatArray = CuArray{<:IEEEFloat}

# A device draw is random state, not a reparameterised sample. The result aliases `x`, so
# `@zero_derivative` would break the primal/fdata aliasing invariant: clear the existing
# fdata instead. Restore `x` on the reverse pass, but deliberately leave `rng` advanced.
@is_primitive MinimalCtx Tuple{typeof(rand!),RNG,<:ArrayLikeSampleable,<:CuFloatArray}
function frule!!(
    ::Dual{typeof(rand!)},
    rng::Dual{<:RNG},
    sampler::Dual{<:ArrayLikeSampleable},
    x::Dual{P,P},
) where {P<:CuFloatArray}
    rand!(primal(rng), primal(sampler), primal(x))
    fill!(tangent(x), zero(eltype(P)))
    return x
end
function rrule!!(
    ::CoDual{typeof(rand!)},
    rng::CoDual{<:RNG},
    sampler::CoDual{<:ArrayLikeSampleable},
    x::CoDual{P,P},
) where {P<:CuFloatArray}
    px, dx = primal(x), tangent(x)
    px_copy = copy(px)
    dx_copy = copy(dx)
    rand!(primal(rng), primal(sampler), px)
    fill!(dx, zero(eltype(P)))
    rng_rdata = zero_rdata(primal(rng))
    sampler_rdata = zero_rdata(primal(sampler))
    function rand!_pb!!(::NoRData)
        copyto!(px, px_copy)
        copyto!(dx, dx_copy)
        return NoRData(), rng_rdata, sampler_rdata, NoRData()
    end
    return x, rand!_pb!!
end

end
