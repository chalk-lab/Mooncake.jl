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
    Lifted,
    MinimalCtx,
    NoRData,
    arrayify,
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
    ::Lifted{typeof(rand!),Nw},
    rng::Lifted{<:RNG,Nw},
    sampler::Lifted{<:ArrayLikeSampleable,Nw},
    x::Lifted{P,Nw},
) where {P<:CuFloatArray,Nw}
    px, dx = arrayify(x)
    rand!(primal(rng), primal(sampler), px)
    for lane in 1:Nw
        fill!(dx[lane], zero(eltype(P)))
    end
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
