module MooncakeChainRulesExt

using ChainRules, LinearAlgebra, Mooncake
using Base: IEEEFloat

import Mooncake:
    @from_rrule,
    @is_primitive,
    CoDual,
    DefaultCtx,
    Dual,
    MinimalCtx,
    NoRData,
    frule!!,
    primal,
    rrule!!,
    tangent

@is_primitive MinimalCtx Tuple{typeof(exp),Matrix{<:IEEEFloat}}

struct ExpPullback{P}
    pb
    Ybar::Matrix{P}
    Xbar::Matrix{P}
end

function (pb::ExpPullback)(::NoRData)
    _, Xbar_inc = pb.pb(pb.Ybar)
    pb.Xbar .+= Xbar_inc
    return NoRData(), NoRData()
end

# `exp(::Matrix{<:IEEEFloat})` — per-lane ChainRules.frule. The primal Y is
# computed once via `exp(X)`; each lane's dY runs `LinearAlgebra.exp!` (with
# its own X copy) and discards the recomputed Y.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(exp),N}, X_dX::Mooncake.Lifted{Matrix{P},N}
) where {N,P<:IEEEFloat}
    X_primal = primal(X_dX)
    Y = exp(X_primal)
    dys = ntuple(Val(N)) do k
        X = copy(X_primal)
        dX = copy(Mooncake.tangent(X_dX, k))
        _, dY = ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)
        dY
    end
    return Mooncake.Lifted{Matrix{P},N}(Y, Mooncake.NTangent(dys))
end

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

@from_rrule DefaultCtx Tuple{typeof(svd),AbstractMatrix{<:IEEEFloat}}

end
