module MooncakeChainRulesExt

using ChainRules, LinearAlgebra, Mooncake
using Base: IEEEFloat

import Mooncake:
    @is_primitive,
    CoDual,
    Dual,
    MinimalCtx,
    NTangent,
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

function frule!!(::Dual{typeof(exp)}, X_dX::Dual{Matrix{P}}) where {P<:IEEEFloat}
    X = primal(X_dX)
    dX = tangent(X_dX)
    if dX isa NTangent
        first_X = copy(X)
        Y, first_lane = ChainRules.frule(
            (ChainRules.NoTangent(), copy(dX[1])), LinearAlgebra.exp!, first_X
        )
        lane_tangents = ntuple(Val(length(dX))) do lane
            lane == 1 && return first_lane
            _, lane_dY = ChainRules.frule(
                (ChainRules.NoTangent(), copy(dX[lane])), LinearAlgebra.exp!, copy(X)
            )
            return lane_dY
        end
        return Dual(Y, NTangent(lane_tangents))
    end
    X_copy = copy(X)
    return Dual(
        ChainRules.frule((ChainRules.NoTangent(), copy(dX)), LinearAlgebra.exp!, X_copy)...
    )
end

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

end
