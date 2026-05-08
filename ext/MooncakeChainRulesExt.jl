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

function frule!!(::Dual{typeof(exp)}, X_dX::Dual{Matrix{P}}) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    return Dual(ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)...)
end
# Bare NDual-matrix variant — V at width 1 for Matrix{<:IEEEFloat} is
# Matrix{NDual{P,1}}; extract primal/tangent via element-wise map and re-pack
# the (y, dy) pair into Matrix{NDual} for canonical V output.
function frule!!(
    ::Dual{typeof(exp)}, X_dX::Matrix{Mooncake.Nfwd.NDual{P,1}}
) where {P<:IEEEFloat}
    X = map(d -> d.value, X_dX)
    dX = map(d -> d.partials[1], X_dX)
    Y, dY = ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)
    return map((y, dy) -> Mooncake.Nfwd.NDual{P,1}(y, (dy,)), Y, dY)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(exp),N}, X_dX::Mooncake.Lifted{Matrix{P}}
) where {N,P<:IEEEFloat}
    bare_result = frule!!(Mooncake._unlift(f), Mooncake._unlift(X_dX))
    P_out = Mooncake._typeof(Mooncake.__get_primal(bare_result))
    return Mooncake._wrap_rule_result(P_out, Val(N), bare_result)
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(exp),<:Matrix{<:IEEEFloat}}}) = true

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

@from_rrule DefaultCtx Tuple{typeof(svd),AbstractMatrix{<:IEEEFloat}}

end
