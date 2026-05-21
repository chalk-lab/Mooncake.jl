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

# `exp(::Matrix)` implementation kernel (no `Dual{typeof(F)}` arg).
# Two overloads parallel the original bare-Dual signatures: the Dual-matrix
# variant for the legacy parallel-storage path, and the NDual-matrix variant
# for the canonical-V path at width 1.
@inline function _exp_matrix_kernel(X_dX::Dual{Matrix{P}}) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    return Dual(ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)...)
end
@inline function _exp_matrix_kernel(
    X_dX::Matrix{Mooncake.Nfwd.NDual{P,1}}
) where {P<:IEEEFloat}
    X = map(d -> d.value, X_dX)
    dX = map(d -> d.partials[1], X_dX)
    Y, dY = ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)
    return map((y, dy) -> Mooncake.Nfwd.NDual{P,1}(y, (dy,)), Y, dY)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(exp),N}, X_dX::Mooncake.Lifted{Matrix{P}}
) where {N,P<:IEEEFloat}
    bare_result = _exp_matrix_kernel(Mooncake._unlift(X_dX))
    return Mooncake._wrap_rule_result(Val(N), bare_result)
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(exp),<:Matrix{<:IEEEFloat}}}) = true

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

@from_rrule DefaultCtx Tuple{typeof(svd),AbstractMatrix{<:IEEEFloat}}

end
