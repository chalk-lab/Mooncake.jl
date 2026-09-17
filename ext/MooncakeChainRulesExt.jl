module MooncakeChainRulesExt

using ChainRules, LinearAlgebra, Mooncake
using Base: IEEEFloat
using LinearAlgebra: BlasFloat, BlasReal

import Mooncake:
    @from_rrule,
    @is_primitive,
    CoDual,
    DefaultCtx,
    Dual,
    MinimalCtx,
    NoRData,
    ReverseMode,
    arrayify,
    frule!!,
    increment_densified_tangent!!,
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

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

@from_rrule DefaultCtx Tuple{typeof(svd),AbstractMatrix{<:IEEEFloat}}

# These spectral rules cover first-order reverse mode without keyword arguments.
@from_rrule DefaultCtx Tuple{typeof(svdvals),Matrix{<:BlasFloat}}

# ChainRules' dense eigvals rule folds independent triangle cotangents at symmetric inputs.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(eigvals),Symmetric{<:BlasReal,<:StridedMatrix}
}

function rrule!!(
    ::CoDual{typeof(eigvals)}, A::CoDual{<:Symmetric{<:BlasReal,<:StridedMatrix}}
)
    y, back = ChainRules.rrule(eigvals, primal(A))
    _, dA = arrayify(A)
    dy = zero(y)
    function eigvals_pullback(::NoRData)
        increment_densified_tangent!!(dA, back(dy)[2])
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), eigvals_pullback
end

end
