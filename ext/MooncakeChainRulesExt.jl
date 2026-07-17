module MooncakeChainRulesExt

using ChainRules, LinearAlgebra, Mooncake
using Base: IEEEFloat

import Mooncake:
    @from_rrule,
    @is_primitive,
    CoDual,
    DefaultCtx,
    Lifted,
    NDual,
    NDualArray,
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

# Per-lane `ChainRules.frule` call. ChainRules expects a Matrix tangent
# per lane; produce both via per-lane partial copy, then rebuild the
# result `NDualArray`.
function frule!!(
    ::Lifted{typeof(exp),Nw},
    X_dX::Lifted{Matrix{P},Nw,NDualArray{P,Nw,2,Matrix{P},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat}
    Xp = primal(X_dX)
    # One `ChainRules.frule` call per lane. ChainRules' matrix-exp frule computes the primal
    # `exp(X)` and the directional derivative together via a single augmented block-matrix
    # exponential, so each lane recomputes `exp(X)` (the dominant cost): it cannot be hoisted out
    # of the loop, as there is no JVP-only path through the ChainRules boundary. `exp!` destroys its
    # input, so `Xc`/`dXc` are reused scratches refilled from `Xp`/the lane's tangent each lane. The
    # frule returns freshly-allocated `Matrix{P}` for both the primal and each partial, so use them
    # directly (no extra `similar`/`copyto!`); the (lane-independent) `Y_primal` is lane 1's (Nw ≥ 1).
    Xc = similar(Xp)
    dXc = similar(Xp)
    copyto!(Xc, Xp)
    copyto!(dXc, tangent(X_dX).partials[1])
    Y_primal, dY_1 = ChainRules.frule((ChainRules.NoTangent(), dXc), LinearAlgebra.exp!, Xc)
    Y_partials = ntuple(Val(Nw)) do lane
        lane == 1 && return dY_1
        copyto!(Xc, Xp)
        copyto!(dXc, tangent(X_dX).partials[lane])
        return ChainRules.frule((ChainRules.NoTangent(), dXc), LinearAlgebra.exp!, Xc)[2]
    end
    return Lifted{Matrix{P},Nw}(
        Y_primal, NDualArray{P,Nw,2,Matrix{P}}(Y_primal, Y_partials)
    )
end

function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ybar = zero(Y)
    return CoDual(Y, Ybar), ExpPullback{P}(pb, Ybar, X.dx)
end

@from_rrule DefaultCtx Tuple{typeof(svd),AbstractMatrix{<:IEEEFloat}}

end
