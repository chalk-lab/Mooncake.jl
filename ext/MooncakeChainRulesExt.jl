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
    Y_partials = ntuple(_ -> similar(Xp), Val(Nw))
    Y_primal = similar(Xp)
    # One `ChainRules.frule` call per lane. ChainRules' matrix-exp frule computes the primal
    # `exp(X)` and the directional derivative together via a single augmented block-matrix
    # exponential, so each lane recomputes `exp(X)` (the dominant cost): it cannot be hoisted out
    # of the loop, as there is no JVP-only path through the ChainRules boundary. `exp!` mutates, so
    # each lane needs a fresh `Xc`; the (lane-independent) `Y_primal` is taken from lane 1, which
    # therefore needs no pre-fill (Nw ≥ 1).
    @inbounds for lane in 1:Nw
        Xc = copy(Xp)
        dXc = copy(tangent(X_dX).partials[lane])
        Y_l, dY_l = ChainRules.frule((ChainRules.NoTangent(), dXc), LinearAlgebra.exp!, Xc)
        if lane == 1
            copyto!(Y_primal, Y_l)
        end
        copyto!(Y_partials[lane], dY_l)
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
