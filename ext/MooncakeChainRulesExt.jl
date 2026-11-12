module MooncakeChainRulesExt

using ChainRules, LinearAlgebra, Mooncake
using Base: IEEEFloat
using LinearAlgebra: BlasFloat, BlasReal

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
    ReverseMode,
    arrayify,
    frule!!,
    increment_densified_tangent!!,
    primal,
    rrule!!,
    tangent,
    tangent_view

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

function frule!!(
    ::Lifted{typeof(exp),Nw},
    X_dX::Lifted{Matrix{P},Nw,<:NDualArray{P,Nw,2,Matrix{P},NDual{P,Nw}}},
) where {Nw,P<:IEEEFloat}
    Xp = primal(X_dX)
    # ChainRules requires Matrix tangents and computes value/JVP together by augmented
    # matrix exponential, with no JVP-only path: each lane must recompute exp(X). exp!
    # mutates its input: refill both scratches. Keep the fresh outputs without copying,
    # taking the lane-independent primal from lane 1 (Nw ≥ 1).
    Xc = similar(Xp)
    dXc = similar(Xp)
    copyto!(Xc, Xp)
    copyto!(dXc, tangent_view(X_dX, 1))
    Y_primal, dY_1 = ChainRules.frule((ChainRules.NoTangent(), dXc), LinearAlgebra.exp!, Xc)
    Y_partials = ntuple(Val(Nw)) do lane
        lane == 1 && return dY_1
        copyto!(Xc, Xp)
        copyto!(dXc, tangent_view(X_dX, lane))
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
